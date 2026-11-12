"""Level 2 LoRA data parallelism: N trainer clients, one weighted gradient all-reduce per optimizer step.

Every trainer owns a full copy of the LoRA weights and optimizer, reads its own data shard, and borrows frozen
compute from the shared swarm (workers hold no adapter state). Before each optimizer step the trainers average
their LoRA gradients (weighted by tokens) and every rank applies the same optimizer update. hivemind's all-reduce
returns each peer `local + (mean - local)`, so ranks agree only up to float rounding (~1e-7 relative per step,
not bitwise). `check_consistency` therefore compares weights against the cross-rank mean with a tolerance,
raises if the ranks really diverged, and snaps the weights back to the mean so rounding cannot accumulate.

    dp = LoRADataParallel.for_model(model, run_id="exp1", rank=r, world_size=n, initial_peers=peers)
    dp.sync_initial_state()                      # barrier + check that every rank starts from the same weights
    for step in ...:
        dp.begin_round()                         # optional: matchmaking overlaps with forward/backward
        ...forward/backward (+ gradient accumulation)...
        loss = dp.allreduce_gradients(weight=n_tokens, loss=local_loss)
        clip_grad_norm_(...); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        dp.maybe_check_consistency(step, every=50)   # drift check + snap to the cross-rank mean

Synchronisation is strict: if the group of `world_size` trainers cannot form within `timeout`, a
`LoRADataParallelError` is raised (no partial averaging, no silent divergence).
"""

import hashlib
import threading
import time
from typing import Optional, Sequence

import torch

from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class LoRADataParallelError(RuntimeError):
    pass


def shard_rows(num_rows: int, rank: int, world_size: int) -> range:
    """Disjoint, exhaustive split of dataset rows over trainers (round-robin)."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Need 0 <= rank < world_size")
    return range(rank, num_rows, world_size)


def flatten_grads(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """fp32 CPU vector of all gradients (missing gradients count as zeros)."""
    pieces = [
        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().to(device="cpu", dtype=torch.float32).flatten()
        for p in parameters
    ]
    return torch.cat(pieces) if pieces else torch.zeros(0)


def call_with_deadline(fn, seconds: float, what: str):
    """Run fn() in a helper thread and give up after `seconds`.

    hivemind's DecentralizedAverager.step() waits on its worker process without a timeout when handing over a new
    round; if that process wedges, the trainer would hang forever. This turns that into a clear error.
    """
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = error

    worker = threading.Thread(target=target, daemon=True, name=f"lora-dp-{what}")
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise LoRADataParallelError(f"{what} did not respond within {seconds:.0f}s (averager process unresponsive)")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def parameters_digest(parameters: Sequence[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for p in parameters:
        digest.update(p.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


class LoRADataParallel:
    def __init__(
        self,
        parameters: Sequence[torch.nn.Parameter],
        *,
        run_id: str,
        rank: int,
        world_size: int,
        initial_peers: Optional[Sequence[str]] = None,
        dht=None,
        timeout: float = 120.0,
        matchmaking_time: float = 2.0,
    ):
        """
        :param parameters: the trainable LoRA parameters, in the same order on every rank
        :param run_id: all trainers of one run must use the same id (it namespaces the DHT keys)
        :param dht: a running, non-client-mode hivemind DHT; if omitted one is started from `initial_peers`
        :param timeout: seconds to wait for all `world_size` trainers before failing
        :param matchmaking_time: seconds the group waits for stragglers before reducing; lower it for fast steps
        """
        from bloombee.utils.hivemind_compat import DHT

        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Need 0 <= rank < world_size")
        if not run_id or "." in run_id:
            raise ValueError("run_id must be a non-empty string without '.'")
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("No trainable parameters to synchronise")
        self.rank, self.world_size, self.timeout = rank, world_size, timeout
        self._owns_dht = dht is None
        if dht is None:
            if not initial_peers:
                raise ValueError("Provide a DHT or initial_peers")
            dht = DHT(initial_peers=list(initial_peers), start=True, client_mode=False)
        self.dht = dht
        self.numel = sum(p.numel() for p in self.parameters)
        # [all gradients..., loss]: the loss rides along so the logged value is the global token-weighted mean.
        self._buffer = torch.zeros(self.numel + 1, dtype=torch.float32)
        from hivemind.averaging import DecentralizedAverager

        self.averager = DecentralizedAverager(
            [self._buffer],
            dht,
            start=True,
            prefix=f"{run_id}_lora_dp",
            target_group_size=world_size,
            min_group_size=world_size,
            min_matchmaking_time=matchmaking_time,
            request_timeout=min(10.0, timeout),
            allreduce_timeout=timeout,
        )
        self._control = None
        self.rounds = 0

    @classmethod
    def for_model(cls, model, **kwargs):
        """`model` is a DistributedLoRA; its adapter parameters are enumerated in a rank-independent order."""
        return cls(list(model._adapter_state().values()), **kwargs)

    # ------------------------------------------------------------------ rendezvous / consistency
    def _all_gather(self, payload):
        started = time.monotonic()
        try:
            gathered = call_with_deadline(
                lambda: self.averager.step(gather=payload, weight=1.0, timeout=self.timeout),
                self.timeout + 30,
                "trainer rendezvous",
            )
        except Exception as error:
            raise LoRADataParallelError(
                f"Rank {self.rank}: could not assemble {self.world_size} trainers within {self.timeout:.0f}s "
                f"(waited {time.monotonic() - started:.0f}s): {error!r}"
            ) from error
        if gathered is None or len(gathered) != self.world_size:
            raise LoRADataParallelError(
                f"Rank {self.rank}: expected {self.world_size} trainers, group had {0 if gathered is None else len(gathered)}"
            )
        return list(gathered.values())

    def _flat_weights(self):
        return torch.cat([p.detach().to(device="cpu", dtype=torch.float32).flatten() for p in self.parameters])

    def check_consistency(self, tol: float = 1e-4) -> float:
        """Verify all ranks hold (nearly) the same weights, then snap every rank to the cross-rank mean.

        Returns the largest relative deviation seen on any rank. Raises (on every rank) if it exceeds `tol`,
        or if the ranks present are not exactly 0..world_size-1.
        """
        flat = self._flat_weights()
        self._buffer[: self.numel] = flat
        self._buffer[self.numel] = 0.0
        reports = self._all_gather(dict(rank=self.rank))  # also reduces the weights into _buffer (mean)
        ranks = sorted(r["rank"] for r in reports)
        if ranks != list(range(self.world_size)):
            raise LoRADataParallelError(f"Trainer ranks {ranks} do not cover 0..{self.world_size - 1} exactly once")
        mean = self._buffer[: self.numel].clone()
        scale = float(flat.abs().max().clamp_min(1e-12))
        deviation = float((flat - mean).abs().max()) / scale
        # a second round so that *every* rank learns the worst deviation and raises (or continues) together
        worst = max(r["deviation"] for r in self._all_gather(dict(rank=self.rank, deviation=deviation)))
        if worst > tol:
            raise LoRADataParallelError(
                f"LoRA weights differ between trainers (relative deviation {worst:.2e} > {tol:.0e}). Start every "
                "rank from identical weights (same --seed/--init_seed or the same adapter checkpoint) and use "
                "the same optimizer settings."
            )
        offset = 0
        with torch.no_grad():
            for p in self.parameters:
                p.copy_(mean[offset : offset + p.numel()].view(p.shape))
                offset += p.numel()
        return worst

    def sync_initial_state(self, tol: float = 1e-4):
        """Barrier plus a check that every rank starts from the same adapter weights."""
        return self.check_consistency(tol)

    def maybe_check_consistency(self, step: int, every: int, tol: float = 1e-4):
        """Call once per optimizer step with the 0-based step index; checks every `every` steps."""
        if every and (step + 1) % every == 0:
            return self.check_consistency(tol)
        return None

    # ------------------------------------------------------------------ gradient all-reduce
    def begin_round(self):
        """Start looking for the other trainers now, so matchmaking overlaps with this step's compute."""
        if self._control is None:
            self._control = call_with_deadline(
                lambda: self.averager.step(weight=1.0, timeout=self.timeout, wait=False, require_trigger=True),
                self.timeout + 30,
                "matchmaking start",
            )

    def allreduce_gradients(self, weight: float, loss: float = 0.0) -> float:
        """Replace every parameter's .grad with the token-weighted mean over trainers; returns the global mean loss.

        :param weight: this trainer's number of supervised tokens in the step (must be > 0)
        """
        if not weight > 0:
            raise ValueError("weight must be positive")
        if self._control is None:
            self.begin_round()
        control, self._control = self._control, None
        self._buffer[: self.numel] = flatten_grads(self.parameters)
        self._buffer[self.numel] = float(loss)
        control.weight = float(weight)
        control.allow_allreduce()  # tensors are only read after this point
        try:
            gathered = control.result(timeout=self.timeout)
        except Exception as error:
            raise LoRADataParallelError(
                f"Rank {self.rank}: gradient all-reduce failed (a trainer is missing, slow or crashed?): {error!r}"
            ) from error
        if gathered is None or len(gathered) != self.world_size:
            raise LoRADataParallelError(f"Rank {self.rank}: all-reduce group size != {self.world_size}")
        offset = 0
        for p in self.parameters:
            chunk = self._buffer[offset : offset + p.numel()].view(p.shape)
            if p.grad is None:
                p.grad = chunk.to(device=p.device, dtype=p.dtype).clone()
            else:
                p.grad.copy_(chunk)
            offset += p.numel()
        self.rounds += 1
        return float(self._buffer[self.numel])

    def shutdown(self):
        if self._control is not None:
            self._control.cancel()
            self._control = None
        self.averager.shutdown()
        if self._owns_dht:
            self.dht.shutdown()
