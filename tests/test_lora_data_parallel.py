"""Level 1 LoRA data parallelism: concurrent micro-batches over replica routes (no network, no GPU).

A fake swarm stands in for `_call_lora`. It has two replicas per layer with *different* span boundaries, and
blocks that apply seeded LoRA dropout, so these tests exercise the same seed/route contract as real workers:
the dropout mask of absolute layer L is always (micro-batch seed + L), whatever the span boundaries.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from bloombee.client import lora as client

NUM_LAYERS, HIDDEN, RANK, BATCH = 4, 6, 2, 4
LAYOUT = {"X": [(0, 2), (2, 4)], "Y": [(0, 3), (3, 4)]}  # different boundaries on purpose


def block(x, a, b, seed, dropout):
    h = x
    if dropout:
        generator = torch.Generator().manual_seed(seed)
        h = h * (torch.rand(h.shape, generator=generator) >= dropout) / (1.0 - dropout)
    return torch.tanh(x + F.linear(F.linear(h, a), b))


class FakeSwarm:
    """Duck-types the parts of RemoteSequenceManager that _LoRARunner uses."""

    def __init__(self, delay=0.0, max_retries=None):
        self.config = SimpleNamespace(max_retries=max_retries)
        self.delay, self.banned = delay, set()
        self.fail_once, self.dead_backward = set(), set()
        self.active = self.peak = 0
        self.calls = []

    def make_sequence(self, start, end, *, mode, peer_load=None):
        assert mode == "max_throughput"
        load, spans, i = peer_load or {}, [], start
        while i < end:
            candidates = [
                (name, s, e)
                for name, layout in LAYOUT.items()
                if name not in self.banned
                for s, e in layout
                if s <= i < e
            ]
            name, _, e = min(candidates, key=lambda c: (load.get(c[0], 0), c[0]))
            spans.append(SimpleNamespace(peer_id=name, start=i, end=e))
            i = e
        return spans

    def on_request_failure(self, peer):
        self.banned.add(peer)

    def on_request_success(self, peer):
        pass

    def get_retry_delay(self, attempt_no):
        return 0

    async def run(self, span, tensors, metadata):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            backward = metadata["backward"]
            self.calls.append((span.peer_id, span.start, span.end, backward))
            if (span.peer_id, backward) in self.fail_once:
                self.fail_once.remove((span.peer_id, backward))
                raise ConnectionError("injected failure")
            if backward and span.peer_id in self.dead_backward:
                raise ConnectionError("injected dead backward")
            n = span.end - span.start
            x, params = tensors[0], tensors[1 : 1 + 2 * n]
            with torch.set_grad_enabled(backward):
                x = x.detach().requires_grad_(backward)
                params = [p.detach().requires_grad_(backward) for p in params]
                h = x
                for j in range(n):
                    h = block(h, params[2 * j], params[2 * j + 1], metadata["seed"] + j, metadata["dropout"])
                if not backward:
                    return [h.detach()]
                return list(torch.autograd.grad(h, (x, *params), tensors[1 + 2 * n]))
        finally:
            self.active -= 1


@pytest.fixture
def swarm(monkeypatch):
    swarm = FakeSwarm()

    async def fake_call(manager, span, tensors, metadata):
        return await manager.run(span, tensors, metadata)

    monkeypatch.setattr(client, "_call_lora", fake_call)
    return swarm


def make_params():
    torch.manual_seed(5)
    return [
        torch.randn(*shape) * 0.5 for _ in range(NUM_LAYERS) for shape in ((RANK, HIDDEN), (HIDDEN, RANK))
    ]


def run_remote(swarm, params, x, micro_batch_size, dropout=0.0, seed=0):
    runner = client._LoRARunner(swarm, num_layers=NUM_LAYERS, width=2)
    torch.manual_seed(seed)  # fixes the per-micro-batch seeds
    ps = [p.clone().requires_grad_() for p in params]
    xin = x.clone().requires_grad_()
    y = client._RemoteLoRASequenceFunction.apply(xin, runner, micro_batch_size, dict(dropout=dropout), *ps)
    y.square().mean().backward()
    return y.detach(), xin.grad, [p.grad for p in ps], runner


def test_micro_batched_gradients_match_local_reference_and_whole_batch(swarm):
    params, x = make_params(), torch.randn(BATCH, 3, HIDDEN)
    ps = [p.clone().requires_grad_() for p in params]
    xin = x.clone().requires_grad_()
    h = xin
    for layer in range(NUM_LAYERS):
        h = block(h, ps[2 * layer], ps[2 * layer + 1], 0, 0.0)
    h.square().mean().backward()

    for mbs in (None, 1, 3):
        y, dx, grads, _ = run_remote(swarm, params, x, mbs)
        torch.testing.assert_close(y, h.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dx, xin.grad, rtol=1e-4, atol=1e-7)
        for got, want in zip(grads, [p.grad for p in ps]):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-7)


def test_micro_batches_run_concurrently_on_both_replicas(swarm):
    swarm.delay = 0.05
    _, _, _, runner = run_remote(swarm, make_params(), torch.randn(BATCH, 3, HIDDEN), micro_batch_size=1)
    assert swarm.peak >= 2
    assert len(runner.last_routes) == BATCH
    assert {peer for route in runner.last_routes for peer, _, _ in route} == {"X", "Y"}
    assert all(runner.load[peer] == 0 for peer in runner.load)  # in-flight counters always return to zero


def test_backward_reroute_onto_other_boundaries_keeps_dropout_masks_exact(swarm):
    params, x = make_params(), torch.randn(BATCH, 3, HIDDEN)
    clean = run_remote(swarm, params, x, None, dropout=0.3, seed=7)
    assert {peer for peer, *_ in swarm.calls} == {"X"}

    rerouted = FakeSwarm()
    rerouted.dead_backward = {"X"}  # every backward through X's boundaries fails -> Y's boundaries
    broken = run_remote(rerouted, params, x, None, dropout=0.3, seed=7)
    assert {(s, e) for peer, s, e, backward in rerouted.calls if backward and peer == "Y"} >= {(0, 2), (2, 3), (3, 4)}
    torch.testing.assert_close(broken[0], clean[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(broken[1], clean[1], rtol=0, atol=1e-6)
    for got, want in zip(broken[2], clean[2]):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)
    # Dropout was really active: a different seed changes the gradients.
    other = run_remote(FakeSwarm(), params, x, None, dropout=0.3, seed=8)
    assert not torch.allclose(other[2][0], clean[2][0])


def test_forward_failure_retries_on_other_replica(swarm):
    params, x = make_params(), torch.randn(BATCH, 3, HIDDEN)
    expected = run_remote(FakeSwarm(), params, x, None)
    swarm.fail_once = {("X", False)}
    got = run_remote(swarm, params, x, None)
    assert "X" in swarm.banned and any(peer == "Y" for peer, *_ in swarm.calls)
    torch.testing.assert_close(got[0], expected[0], rtol=0, atol=1e-6)
    for a, b in zip(got[2], expected[2]):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_error_propagates_when_retries_are_exhausted(swarm):
    swarm.config.max_retries = 1
    swarm.fail_once = {("X", False)}
    with pytest.raises(ConnectionError):
        run_remote(swarm, make_params(), torch.randn(BATCH, 3, HIDDEN), None)


def test_micro_batch_size_validation():
    client._check_micro_batch_size(None)
    client._check_micro_batch_size(3)
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="micro_batch_size"):
            client._check_micro_batch_size(bad)


def test_peer_load_steers_max_throughput_routing():
    from bloombee.client.routing.sequence_manager import RemoteSequenceManager
    from bloombee.data_structures import RemoteSpanInfo

    spans = [RemoteSpanInfo("a", 0, 2, None), RemoteSpanInfo("b", 0, 2, None)]
    stub = SimpleNamespace(
        ping_aggregator=SimpleNamespace(to_dict=lambda: {}),
        state=SimpleNamespace(sequence_info=SimpleNamespace(spans_containing_block=[spans, spans])),
    )

    def share_a(load):
        picks = [
            RemoteSequenceManager._make_sequence_with_max_throughput(stub, 0, 2, peer_load=load)[0].peer_id
            for _ in range(400)
        ]
        return picks.count("a") / len(picks)

    np.random.seed(0)
    assert 0.4 < share_a(None) < 0.6  # default behaviour unchanged
    assert 0.4 < share_a({}) < 0.6
    assert share_a({"a": 9}) < 0.2  # weight 2/10 vs 2 -> about 9%
    assert share_a({"a": 3, "b": 3}) > 0.35  # equal load cancels out


# ---------------------------------------------------------------------------------------------------------
# Level 2: N trainers, weighted gradient all-reduce (real hivemind DHT + averagers, CPU only)
# ---------------------------------------------------------------------------------------------------------
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

if sys.platform == "darwin":  # hivemind's averager process assumes fork (the Linux default)
    multiprocessing.set_start_method("fork", force=True)

from bloombee.client.lora_data_parallel import (  # noqa: E402
    LoRADataParallel,
    LoRADataParallelError,
    call_with_deadline,
    flatten_grads,
    parameters_digest,
    shard_rows,
)
from bloombee.utils.hivemind_compat import DHT  # noqa: E402

SHAPES = [(2, 6), (6, 2), (3, 5)]


def make_trainer_params(seed):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s)) for s in SHAPES]


# hivemind's averager is a forked multiprocessing.Process and assumes Linux. On macOS multi-peer rounds
# occasionally wedge inside multiprocessing's resource sharer, so the multi-peer tests run on Linux/CI
# (set BLOOMBEE_TEST_HIVEMIND_ON_MAC=1 to try them locally anyway).
needs_linux_hivemind = pytest.mark.skipif(
    sys.platform == "darwin" and not __import__("os").environ.get("BLOOMBEE_TEST_HIVEMIND_ON_MAC"),
    reason="hivemind averager processes are only reliable on Linux",
)


def test_call_with_deadline():
    assert call_with_deadline(lambda: 7, 5, "x") == 7
    with pytest.raises(KeyError):
        call_with_deadline(lambda: {}["missing"], 5, "x")  # errors propagate unchanged
    release = threading.Event()
    with pytest.raises(LoRADataParallelError, match="did not respond within 1s"):
        call_with_deadline(release.wait, 1, "stuck call")
    release.set()


@pytest.fixture
def dhts():
    created = []

    def make(n):
        root = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
        created.append(root)
        peers = root.get_visible_maddrs()
        for _ in range(n - 1):
            created.append(
                DHT(initial_peers=peers, start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
            )
        return list(created)

    yield make
    for dht in created:
        dht.shutdown()


def run_ranks(trainers, fn):
    """Run fn(rank, dp) for every trainer in its own thread; re-raise the first error."""
    results, errors = [None] * len(trainers), []

    def target(rank, dp):
        try:
            results[rank] = fn(rank, dp)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=target, args=(r, dp)) for r, dp in enumerate(trainers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    if errors:
        raise errors[0]
    return results


def test_shard_rows_is_disjoint_and_exhaustive():
    for n, world in ((10, 3), (7, 7), (3, 5), (1, 1)):
        parts = [list(shard_rows(n, r, world)) for r in range(world)]
        assert sorted(sum(parts, [])) == list(range(n))
        assert len({i for part in parts for i in part}) == n
    with pytest.raises(ValueError):
        shard_rows(5, 2, 2)


def test_flatten_grads_and_digest():
    params = make_trainer_params(1)
    params[0].grad = torch.ones_like(params[0])
    params[2].grad = torch.full_like(params[2], 2.0)  # params[1] has no grad -> zeros
    flat = flatten_grads(params)
    assert flat.shape == (sum(p.numel() for p in params),)
    assert flat[: params[0].numel()].eq(1).all() and flat[params[0].numel() : params[0].numel() + 12].eq(0).all()
    before = parameters_digest(params)
    assert before == parameters_digest([torch.nn.Parameter(p.detach().clone()) for p in params])
    with torch.no_grad():
        params[1].add_(1e-6)
    assert parameters_digest(params) != before


@needs_linux_hivemind
@pytest.mark.parametrize("world_size", [2, 3])
def test_weighted_gradient_allreduce_is_exact_and_identical_on_all_ranks(dhts, world_size):
    nodes = dhts(world_size)
    trainers = [
        LoRADataParallel(
            make_trainer_params(0),
            run_id="t1",
            rank=r,
            world_size=world_size,
            dht=nodes[r],
            timeout=60,
            matchmaking_time=0.5,
        )
        for r in range(world_size)
    ]
    weights = [3.0, 1.0, 2.0][:world_size]
    grads = [[torch.randn(*s) for s in SHAPES] for _ in range(world_size)]
    losses = [float(r + 1) for r in range(world_size)]

    def step(rank, dp):
        dp.sync_initial_state()
        for p, g in zip(dp.parameters, grads[rank]):
            p.grad = g.clone()
        if rank % 2 == 0:
            dp.begin_round()  # both the overlapped and the blocking path must work in one group
        mean_loss = dp.allreduce_gradients(weights[rank], loss=losses[rank])
        return mean_loss, [p.grad.clone() for p in dp.parameters]

    try:
        results = run_ranks(trainers, step)
    finally:
        for t in trainers:
            t.shutdown()
    total = sum(weights)
    expected = [sum(w * g[i] for w, g in zip(weights, grads)) / total for i in range(len(SHAPES))]
    expected_loss = sum(w * l for w, l in zip(weights, losses)) / total
    for mean_loss, got in results:
        assert abs(mean_loss - expected_loss) < 1e-5
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for other in results[1:]:  # ranks agree up to float rounding (hivemind returns local + (mean - local))
        for a, b in zip(results[0][1], other[1]):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


@needs_linux_hivemind
def test_identical_ranks_take_identical_optimizer_steps_over_several_rounds(dhts):
    nodes = dhts(2)
    trainers = [
        LoRADataParallel(
            make_trainer_params(7), run_id="t2", rank=r, world_size=2, dht=nodes[r], timeout=60, matchmaking_time=0.5
        )
        for r in range(2)
    ]
    opts = [torch.optim.AdamW(t.parameters, lr=0.1) for t in trainers]

    def train(rank, dp):
        dp.sync_initial_state()
        for step in range(3):
            for p in dp.parameters:
                p.grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(100 * rank + step))
            dp.begin_round()
            dp.allreduce_gradients(weight=1.0 + rank)
            opts[rank].step()
            opts[rank].zero_grad(set_to_none=True)
            deviation = dp.maybe_check_consistency(step, every=1)  # raises if the weights ever diverge
            assert deviation < 1e-5  # only float rounding between ranks
        return [p.detach().clone() for p in dp.parameters]

    try:
        weights = run_ranks(trainers, train)
    finally:
        for t in trainers:
            t.shutdown()
    for a, b in zip(*weights):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    # and they really trained (weights moved away from the common start)
    assert not torch.allclose(weights[0][0], make_trainer_params(7)[0].detach())


@needs_linux_hivemind
def test_initial_state_mismatch_is_detected(dhts):
    nodes = dhts(2)
    trainers = [
        LoRADataParallel(
            make_trainer_params(r), run_id="t3", rank=r, world_size=2, dht=nodes[r], timeout=60, matchmaking_time=0.5
        )
        for r in range(2)  # different seeds -> different weights
    ]
    try:
        with pytest.raises(LoRADataParallelError, match="differ"):
            run_ranks(trainers, lambda rank, dp: dp.sync_initial_state())  # every rank raises, none hangs
    finally:
        for t in trainers:
            t.shutdown()


@needs_linux_hivemind
def test_missing_trainer_fails_fast_with_a_clear_error(dhts):
    (node,) = dhts(1)
    trainer = LoRADataParallel(
        make_trainer_params(0), run_id="t4", rank=0, world_size=2, dht=node, timeout=8, matchmaking_time=0.5
    )
    for p in trainer.parameters:
        p.grad = torch.zeros_like(p)
    try:
        with pytest.raises(LoRADataParallelError, match="all-reduce failed|could not assemble"):
            trainer.allreduce_gradients(weight=1.0)
    finally:
        trainer.shutdown()


def test_invalid_arguments():
    params = make_trainer_params(0)
    for kwargs in (dict(rank=2, world_size=2), dict(rank=0, world_size=0)):
        with pytest.raises(ValueError):
            LoRADataParallel(params, run_id="x", dht=object(), **kwargs)
    with pytest.raises(ValueError, match="run_id"):
        LoRADataParallel(params, run_id="a.b", rank=0, world_size=1, dht=object())
    with pytest.raises(ValueError, match="initial_peers"):
        LoRADataParallel(params, run_id="x", rank=0, world_size=1)
