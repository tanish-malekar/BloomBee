"""Throughput benchmark for LoRA training: single-chain (old behaviour) vs concurrent micro-batches.

Starts a private DHT and local workers on the GPUs you name, trains a random Llama checkpoint with remote LoRA
and reports tokens/s plus a gradient-equivalence check across micro-batch sizes. Nothing is downloaded.

  python benchmarks/benchmark_lora_dp.py --gpus 0,2 --topology full --replicas 2 --mbs none,1,2,4 --output out/

Topologies (the servers are the only thing that differs):
  full   every worker holds all layers; --replicas workers, one per GPU in --gpus  (data-parallel replicas)
  split  one pipeline: the layers are cut evenly over the GPUs in --gpus           (pipeline parallel)

Run the same command with PYTHONPATH pointing at an older checkout to get the pre-change baseline: the script
detects a client without `micro_batch_size` support and benchmarks it once, labelled "old".

Safety: workers are only launched on GPUs that are idle (little memory used, no compute processes, low
utilisation). The check runs again right before every launch; otherwise the script refuses to start.
"""

import argparse
import inspect
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def nvidia_smi(query, section="gpu"):
    out = subprocess.run(
        ["nvidia-smi", f"--query-{section}={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[cell.strip() for cell in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def assert_gpus_idle(gpus, max_mem_mib=1000, max_util=10):
    """Refuse to use a GPU anyone else is using. Called again immediately before each worker launch."""
    rows = {int(r[0]): r for r in nvidia_smi("index,uuid,memory.used,utilization.gpu")}
    busy_uuids = {r[0] for r in nvidia_smi("gpu_uuid,pid", section="compute-apps")}
    for gpu in gpus:
        if gpu not in rows:
            raise RuntimeError(f"GPU {gpu} does not exist on this machine")
        _, uuid_, mem, util = rows[gpu]
        if int(mem) > max_mem_mib or int(util) > max_util or uuid_ in busy_uuids:
            raise RuntimeError(f"GPU {gpu} is not idle (mem={mem} MiB, util={util}%, compute apps={uuid_ in busy_uuids})")


# The client must see only its own GPU: fix this before torch is imported.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpus", default="0")
_pre.add_argument("--client-gpu", type=int, default=None)
_known, _ = _pre.parse_known_args()
_GPUS = [int(g) for g in _known.gpus.split(",")]
_CLIENT_GPU = _known.client_gpu if _known.client_gpu is not None else _GPUS[0]
if "--help" not in sys.argv and "-h" not in sys.argv:
    assert_gpus_idle(sorted(set(_GPUS + [_CLIENT_GPU])))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_CLIENT_GPU)

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig  # noqa: E402
from bloombee.data_structures import ServerState  # noqa: E402
from bloombee.utils.dht import get_remote_module_infos  # noqa: E402
from bloombee.utils.hivemind_compat import DHT  # noqa: E402


def make_checkpoint(path, args):
    marker = path / "bench_config.json"
    spec = dict(layers=args.layers, hidden=args.hidden, vocab=args.vocab)
    if marker.exists() and json.loads(marker.read_text()) == spec:
        return
    torch.manual_seed(173)
    config = LlamaConfig(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 11 // 4 // 64 * 64,
        num_hidden_layers=args.layers,
        num_attention_heads=args.hidden // 128,
        num_key_value_heads=args.hidden // 128,
        max_position_embeddings=args.seq * 2,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).half().save_pretrained(path)
    marker.write_text(json.dumps(spec))


class Swarm:
    def __init__(self, args, base_dir, out):
        self.args, self.base_dir, self.out = args, base_dir, out
        self.prefix = "lorabench_" + uuid.uuid4().hex[:10]
        self.dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
        self.peers = [str(a) for a in self.dht.get_visible_maddrs()]
        self.processes, self.logs = [], []

    def plan(self):
        gpus, layers = _GPUS, self.args.layers
        if self.args.topology == "full":
            return [(gpu, 0, layers) for gpu in gpus[: self.args.replicas]]
        cuts = [round(i * layers / len(gpus)) for i in range(len(gpus) + 1)]
        return [(gpu, cuts[i], cuts[i + 1]) for i, gpu in enumerate(gpus)]

    def start(self):
        plan = self.plan()
        checked = set()
        for index, (gpu, start, end) in enumerate(plan):
            if gpu not in checked:  # a GPU may host several of our own workers (e.g. --gpus 2,2)
                assert_gpus_idle([gpu])  # re-check immediately before first using the GPU
                checked.add(gpu)
            log = (self.out / f"worker-{index}.log").open("w")
            self.logs.append(log)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "BLOOMBEE_ENABLE_LORA": "1",
                "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
                "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE": "0",
                "BLOOMBEE_LOSSLESS_WRAPPER": "0",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "HF_HUB_OFFLINE": "1",
            }
            command = [
                sys.executable, "-m", "bloombee.cli.run_server", str(self.base_dir.resolve()),
                "--block_indices", f"{start}:{end}",
                "--dht_prefix", self.prefix,
                "--num_handlers", "1",
                "--batch_size", "2",
                "--torch_dtype", "float16",
                "--throughput", "100",
                "--update_period", "1",
                "--expiration", "30",
                "--inference_max_length", "64",
                "--attn_cache_tokens", "128",
                "--max_batch_size", "65536",
                "--balance_quality", "0",
                "--no_auto_relay",
                "--host_maddrs", "/ip4/127.0.0.1/tcp/0",
                "--skip_reachability_check",
                "--cache_gpu_percent", "100",
                "--w_gpu_percent", "100",
                "--initial_peers", *self.peers,
            ]  # fmt: skip
            self.processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        uids = [f"{self.prefix}.{i}" for i in range(self.args.layers)]
        expected = [sum(start <= i < end for _, start, end in plan) for i in range(self.args.layers)]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in self.processes):
                raise RuntimeError(f"A worker exited early; see {self.out}/worker-*.log")
            infos = get_remote_module_infos(self.dht, uids, latest=True)
            online = [sum(s.state == ServerState.ONLINE for s in item.servers.values()) for item in infos]
            if all(o >= e for o, e in zip(online, expected)):
                return plan
            time.sleep(1)
        raise TimeoutError("Workers did not come online in 10 minutes")

    def stop(self):
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in self.processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in self.logs:
            log.close()
        self.dht.shutdown()


class GpuSampler(threading.Thread):
    """Mean utilisation of the benchmark's own GPUs while a measurement runs."""

    def __init__(self, gpus):
        super().__init__(daemon=True)
        self.gpus, self.samples, self.stop_flag = gpus, {g: [] for g in gpus}, threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                for row in nvidia_smi("index,utilization.gpu"):
                    if int(row[0]) in self.samples:
                        self.samples[int(row[0])].append(int(row[1]))
            except Exception:
                pass
            time.sleep(0.5)

    def finish(self):
        self.stop_flag.set()
        self.join()
        return {g: round(statistics.mean(v), 1) if v else None for g, v in self.samples.items()}


def flat_grads(model):
    return torch.cat([p.grad.detach().float().flatten().cpu() for p in model._adapter_state().values()])


def fp32_truth_grads(lora, base_dir, ids, out_dir):
    """Gradients of the same adapter on a local fp32 HuggingFace+PEFT model (same fp16 base weights, upcast)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_dir = out_dir / "adapter-for-truth"
    lora.save_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="eager").cuda()
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True).train()
    for name, parameter in model.named_parameters():
        if ".lora_" in name:
            parameter.data = parameter.data.float()
    model(ids, labels=ids, use_cache=False).loss.backward()
    named = dict(model.named_parameters())
    flat = torch.cat(
        [
            named[key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight")].grad.detach().float().flatten().cpu()
            for key in lora._adapter_state()
        ]
    )
    del model, base
    torch.cuda.empty_cache()
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="comma-separated idle GPU ids to host workers")
    parser.add_argument("--client-gpu", type=int, default=None, help="GPU for the client (default: first of --gpus)")
    parser.add_argument("--topology", choices=("full", "split"), default="full")
    parser.add_argument("--replicas", type=int, default=1, help="full topology: number of worker replicas")
    parser.add_argument("--mbs", default="none,1,2,4", help="micro-batch sizes to compare ('none' = one chain)")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--label", default=None, help="name for this run in the report")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="reuse a generated random checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_dir = args.checkpoint_dir or args.output / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    make_checkpoint(base_dir, args)

    supports_mbs = "micro_batch_size" in inspect.signature(DistributedLoRA.__init__).parameters
    sizes = [None if m == "none" else int(m) for m in args.mbs.split(",")]
    if not supports_mbs:
        sizes = [None]
    report = dict(
        label=args.label or ("new" if supports_mbs else "old"),
        client_supports_micro_batches=supports_mbs,
        topology=args.topology,
        replicas=args.replicas if args.topology == "full" else 1,
        gpus=_GPUS,
        model=dict(layers=args.layers, hidden=args.hidden, vocab=args.vocab),
        batch=args.batch,
        seq=args.seq,
        tokens_per_step=args.batch * args.seq,
        device_names=[r[1] for r in nvidia_smi("index,name")],
        results=[],
        status="running",
    )
    swarm = Swarm(args, base_dir, args.output)
    try:
        plan = swarm.start()
        report["plan"] = [dict(gpu=g, blocks=f"{s}:{e}") for g, s, e in plan]
        print("workers online:", report["plan"], flush=True)
        client = AutoDistributedModelForCausalLM.from_pretrained(
            str(base_dir.resolve()),
            dht_prefix=swarm.prefix,
            initial_peers=swarm.peers,
            torch_dtype=torch.float16,
            connect_timeout=30,
            request_timeout=600,
            max_retries=2,
            update_period=1,
        ).cuda()
        torch.manual_seed(0)
        lora = DistributedLoRA(client, LoRAConfig(r=args.r, lora_alpha=16, target_modules=("q_proj", "v_proj")))
        lora.cuda().train()
        with torch.no_grad():  # nonzero B so that every adapter matrix receives gradient
            for name, p in lora.named_parameters():
                if p.requires_grad and "lora_B" in name:
                    p.normal_(0, 0.02)
        ids = torch.randint(0, args.vocab, (args.batch, args.seq), generator=torch.Generator().manual_seed(1)).cuda()
        params = [p for p in lora.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-5)
        layers_module = lora.remote_layers

        def set_mbs(size):
            if supports_mbs:
                layers_module.micro_batch_size = size

        # 1) gradient equivalence on one fixed batch (dropout is 0, so the result is deterministic up to fp16 noise)
        reference, all_grads = None, []
        for size in sizes:
            set_mbs(size)
            optimizer.zero_grad(set_to_none=True)
            lora(ids, labels=ids).loss.backward()
            grads = flat_grads(lora)
            all_grads.append(grads)
            reference = grads if reference is None else reference
            rel = float((grads - reference).norm() / reference.norm().clamp_min(1e-12))
            entry = dict(micro_batch_size=size, grad_rel_diff_vs_first=rel)
            if size is not None:
                # Same chunks, same kernels, but run one after another through the single-chain path.
                # The concurrent path must match this almost exactly; any gap vs the full batch is then
                # fp16 batch-size sensitivity of the workers, not the scheduling.
                set_mbs(None)
                optimizer.zero_grad(set_to_none=True)
                chunks = ids.split(size)
                for chunk in chunks:
                    (lora(chunk, labels=chunk).loss / len(chunks)).backward()
                manual = flat_grads(lora)
                entry["grad_rel_diff_vs_manual_accumulation"] = float(
                    (grads - manual).norm() / manual.norm().clamp_min(1e-12)
                )
                entry["manual_accumulation_rel_diff_vs_full_batch"] = float(
                    (manual - reference).norm() / reference.norm().clamp_min(1e-12)
                )
                set_mbs(size)
            report["results"].append(entry)
        optimizer.zero_grad(set_to_none=True)
        truth = fp32_truth_grads(lora, base_dir, ids, args.output)
        for entry, grads in zip(report["results"], all_grads):
            entry["grad_rel_err_vs_fp32_truth"] = float((grads - truth).norm() / truth.norm().clamp_min(1e-12))

        # 2) throughput
        for entry, size in zip(report["results"], sizes):
            set_mbs(size)
            times = []
            sampler = None
            for step in range(args.warmup + args.steps):
                if step == args.warmup:
                    sampler = GpuSampler(sorted(set(g for g, _, _ in plan)))
                    sampler.start()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                loss = lora(ids, labels=ids).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if step >= args.warmup:
                    times.append(time.perf_counter() - begin)
            util = sampler.finish()
            routes = getattr(layers_module, "_runner", None)
            routes = routes.last_routes if routes is not None else []
            entry.update(
                step_time_s_median=statistics.median(times),
                step_time_s_all=[round(t, 4) for t in times],
                tokens_per_s=args.batch * args.seq / statistics.median(times),
                server_gpu_util_mean_pct=util,
                distinct_peers_in_last_step=len({peer for route in routes for peer, _, _ in route}),
                final_loss=float(loss),
            )
            print(json.dumps(entry), flush=True)
        report["status"] = "done"
        layers_module.sequence_manager.shutdown()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        swarm.stop()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
