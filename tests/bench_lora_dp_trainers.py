"""Level 1 vs Level 2 throughput on a private swarm (idle GPUs only).

  PYTHONPATH=src:tests python tests/bench_lora_dp_trainers.py --checkpoint-dir CKPT --gpus 2,2 \
      --client-gpu 2 --layers 8 --vocab 32000 --seq 512 --batch 8 --mbs 4 --output OUT

Same global batch two ways: ONE trainer with 2*batch rows (Level 1 only: micro-batches over replicas) versus
TWO trainers with `batch` rows each plus a gradient all-reduce per step (Level 2). Also the one-trainer
`batch`-row run as a reference. Reports global tokens/s (median step time, warmup excluded).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from lora_swarm import Swarm, assert_gpus_idle

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="existing random Llama checkpoint")
    parser.add_argument("--gpus", default="0,0")
    parser.add_argument("--client-gpu", type=int, default=None)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8, help="rows per trainer per step")
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    client_gpu = args.client_gpu if args.client_gpu is not None else gpus[0]
    assert_gpus_idle(sorted(set(gpus + [client_gpu])))
    args.output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(client_gpu), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
    swarm = Swarm(args.checkpoint_dir, args.layers, [(g, 0, args.layers) for g in gpus], args.output)
    report = dict(gpus=gpus, seq=args.seq, per_trainer_batch=args.batch, micro_batch_size=args.mbs, runs={})
    worker = [sys.executable, str(ROOT / "tests" / "lora_dp_worker.py")]
    try:
        swarm.start(timeout=600)
        common = ["--model", str(args.checkpoint_dir.resolve()), "--prefix", swarm.prefix, "--peers", *swarm.peers,
                  "--steps", str(args.steps), "--seq", str(args.seq), "--vocab", str(args.vocab),
                  "--rows", str(8 * args.batch), "--micro_batch_size", str(args.mbs)]  # fmt: skip

        def run(name, mode, world, batch):
            outs, procs = [], []
            for rank in range(world):
                out = args.output / f"{name}-{rank}.pt"
                cmd = worker + ["--mode", mode, "--rank", str(rank), "--world_size", str(world), "--run_id", name,
                                "--batch", str(batch), "--out", str(out)] + common  # fmt: skip
                if mode == "dp":
                    cmd.append("--overlap")  # matchmaking runs during forward/backward
                procs.append(subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=(args.output / f"{name}-{rank}.log").open("w"),
                                              stderr=subprocess.STDOUT))  # fmt: skip
                outs.append(out)
            for p in procs:
                p.wait(timeout=1800)
            assert all(p.returncode == 0 for p in procs), f"{name} failed; see {name}-*.log"
            times = [torch.load(o)["step_times"][args.warmup :] for o in outs]
            step = statistics.median(statistics.median(t) for t in times)
            tokens = world * batch * args.seq
            report["runs"][name] = dict(
                trainers=world, rows_per_trainer=batch, global_rows=world * batch,
                step_time_s=round(step, 3), global_tokens_per_s=round(tokens / step, 1),
            )  # fmt: skip
            print(name, report["runs"][name], flush=True)

        run("one_trainer_half_batch", "ref", 1, args.batch)
        run("one_trainer_full_batch", "ref", 1, 2 * args.batch)
        run("two_trainers_dp", "dp", 2, args.batch)
        full, dp = report["runs"]["one_trainer_full_batch"], report["runs"]["two_trainers_dp"]
        report["dp_speedup_vs_one_trainer_same_global_batch"] = round(
            dp["global_tokens_per_s"] / full["global_tokens_per_s"], 3
        )
        report["status"] = "done"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        swarm.stop()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
