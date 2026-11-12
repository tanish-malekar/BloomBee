"""Level 2 integration test: private swarm on idle GPUs + real data-parallel LoRA trainers.

  PYTHONPATH=src python tests/run_lora_dp.py --gpus 2,2 --client-gpu 2 --output /path/to/out

--gpus lists the GPU of each worker replica (repeat an id to run several replicas on one GPU). Only idle GPUs
are used; the check is repeated right before launch. Tiny random Llama (2 layers), never downloads weights.

Checks
  A. exactness : 2 DP trainers (1 row each, all-reduced) == 1 reference trainer accumulating both rows
                 -> same step-0 gradients, same weights after N AdamW steps, ranks agree with each other.
  B. CLI+resume: examples/train_lora.py with --world_size 2; interrupted+resumed run == uninterrupted run.
  C. failure   : a trainer killed mid-run makes the surviving trainer exit with a clear error (no hang).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from lora_swarm import Swarm, assert_gpus_idle
from run_lora_rpc import save_tiny_tokenizer
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent


def spawn(cmd, log, env):
    return subprocess.Popen(
        cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True
    )


def wait_all(procs, timeout):
    deadline = time.monotonic() + timeout
    for p in procs:
        p.wait(timeout=max(1, deadline - time.monotonic()))
    return [p.returncode for p in procs]


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,0", help="GPU of each worker replica, e.g. 2,2")
    parser.add_argument("--client-gpu", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    client_gpu = args.client_gpu if args.client_gpu is not None else gpus[0]
    assert_gpus_idle(sorted(set(gpus + [client_gpu])))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    base = out / "base"
    torch.manual_seed(173)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
    )  # fmt: skip
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).half().save_pretrained(base)
    save_tiny_tokenizer(base)

    report = dict(gpus=gpus, client_gpu=client_gpu, status="running")
    swarm = Swarm(base, 2, [(g, 0, 2) for g in gpus], out)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(client_gpu), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
    live = []
    try:
        swarm.start()
        print("workers online", flush=True)
        common = ["--model", str(base), "--prefix", swarm.prefix, "--peers", *swarm.peers]

        # ---- A. exactness --------------------------------------------------------------------------
        steps = 3
        worker = [sys.executable, str(ROOT / "tests" / "lora_dp_worker.py")]
        ref = spawn(worker + ["--mode", "ref", "--steps", str(steps), "--out", str(out / "ref.pt")] + common,
                    out / "ref.log", env)  # fmt: skip
        assert wait_all([ref], 240) == [0], "reference trainer failed; see ref.log"
        runs = {}
        for name, extra in (("dp", []), ("dp_overlap", ["--overlap"])):
            procs = [
                spawn(worker + ["--mode", "dp", "--rank", str(r), "--world_size", "2", "--run_id", name,
                                "--steps", str(steps), "--out", str(out / f"{name}{r}.pt")] + common + extra,
                      out / f"{name}{r}.log", env)
                for r in range(2)
            ]  # fmt: skip
            live += procs
            assert wait_all(procs, 300) == [0, 0], f"{name} trainers failed; see {name}*.log"
            runs[name] = [torch.load(out / f"{name}{r}.pt") for r in range(2)]
        ref_state = torch.load(out / "ref.pt")
        exact = {}
        for name, (r0, r1) in runs.items():
            exact[name] = dict(
                step0_grad_rel_err_vs_reference=rel(r0["grads0"], ref_state["grads0"]),
                final_weights_rel_err_vs_reference=rel(r0["final"], ref_state["final"]),
                final_weights_rel_err_between_ranks=rel(r0["final"], r1["final"]),
                step0_grad_rel_err_between_ranks=rel(r0["grads0"], r1["grads0"]),
                consistency_deviation=[r0["deviation"], r1["deviation"]],
                losses_rank0=r0["losses"], losses_rank1=r1["losses"], losses_reference=ref_state["losses"],
            )  # fmt: skip
            e = exact[name]
            assert e["step0_grad_rel_err_vs_reference"] < 1e-3, e
            assert e["final_weights_rel_err_vs_reference"] < 1e-3, e
            assert e["final_weights_rel_err_between_ranks"] < 1e-5, e
            assert e["losses_rank0"] == e["losses_rank1"], e
            assert max(e["consistency_deviation"]) < 1e-5, e
        assert ref_state["grads0"].norm() > 0
        report["exactness"] = exact

        # ---- B. CLI + resume -------------------------------------------------------------------------
        data = out / "train.jsonl"
        data.write_text(
            "\n".join(json.dumps({"text": " ".join(f"w{i}" for i in range(s, s + 16))}) for s in (3, 19, 35)) + "\n"
        )

        def cli(name, run_id, steps_, resume=None, ranks=(0, 1), extra=()):
            cmds = []
            for r in ranks:
                cmd = [sys.executable, "examples/train_lora.py", "--model", str(base.resolve()),
                       "--initial_peers", *swarm.peers, "--dht_prefix", swarm.prefix, "--data", str(data),
                       "--output", str(out / name), "--steps", str(steps_), "--seq_len", "8", "--batch_size", "1",
                       "--gradient_accumulation_steps", "2", "--micro_batch_size", "1", "--r", "4",
                       "--lora_alpha", "8", "--lora_dropout", ".3", "--lr", ".01", "--checkpoint_every", "2",
                       "--world_size", "2", "--rank", str(r), "--dp_run_id", run_id, "--dp_timeout", "60",
                       "--dp_matchmaking_time", "1", "--dp_check_every", "1000", *extra]  # fmt: skip
                if resume:
                    cmd += ["--resume_checkpoint", str(resume)]
                cmds.append(spawn(cmd, out / f"{name}-rank{r}.log", env))
            return cmds

        for name, run_id, steps_, resume in (
            ("continuous", "cli1", 4, None),
            ("interrupted", "cli2", 2, None),
            ("resumed", "cli3", 4, out / "interrupted" / "checkpoint-2"),
        ):
            procs = cli(name, run_id, steps_, resume)
            live += procs
            assert wait_all(procs, 300) == [0, 0], f"CLI run {name} failed; see {name}-rank*.log"
        for name in ("continuous", "interrupted", "resumed"):
            assert (out / name / "adapter_model.safetensors").exists()
        for r in range(2):
            assert (out / "continuous" / "checkpoint-2" / f"rank-{r}" / "training_state.pt").exists()
        a = load_file(str(out / "continuous" / "adapter_model.safetensors"))
        b = load_file(str(out / "resumed" / "adapter_model.safetensors"))
        resume_err = max(float((a[k] - b[k]).abs().max()) for k in a)
        losses = json.loads((out / "continuous" / "training.json").read_text())["losses"]
        assert resume_err < 1e-6, resume_err
        report["cli_resume"] = dict(max_abs_adapter_diff_resumed_vs_continuous=resume_err, losses=losses)

        # ---- C. failure handling ------------------------------------------------------------------
        victim = cli("fail", "cli4", 1000, ranks=(0, 1), extra=("--dp_timeout", "30"))
        live += victim
        deadline = time.monotonic() + 240
        log0 = out / "fail-rank0.log"
        while time.monotonic() < deadline and "step=2" not in (log0.read_text() if log0.exists() else ""):
            time.sleep(1)
        assert "step=2" in log0.read_text(), "trainers did not reach step 2"
        os.killpg(victim[1].pid, signal.SIGKILL)
        victim[1].wait()
        started = time.monotonic()
        code = victim[0].wait(timeout=180)
        text = log0.read_text()
        assert code != 0 and "LoRADataParallelError" in text, text[-1500:]
        report["failure"] = dict(survivor_exit_code=code, seconds_to_fail=round(time.monotonic() - started, 1),
                                 message=[l for l in text.splitlines() if "LoRADataParallelError" in l][-1][:300])  # fmt: skip
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        for p in live:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        swarm.stop()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
