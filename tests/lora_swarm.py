"""Shared helpers for the LoRA integration scripts: a private DHT + local workers on idle GPUs.

Imported by tests/run_lora_dp.py. No side effects at import time.
"""

import os
import signal
import subprocess
import sys
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
    """Refuse to use a GPU anyone else is using."""
    rows = {int(r[0]): r for r in nvidia_smi("index,uuid,memory.used,utilization.gpu")}
    busy_uuids = {r[0] for r in nvidia_smi("gpu_uuid,pid", section="compute-apps")}
    for gpu in gpus:
        if gpu not in rows:
            raise RuntimeError(f"GPU {gpu} does not exist on this machine")
        _, uuid_, mem, util = rows[gpu]
        if int(mem) > max_mem_mib or int(util) > max_util or uuid_ in busy_uuids:
            raise RuntimeError(f"GPU {gpu} is not idle (mem={mem} MiB, util={util}%, compute apps={uuid_ in busy_uuids})")


class Swarm:
    """Private DHT plus one worker process per entry of `plan` = [(gpu, first_block, end_block), ...]."""

    def __init__(self, model_dir: Path, num_layers: int, plan, out_dir: Path):
        from bloombee.utils.hivemind_compat import DHT

        self.model_dir, self.num_layers, self.plan, self.out_dir = model_dir, num_layers, plan, out_dir
        self.prefix = "lorasw_" + uuid.uuid4().hex[:10]
        self.dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
        self.peers = [str(a) for a in self.dht.get_visible_maddrs()]
        self.processes, self.logs = [], []

    def start(self, timeout=300):
        from bloombee.data_structures import ServerState
        from bloombee.utils.dht import get_remote_module_infos

        checked = set()
        for index, (gpu, start, end) in enumerate(self.plan):
            if gpu not in checked:
                assert_gpus_idle([gpu])  # immediately before first using this GPU
                checked.add(gpu)
            log = (self.out_dir / f"worker-{index}.log").open("w")
            self.logs.append(log)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "BLOOMBEE_ENABLE_LORA": "1",
                "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
                "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE": "0",
                "BLOOMBEE_LOSSLESS_WRAPPER": "0",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
            }
            command = [
                sys.executable, "-m", "bloombee.cli.run_server", str(self.model_dir.resolve()),
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
        uids = [f"{self.prefix}.{i}" for i in range(self.num_layers)]
        expected = [sum(s <= i < e for _, s, e in self.plan) for i in range(self.num_layers)]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in self.processes):
                raise RuntimeError(f"A worker exited early; see {self.out_dir}/worker-*.log")
            infos = get_remote_module_infos(self.dht, uids, latest=True)
            online = [sum(s.state == ServerState.ONLINE for s in item.servers.values()) for item in infos]
            if all(o >= e for o, e in zip(online, expected)):
                return
            time.sleep(1)
        raise TimeoutError("Workers did not come online")

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
