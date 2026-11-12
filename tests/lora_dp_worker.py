"""One LoRA trainer process for tests/run_lora_dp.py.

mode=dp : data-parallel rank R of N. Each step it trains on its own row and all-reduces the gradients.
mode=ref: a single trainer that trains on the rows of *all* N ranks with plain gradient accumulation
          (each row is its own forward/backward, loss/N), i.e. what DP is supposed to be equivalent to.
Both start from the same adapter and use identical clipping/AdamW, and write gradients of step 0
(after reduction, before the optimizer) and the final adapter weights to --out.
"""

import argparse
import time

import torch

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig
from bloombee.client.lora_data_parallel import LoRADataParallel
from bloombee.utils.lora import PROJECTIONS

SEQ, ROWS, VOCAB = 8, 16, 64  # defaults; overridable with --seq/--rows/--vocab/--batch for benchmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dp", "ref"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--peers", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--run_id", default="exact")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--seq", type=int, default=SEQ)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--rows", type=int, default=ROWS, help="size of the fixed token table")
    parser.add_argument("--batch", type=int, default=1, help="rows per trainer per step")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seq = args.seq

    client = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        dht_prefix=args.prefix,
        initial_peers=args.peers,
        torch_dtype=torch.float16,
        connect_timeout=30,
        request_timeout=120,
        max_retries=2,
        update_period=1,
    ).cuda()
    torch.manual_seed(1234)  # identical adapter init in every process
    model = DistributedLoRA(
        client, LoRAConfig(r=4, lora_alpha=8, target_modules=PROJECTIONS), micro_batch_size=args.micro_batch_size
    ).cuda()
    with torch.no_grad():  # nonzero B so every A/B matrix receives gradient
        for name, p in model.named_parameters():
            if p.requires_grad and "lora_B" in name:
                p.normal_(0, 0.05)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    table = torch.randint(3, args.vocab, (args.rows, seq), generator=torch.Generator().manual_seed(123)).cuda()

    def row(step, rank):
        first = (args.world_size * step + rank) * args.batch
        return table[[(first + i) % args.rows for i in range(args.batch)]]

    dp = None
    if args.mode == "dp":
        dp = LoRADataParallel.for_model(
            model,
            run_id=args.run_id,
            rank=args.rank,
            world_size=args.world_size,
            initial_peers=args.peers,
            timeout=120,
            matchmaking_time=1.0,
        )
        dp.sync_initial_state()
    grads0, losses, step_times = None, [], []
    try:
        for step in range(args.steps):
            torch.cuda.synchronize()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if dp is not None:
                if args.overlap:
                    dp.begin_round()
                ids = row(step, args.rank)
                loss = model(ids, labels=ids).loss
                loss.backward()
                losses.append(dp.allreduce_gradients(args.batch * (seq - 1), loss=float(loss)))
            else:
                total = 0.0
                for rank in range(args.world_size):
                    ids = row(step, rank)
                    loss = model(ids, labels=ids).loss
                    (loss / args.world_size).backward()
                    total += float(loss) / args.world_size
                losses.append(total)
            if step == 0:
                grads0 = torch.cat([p.grad.detach().float().flatten().cpu() for p in params])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - began)
        if dp is not None:
            deviation = dp.check_consistency()
        else:
            deviation = 0.0
        torch.save(
            dict(
                grads0=grads0,
                final=torch.cat([p.detach().float().flatten().cpu() for p in params]),
                losses=losses,
                deviation=deviation,
                step_times=step_times,
            ),
            args.out,
        )
    finally:
        if dp is not None:
            dp.shutdown()
        model.remote_layers.sequence_manager.shutdown()


if __name__ == "__main__":
    main()
