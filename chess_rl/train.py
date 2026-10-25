"""Main reinforcement-learning loop: self-play -> train -> evaluate -> repeat.

Example:
    python -m chess_rl.train --run-dir runs/first --iterations 100
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import POLICY_SIZE
from .evaluate import MCTSPlayer, RandomPlayer, play_match
from .mcts import MCTSConfig
from .model import ChessNet, ModelConfig, NetworkEvaluator, load_checkpoint, pick_device, save_checkpoint
from .selfplay import ReplayBuffer, Sample, SelfPlayConfig, play_games


def make_batch(samples: list[Sample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    planes = torch.from_numpy(np.stack([s.planes for s in samples]).astype(np.float32))
    policy = torch.zeros(len(samples), POLICY_SIZE)
    for i, s in enumerate(samples):
        policy[i, torch.from_numpy(s.policy_indices)] = torch.from_numpy(s.policy_probs)
    values = torch.tensor([s.value for s in samples], dtype=torch.float32)
    return planes.to(device), policy.to(device), values.to(device)


def train_steps(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = Counter()
    for _ in range(steps):
        planes, target_policy, target_value = make_batch(buffer.sample_batch(batch_size), device)
        logits, value = model(planes)
        policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals["policy_loss"] += policy_loss.item()
        totals["value_loss"] += value_loss.item()
    return {k: v / max(steps, 1) for k, v in totals.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/default", help="where checkpoints and logs go")
    p.add_argument("--resume", action="store_true", help="resume from <run-dir>/latest.pt")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    # model
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    # self-play
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--parallel-games", type=int, default=32)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--temperature-moves", type=int, default=30)
    # training
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--min-buffer", type=int, default=5_000, help="samples needed before training starts")
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # evaluation
    p.add_argument("--eval-every", type=int, default=5, help="iterations between evaluations (0 = never)")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-simulations", type=int, default=32)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    start_iter = 0
    if args.resume and (run_dir / "latest.pt").exists():
        model, payload = load_checkpoint(str(run_dir / "latest.pt"), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_iter = payload.get("iteration", 0)
        print(f"Resumed from iteration {start_iter}")
    else:
        model = ChessNet(ModelConfig(channels=args.channels, blocks=args.blocks)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        save_checkpoint(str(run_dir / "iter_0000.pt"), model, iteration=0)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    selfplay_cfg = SelfPlayConfig(
        games_per_batch=args.parallel_games,
        temperature_moves=args.temperature_moves,
        max_plies=args.max_plies,
        mcts=MCTSConfig(num_simulations=args.simulations),
    )
    evaluator = NetworkEvaluator(model, device)
    buffer = ReplayBuffer(args.buffer_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | params: {n_params:,} | self-play config: {asdict(selfplay_cfg)}")

    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        t0 = time.time()
        records = play_games(evaluator, args.games_per_iter, selfplay_cfg, rng)
        for r in records:
            buffer.add_game(r)
        t_selfplay = time.time() - t0

        stats: dict = {
            "iteration": iteration,
            "buffer": len(buffer),
            "selfplay_sec": round(t_selfplay, 1),
            "avg_plies": float(np.mean([len(r.moves) for r in records])),
            "results": dict(Counter({1.0: "white", -1.0: "black", 0.0: "draw"}[r.result] for r in records)),
            "terminations": dict(Counter(r.termination for r in records)),
        }

        if len(buffer) >= args.min_buffer:
            t1 = time.time()
            stats.update(train_steps(model, optimizer, buffer, args.train_steps, args.batch_size, device))
            stats["train_sec"] = round(time.time() - t1, 1)

        save_checkpoint(str(run_dir / "latest.pt"), model, optimizer, iteration=iteration)

        if args.eval_every and iteration % args.eval_every == 0:
            ckpt = run_dir / f"iter_{iteration:04d}.pt"
            save_checkpoint(str(ckpt), model, iteration=iteration)
            eval_cfg = MCTSConfig(num_simulations=args.eval_simulations, add_noise=False)
            current = MCTSPlayer(evaluator, eval_cfg)
            stats["vs_random"] = play_match(current, RandomPlayer(), args.eval_games, rng=rng).summary()
            previous = sorted(run_dir.glob("iter_*.pt"))
            if len(previous) >= 2:
                old_model, _ = load_checkpoint(str(previous[-2]), device)
                old = MCTSPlayer(NetworkEvaluator(old_model, device), eval_cfg)
                stats[f"vs_{previous[-2].stem}"] = play_match(current, old, args.eval_games, rng=rng).summary()

        print(json.dumps(stats))
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")


if __name__ == "__main__":
    main()
