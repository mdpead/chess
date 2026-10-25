"""Minimal UCI engine wrapper, so checkpoints can be used from chess GUIs
(Arena, Cute Chess, lichess-bot, ...).

Example:
    python -m chess_rl.uci runs/first/latest.pt --simulations 400
"""

from __future__ import annotations

import argparse
import sys

import chess
import numpy as np

from .mcts import MCTSConfig
from .evaluate import MCTSPlayer
from .model import NetworkEvaluator, load_checkpoint, pick_device


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    engine = MCTSPlayer(NetworkEvaluator(model, device), MCTSConfig(num_simulations=args.simulations, add_noise=False))
    board = chess.Board()
    rng = np.random.default_rng()

    def send(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            send("id name chess-rl")
            send("id author chess-rl")
            send("uciok")
        elif cmd == "isready":
            send("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            if len(tokens) > 1 and tokens[1] == "startpos":
                board = chess.Board()
                rest = tokens[2:]
            elif len(tokens) > 1 and tokens[1] == "fen":
                fen_end = tokens.index("moves") if "moves" in tokens else len(tokens)
                board = chess.Board(" ".join(tokens[2:fen_end]))
                rest = tokens[fen_end:]
            else:
                continue
            if rest and rest[0] == "moves":
                for uci in rest[1:]:
                    board.push_uci(uci)
        elif cmd == "go":
            if board.is_game_over():
                send("bestmove 0000")
            else:
                send(f"bestmove {engine.choose_move(board, rng).uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
