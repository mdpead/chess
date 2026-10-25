"""Play against a trained checkpoint in the terminal.

Example:
    python -m chess_rl.play runs/first/latest.pt --color white
"""

from __future__ import annotations

import argparse

import chess
import numpy as np

from .mcts import MCTSConfig
from .evaluate import MCTSPlayer
from .model import NetworkEvaluator, load_checkpoint, pick_device


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--color", choices=["white", "black"], default="white", help="your colour")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    engine = MCTSPlayer(NetworkEvaluator(model, device), MCTSConfig(num_simulations=args.simulations, add_noise=False))
    human = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()
    rng = np.random.default_rng()

    while board.outcome(claim_draw=True) is None:
        print(board.unicode(invert_color=True, borders=True, orientation=human), "\n")
        if board.turn == human:
            text = input("Your move (SAN or UCI, 'quit' to exit): ").strip()
            if text in ("quit", "exit"):
                return
            try:
                move = board.parse_san(text)
            except ValueError:
                try:
                    move = chess.Move.from_uci(text)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Illegal or unrecognised move.")
                    continue
        else:
            move = engine.choose_move(board, rng)
            print(f"Engine plays {board.san(move)}\n")
        board.push(move)

    print(board.unicode(invert_color=True, borders=True, orientation=human))
    print("Game over:", board.outcome(claim_draw=True).result())


if __name__ == "__main__":
    main()
