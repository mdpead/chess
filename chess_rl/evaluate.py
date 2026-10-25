"""Pit players against each other to measure progress.

Example:
    python -m chess_rl.evaluate runs/first/latest.pt random --games 20
    python -m chess_rl.evaluate runs/first/latest.pt runs/first/iter_0010.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import chess
import numpy as np

from .mcts import Evaluator, MCTSConfig, SearchTree, game_outcome, run_search, select_move
from .model import NetworkEvaluator, load_checkpoint, pick_device


class Player:
    name = "player"

    def choose_move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        raise NotImplementedError


class RandomPlayer(Player):
    name = "random"

    def choose_move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]


class MCTSPlayer(Player):
    def __init__(self, evaluator: Evaluator, config: MCTSConfig, name: str = "mcts"):
        self.evaluator = evaluator
        self.config = config
        self.name = name

    def choose_move(self, board: chess.Board, rng: np.random.Generator, temperature: float = 0.0) -> chess.Move:
        tree = SearchTree(board, self.config, rng)
        run_search([tree], self.evaluator, self.config.num_simulations)
        moves, counts = tree.visit_counts()
        return select_move(moves, counts, temperature, rng)


@dataclass
class MatchResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    def elo_diff(self) -> float:
        s = min(max(self.score, 1e-3), 1 - 1e-3)
        return -400 * math.log10(1 / s - 1)

    def summary(self) -> dict:
        return {"W": self.wins, "D": self.draws, "L": self.losses, "score": round(self.score, 3),
                "elo_diff": round(self.elo_diff())}


def play_match(
    player_a: Player,
    player_b: Player,
    games: int,
    max_plies: int = 300,
    random_opening_plies: int = 2,
    rng: np.random.Generator | None = None,
) -> MatchResult:
    """Play `games` games alternating colours; results are from A's perspective.

    The first `random_opening_plies` plies are random so deterministic players
    don't replay the same game over and over.
    """
    rng = rng or np.random.default_rng()
    result = MatchResult()
    for g in range(games):
        a_is_white = g % 2 == 0
        board = chess.Board()
        while game_outcome(board) is None and board.ply() < max_plies:
            if board.ply() < random_opening_plies:
                move = RandomPlayer().choose_move(board, rng)
            else:
                mover = player_a if (board.turn == chess.WHITE) == a_is_white else player_b
                move = mover.choose_move(board, rng)
            board.push(move)
        outcome = game_outcome(board)
        if outcome is None or outcome.winner is None:
            result.draws += 1
        elif (outcome.winner == chess.WHITE) == a_is_white:
            result.wins += 1
        else:
            result.losses += 1
    return result


def load_player(spec: str, simulations: int, device) -> Player:
    if spec == "random":
        return RandomPlayer()
    model, _ = load_checkpoint(spec, device)
    return MCTSPlayer(NetworkEvaluator(model, device), MCTSConfig(num_simulations=simulations, add_noise=False), spec)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("player_a", help="checkpoint path or 'random'")
    p.add_argument("player_b", help="checkpoint path or 'random'")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    a = load_player(args.player_a, args.simulations, device)
    b = load_player(args.player_b, args.simulations, device)
    result = play_match(a, b, args.games, rng=np.random.default_rng(args.seed))
    print(f"{args.player_a} vs {args.player_b}: {result.summary()}")


if __name__ == "__main__":
    main()
