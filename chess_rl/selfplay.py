"""Self-play game generation and the replay buffer."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import chess
import numpy as np

from .encoding import encode_board, move_to_index
from .mcts import Evaluator, MCTSConfig, SearchTree, run_search, select_move


@dataclass
class SelfPlayConfig:
    games_per_batch: int = 32  # games played concurrently (sharing NN batches)
    temperature_moves: int = 30  # plies sampled with temperature 1, then greedy
    max_plies: int = 400  # games longer than this are scored as draws
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


@dataclass
class Sample:
    planes: np.ndarray  # (NUM_PLANES, 8, 8) float16
    policy_indices: np.ndarray  # int64 indices of moves with visits
    policy_probs: np.ndarray  # float32 visit distribution over those moves
    value: float  # game result from the side-to-move's perspective


@dataclass
class GameRecord:
    samples: list[Sample]
    result: float  # +1 white win, -1 black win, 0 draw
    moves: list[chess.Move]
    termination: str


@dataclass
class _ActiveGame:
    board: chess.Board
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, chess.Color]] = field(default_factory=list)


def _game_result(board: chess.Board, max_plies: int) -> tuple[float, str] | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
        return result, outcome.termination.name.lower()
    if board.ply() >= max_plies:
        return 0.0, "max_plies"
    return None


def play_games(
    evaluator: Evaluator,
    num_games: int,
    config: SelfPlayConfig,
    rng: np.random.Generator | None = None,
) -> list[GameRecord]:
    """Play `num_games` self-play games, `config.games_per_batch` at a time."""
    rng = rng or np.random.default_rng()
    records: list[GameRecord] = []
    remaining = num_games

    while remaining > 0:
        active = [_ActiveGame(chess.Board()) for _ in range(min(config.games_per_batch, remaining))]
        remaining -= len(active)

        while active:
            trees = [SearchTree(g.board, config.mcts, rng) for g in active]
            run_search(trees, evaluator, config.mcts.num_simulations)

            still_active = []
            for game, tree in zip(active, trees):
                board = game.board
                moves, counts = tree.visit_counts()
                probs = (counts / counts.sum()).astype(np.float32)
                indices = np.array([move_to_index(m, board.turn) for m in moves], dtype=np.int64)
                game.pending.append((encode_board(board).astype(np.float16), indices, probs, board.turn))

                temperature = 1.0 if board.ply() < config.temperature_moves else 0.0
                board.push(select_move(moves, counts, temperature, rng))

                finished = _game_result(board, config.max_plies)
                if finished is None:
                    still_active.append(game)
                    continue
                result, termination = finished
                samples = [
                    Sample(planes, idx, p, result if turn == chess.WHITE else -result)
                    for planes, idx, p, turn in game.pending
                ]
                records.append(GameRecord(samples, result, list(board.move_stack), termination))
            active = still_active

    return records


class ReplayBuffer:
    """Fixed-capacity ring buffer of training samples (oldest evicted first)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.samples: list[Sample] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add_game(self, record: GameRecord) -> None:
        for sample in record.samples:
            if len(self.samples) < self.capacity:
                self.samples.append(sample)
            else:
                self.samples[self._next] = sample
            self._next = (self._next + 1) % self.capacity

    def sample_batch(self, batch_size: int) -> list[Sample]:
        return random.choices(self.samples, k=batch_size)
