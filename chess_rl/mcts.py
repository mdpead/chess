"""Monte Carlo Tree Search with PUCT selection (AlphaZero style).

The search is split into `select()` and `expand_and_backup()` so that many
trees (e.g. many concurrent self-play games) can share one batched network
evaluation per simulation step. `run_search()` drives a group of trees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np


class Evaluator(Protocol):
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[list[chess.Move], np.ndarray, float]]: ...


@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_noise: bool = True


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "terminal_value")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        # Sum of values from the perspective of the player who moved INTO this node.
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.terminal_value: float | None = None

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def terminal_value(board: chess.Board) -> float | None:
    """Value for the side to move if the game is over, else None."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class SearchTree:
    def __init__(self, board: chess.Board, config: MCTSConfig, rng: np.random.Generator | None = None):
        self.board = board.copy()
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.root = Node(1.0)

    def select(self) -> tuple[list[Node], chess.Board] | None:
        """Walk down to a leaf. Returns (path, board) if the leaf needs a
        network evaluation, or None if it was terminal (already backed up)."""
        node = self.root
        board = self.board.copy()
        path = [node]
        while node.expanded:
            move, node = self._select_child(node)
            board.push(move)
            path.append(node)

        if node.terminal_value is None and node is not self.root:
            node.terminal_value = terminal_value(board)
        if node.terminal_value is not None:
            self._backup(path, node.terminal_value)
            return None
        return path, board

    def expand_and_backup(self, path: list[Node], moves: list[chess.Move], priors: np.ndarray, value: float) -> None:
        leaf = path[-1]
        if not leaf.expanded:
            if leaf is self.root and self.config.add_noise and len(moves) > 0:
                noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(moves))
                eps = self.config.dirichlet_epsilon
                priors = (1 - eps) * priors + eps * noise
            for move, p in zip(moves, priors):
                leaf.children[move] = Node(float(p))
        self._backup(path, value)

    def _select_child(self, node: Node) -> tuple[chess.Move, Node]:
        c_sqrt_n = self.config.c_puct * math.sqrt(node.visit_count)
        best_score = -math.inf
        best = None
        for move, child in node.children.items():
            score = child.q() + c_sqrt_n * child.prior / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best = (move, child)
        return best

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        # `value` is from the perspective of the side to move at the leaf; each
        # node stores value from the perspective of the player who moved into it.
        for node in reversed(path):
            value = -value
            node.value_sum += value
            node.visit_count += 1

    def visit_counts(self) -> tuple[list[chess.Move], np.ndarray]:
        moves = list(self.root.children.keys())
        counts = np.array([self.root.children[m].visit_count for m in moves], dtype=np.float64)
        return moves, counts

    def root_value(self) -> float:
        """Estimated value of the root position for the side to move."""
        return -self.root.q()


def run_search(trees: list[SearchTree], evaluator: Evaluator, num_simulations: int) -> None:
    """Run `num_simulations` simulations on every tree, batching evaluations."""
    for _ in range(num_simulations):
        pending = []
        for tree in trees:
            leaf = tree.select()
            if leaf is not None:
                pending.append((tree, leaf))
        if not pending:
            continue
        results = evaluator.evaluate([board for _, (_, board) in pending])
        for (tree, (path, _)), (moves, priors, value) in zip(pending, results):
            tree.expand_and_backup(path, moves, priors, value)


def select_move(moves: list[chess.Move], counts: np.ndarray, temperature: float, rng: np.random.Generator) -> chess.Move:
    if temperature <= 1e-6:
        best = np.flatnonzero(counts == counts.max())
        return moves[int(rng.choice(best))]
    weights = counts ** (1.0 / temperature)
    return moves[int(rng.choice(len(moves), p=weights / weights.sum()))]
