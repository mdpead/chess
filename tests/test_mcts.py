import chess
import numpy as np

from chess_rl.mcts import MCTSConfig, SearchTree, run_search


class UniformEvaluator:
    def evaluate(self, boards):
        out = []
        for b in boards:
            moves = list(b.legal_moves)
            out.append((moves, np.full(len(moves), 1.0 / len(moves)), 0.0))
        return out


def best_move(fen, sims=400):
    tree = SearchTree(chess.Board(fen), MCTSConfig(num_simulations=sims, add_noise=False), np.random.default_rng(0))
    run_search([tree], UniformEvaluator(), sims)
    moves, counts = tree.visit_counts()
    return moves[int(counts.argmax())], tree


def test_finds_mate_in_one_white():
    move, tree = best_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert move == chess.Move.from_uci("a1a8")
    assert tree.root_value() > 0.5


def test_finds_mate_in_one_black():
    move, _ = best_move("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    assert move == chess.Move.from_uci("a8a1")


def test_visit_counts_sum_to_simulations():
    sims = 50
    _, tree = best_move(chess.STARTING_FEN, sims)
    _, counts = tree.visit_counts()
    assert counts.sum() == sims - 1  # first simulation expands the root
