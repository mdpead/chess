import random

import chess
import numpy as np

from chess_rl.encoding import NUM_PLANES, POLICY_SIZE, encode_board, legal_move_indices, move_to_index


def random_positions(n=200, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        yield board


def test_legal_moves_map_to_unique_indices():
    for board in random_positions():
        moves, idx = legal_move_indices(board)
        assert len(set(idx.tolist())) == len(moves)
        assert all(0 <= i < POLICY_SIZE for i in idx)


def test_promotions_are_distinct():
    board = chess.Board("1n5k/P7/8/8/8/8/8/K7 w - - 0 1")
    moves, idx = legal_move_indices(board)
    promos = [i for m, i in zip(moves, idx) if m.promotion]
    assert len(promos) == 8  # 4 pieces x (push, capture on b8)
    assert len(set(promos)) == 8


def test_black_moves_are_mirrored():
    white = chess.Board()
    black = chess.Board()
    black.push_uci("a2a3")
    for uci_w, uci_b in [("e2e4", "e7e5"), ("g1f3", "g8f6"), ("b1c3", "b8c6")]:
        assert move_to_index(chess.Move.from_uci(uci_w), white.turn) == move_to_index(
            chess.Move.from_uci(uci_b), black.turn
        )


def test_encoding_is_side_to_move_relative():
    planes_w = encode_board(chess.Board())
    planes_b = encode_board(chess.Board().mirror())  # same position, Black to move, colours swapped
    assert planes_w.shape == (NUM_PLANES, 8, 8)
    np.testing.assert_array_equal(planes_w, planes_b)
    # own pawns on the second rank
    assert planes_w[0, 1].sum() == 8
