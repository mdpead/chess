"""Board and move encodings.

Everything is encoded from the perspective of the side to move: when it is
Black's turn the board is mirrored vertically so the network always "plays
up the board". This halves what the network has to learn.

Board planes (NUM_PLANES x 8 x 8):
    0-5    side-to-move pieces   (P, N, B, R, Q, K)
    6-11   opponent pieces       (P, N, B, R, Q, K)
    12     position has occurred before (repetition)
    13-14  side-to-move castling rights (kingside, queenside)
    15-16  opponent castling rights     (kingside, queenside)
    17     en passant target square
    18     halfmove clock / 100 (fifty-move rule progress)

Moves use the AlphaZero 8x8x73 scheme (4672 indices): for each from-square,
56 "queen-like" moves (8 directions x 7 distances), 8 knight moves and
9 under-promotions (3 directions x {knight, bishop, rook}). Queen promotions
are encoded as ordinary queen-like moves.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 19
NUM_MOVE_PLANES = 73
POLICY_SIZE = 64 * NUM_MOVE_PLANES  # 4672

# (file delta, rank delta) unit vectors: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}
_KNIGHT_INDEX = {d: i for i, d in enumerate(KNIGHT_DELTAS)}
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _bitboard_to_plane(bb: int) -> np.ndarray:
    """Convert a 64-bit bitboard to an 8x8 array indexed [rank][file]."""
    raw = np.frombuffer(bb.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as a float32 array of shape (NUM_PLANES, 8, 8)."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    us = board.turn
    them = not us
    flip = us == chess.BLACK

    for i, piece_type in enumerate(_PIECE_TYPES):
        for offset, color in ((0, us), (6, them)):
            bb = board.pieces_mask(piece_type, color)
            if flip:
                bb = chess.flip_vertical(bb)
            planes[offset + i] = _bitboard_to_plane(bb)

    if board.is_repetition(2):
        planes[12] = 1.0

    planes[13] = board.has_kingside_castling_rights(us)
    planes[14] = board.has_queenside_castling_rights(us)
    planes[15] = board.has_kingside_castling_rights(them)
    planes[16] = board.has_queenside_castling_rights(them)

    if board.ep_square is not None and board.has_legal_en_passant():
        sq = chess.square_mirror(board.ep_square) if flip else board.ep_square
        planes[17, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    planes[18] = min(board.halfmove_clock, 100) / 100.0
    return planes


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Map a move to its policy index in [0, POLICY_SIZE)."""
    from_sq, to_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)

    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + (df + 1) * 3 + UNDERPROMOTIONS.index(move.promotion)
    elif (df, dr) in _KNIGHT_INDEX:
        plane = 56 + _KNIGHT_INDEX[(df, dr)]
    else:
        distance = max(abs(df), abs(dr))
        direction = (int(np.sign(df)), int(np.sign(dr)))
        plane = _QUEEN_DIR_INDEX[direction] * 7 + (distance - 1)

    return from_sq * NUM_MOVE_PLANES + plane


def legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    """Return the legal moves and their policy indices."""
    moves = list(board.legal_moves)
    indices = np.fromiter((move_to_index(m, board.turn) for m in moves), dtype=np.int64, count=len(moves))
    return moves, indices
