""" 
    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
    and https://github.com/ginoperrotta/reconchess-strangefish2
    Copyright (c) 2021, The Johns Hopkins University Applied Physics Laboratory LLC
"""

import os
import sys
dirname=os.path.dirname
this_folder = dirname(dirname(__file__))
dirname = os.path.join(this_folder,  "../data-preprocessing").replace("\\", "/")
sys.path.insert(0, dirname)

from functools import partial
from typing import Iterable, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import signal
import numpy as np
from utils import get_row_col, get_board_position_index

import chess

from reconchess.utilities import (
    without_opponent_pieces,
    is_illegal_castle,
    is_psuedo_legal_castle,
    slide_move,
    moves_without_opponent_pieces,
    pawn_capture_moves_on,
    capture_square_of_move,
)


# These are the possible squares to search–all squares that aren't on the edge of the board.
SEARCH_SPOTS = [
    9, 10, 11, 12, 13, 14,
    17, 18, 19, 20, 21, 22,
    25, 26, 27, 28, 29, 30,
    33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46,
    49, 50, 51, 52, 53, 54,
]

# Centipawn scores for pieces
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.ROOK: 563,
    chess.KNIGHT: 305,
    chess.BISHOP: 333,
    chess.QUEEN: 950,
    # chess.KING: 0
}

# Add a hash method for chess.Board objects so that they can be tested for uniqueness, and
# modify the equality method to match. (Neither now uses turn counters, which don't matter in RBC)
chess.Board.__hash__ = lambda self: hash(self._transposition_key())
chess.Board.__eq__ = lambda self, board: self._transposition_key() == board._transposition_key()


def generate_input_for_model(board: chess.Board, opponent_capture_square, sense_history):
    board3d = np.zeros((21, 8, 8), dtype=np.uint8)

    me = board.turn

    # Plane for each piece type for each player 
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            row, col = get_row_col(square, me == chess.BLACK)
            board3d[piece - 1][row][col] = 1
            board3d[12][row][col] = 1
        for square in board.pieces(piece, chess.BLACK):
            row, col = get_row_col(square, me == chess.BLACK)
            board3d[piece + 5][row][col] = 1
            board3d[13][row][col] = 1

    # Color x 1
    if me == chess.WHITE: board3d[14, :, :] = 1

    # capture square after opponent move 
    if opponent_capture_square != None:
        row, col = get_row_col(opponent_capture_square, me == chess.BLACK)
        board3d[15][row][col] = 1

    # Add history of sense locations of the last 5 steps
    plane_index = 16
    for i in range(5): 
        sense_position = sense_history[i]
        if sense_position != None: 
            row, col = get_row_col(sense_position, me == chess.BLACK)
            if row not in [0,7] and col not in [0,7]: 
                # for delta_rank in [1, 0, -1]:
                #     for delta_file in [-1, 0, 1]:
                        # board3d[plane_index][row + delta_rank][col + delta_file] = 1
                    board3d[plane_index][row][col] = 1
                
        plane_index += 1
        

    # return np.asarray([board3d])
    return board3d

        
def revert_sense_square(sense_square, flip): 
    if sense_square != None: 
        row, col = get_row_col(sense_square, flip, width = 6)
        row += 1
        col += 1
        sense_square = get_board_position_index(row, col)

        return sense_square
    else: 
        return None

# print('sense square ', revert_sense_square(6, False))


# for i in range(36): 
#     sense_square = revert_sense_square(i, False)
#     print('pos', i , 'sense square ', sense_square)

# Generate all RBC-legal moves for a board
def generate_rbc_moves(board: chess.Board) -> Iterable[chess.Move]:
    for move in board.pseudo_legal_moves:
        yield move
    for move in without_opponent_pieces(board).generate_castling_moves():
        if not is_illegal_castle(board, move):
            yield move
    yield chess.Move.null()


# Generate all possible moves from just our own pieces
def generate_moves_without_opponent_pieces(board: chess.Board) -> Iterable[chess.Move]:
    for move in moves_without_opponent_pieces(board):
        yield move
    for move in pawn_capture_moves_on(board):
        yield move
    yield chess.Move.null()

# Produce a sense result from a hypothetical true board and a sense square
def simulate_sense(board, square):  # copied (with modifications) from LocalGame
    if square is None:
        # don't sense anything
        sense_result = []
    else:
        if square not in list(chess.SQUARES):
            raise ValueError("LocalGame::sense({}): {} is not a valid square.".format(square, square))
        rank, file = chess.square_rank(square), chess.square_file(square)
        sense_result = []
        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                if 0 <= rank + delta_rank <= 7 and 0 <= file + delta_file <= 7:
                    sense_square = chess.square(file + delta_file, rank + delta_rank)
                    sense_result.append((sense_square, board.piece_at(sense_square)))
    return tuple(sense_result)


# test an attempted move on a board to see what move is actually taken
def simulate_move(board, move, pseudo_legal_moves=None):
    if move == chess.Move.null():
        return None
    if pseudo_legal_moves is None:
        pseudo_legal_moves = list(board.generate_pseudo_legal_moves())
    # if its a legal move, don't change it at all (generate_pseudo_legal_moves does not include pseudo legal castles)
    if move in pseudo_legal_moves or is_psuedo_legal_castle(board, move):
        return move
    if is_illegal_castle(board, move):
        return None
    # if the piece is a sliding piece, slide it as far as it can go
    piece = board.piece_at(move.from_square)
    if piece.piece_type in [chess.PAWN, chess.ROOK, chess.BISHOP, chess.QUEEN]:
        move = slide_move(board, move)
    return move if move in pseudo_legal_moves else None

# check if a taken move would have happened on a board
def validate_move_on_board(
    epd,
    requested_move: Optional[chess.Move],
    taken_move: Optional[chess.Move],
    captured_opponent_piece: bool,
    capture_square: Optional[chess.Square],
) -> bool:
    board = chess.Board(epd)
    # if the taken move was a capture...
    if captured_opponent_piece:
        # the board is invalid if the capture would not have happened
        if not board.is_capture(taken_move):
            return False
        # the board is invalid if the captured piece would have been the king
        # (wrong if it really was the king, but then the game is over)
        captured_piece = board.piece_at(capture_square)
        if captured_piece and captured_piece.piece_type == chess.KING:
            return False
    # if the taken move was not a capture...
    elif taken_move != chess.Move.null():
        # the board is invalid if a capture would have happened
        if board.is_capture(taken_move):
            return False
    # invalid if the requested move would have not resulted in the taken move
    if (simulate_move(board, requested_move) or chess.Move.null()) != taken_move:
        return False
    # otherwise the board is still valid
    return True

# check if a taken move would have happened on a board
def update_board_by_move(
    board: chess.Board,
    requested_move: Optional[chess.Move],
    taken_move: Optional[chess.Move],
    captured_opponent_piece: bool,
    capture_square: Optional[chess.Square],
) -> Optional[chess.Board]:
    # if the taken move was a capture...
    if captured_opponent_piece:
        # the board is invalid if the capture would not have happened
        if not board.is_capture(taken_move):
            return None
        # the board is invalid if the captured piece would have been the king
        # (wrong if it really was the king, but then the game is over)
        captured_piece = board.piece_at(capture_square)
        if captured_piece and captured_piece.piece_type == chess.KING:
            return None
    # if the taken move was not a capture...
    elif taken_move != chess.Move.null():
        # the board is invalid if a capture would have happened
        if board.is_capture(taken_move):
            return None
    # invalid if the requested move would have not resulted in the taken move
    if (simulate_move(board, requested_move) or chess.Move.null()) != taken_move:
        return None
    # otherwise the board is still valid
    board.push(taken_move)
    return board


# Expand one turn's boards into next turn's set by all possible moves. Store as dictionary keyed by capture square.
def populate_next_board_set(board_set: Set[chess.Board], my_color, rc_disable_pbar: bool = False):
    next_turn_boards = defaultdict(set)
    priority_boards = set()
    iter_boards = tqdm(
        board_set,
        disable=rc_disable_pbar,
        unit="boards",
        desc=f"{chess.COLOR_NAMES[my_color]} Expanding {len(board_set)} boards into new set",
    )
    for result in map(partial(get_next_boards_and_capture_squares, my_color), iter_boards):
        for next_board, capture_square, priority in result:
            next_turn_boards[capture_square].add(next_board)
            if priority:
                priority_boards.add(next_board)
    return next_turn_boards, priority_boards

# Check if a board could have produced these sense results
def board_matches_sense(board: chess.Board, sense_result):
    for square, piece in sense_result:
        if board.piece_at(square) != piece:
            return None
    return board

# Generate tuples of next turn's boards and capture squares for one current board
def get_next_boards_and_capture_squares(my_color: chess.Color, board: chess.Board):
    # Calculate all possible opponent moves from this board state
    board.turn = not my_color
    starts_in_check = board.is_check()
    results = []
    for move in generate_rbc_moves(board):
        next_board = board.copy(stack=False)
        priority = False
        capture_square = capture_square_of_move(next_board, move)
        next_board.push(move)
        if board.is_check():  # If the player is in check
            priority = True
        elif starts_in_check and not board.was_into_check():  # Opponent evaded check
            priority = True
        results.append((next_board, capture_square, priority))
    return results


def sense_partition_leq(partition_a, partition_b):
    # Determines if the board set division of partition_a is at least as informative as partition_b
    # (leq = less than or equal which for a set partition indicates a finer division)
    return all(any(subset_a.issubset(subset_b) for subset_b in partition_b) for subset_a in partition_a)

# Change any promotion moves to choose queen
def force_promotion_to_queen(move: chess.Move):
    return move if len(move.uci()) == 4 else chess.Move.from_uci(move.uci()[:4] + "q")


def ignore_one_term(signum, frame):  # Let a sub-process survive the first ctrl-c call for graceful game exiting
    # reset to default response to interrupt signals
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def count_set_bits(x: int):
    return bin(x).count("1")