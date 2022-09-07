""" 
    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
    and https://github.com/ginoperrotta/reconchess-strangefish2
    Copyright (c) 2021, The Johns Hopkins University Applied Physics Laboratory LLC
"""

from collections import defaultdict
from functools import partial
from time import time, sleep
from typing import Optional, List, Tuple, Set

import chess.engine
from reconchess import Player, Color, GameHistory, WinReason, Square
from tqdm import tqdm

from myBot.utilities import (
    board_matches_sense,
    update_board_by_move,
    populate_next_board_set,
)

from myBot.utilities.player_logging import create_main_logger
from myBot.utilities.timing import Timer

class RBCAgentCore(Player):
    def __init__(
        self,
        log_to_file: bool,
        rc_disable_pbar: bool 
    ) -> None:

        self.board: chess.Board = None
        self.boards: Set[chess.Board] = set()
        self.priority_boards: Set[chess.Board] = set()
        self.next_turn_boards: defaultdict[Optional[Square], Set] = defaultdict(set)
        self.next_turn_boards_unsorted: Set[chess.Board] = set()
        self.capture_square = None
        self.sense_history = [None, None, None, None, None]

        self.color = None
        self.turn_num = None

        self.rc_disable_pbar = rc_disable_pbar

        self.logger = create_main_logger(log_to_file=log_to_file)
        self.logger.debug("A new StrangeFish player was initialized.")


    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        color_name = chess.COLOR_NAMES[color]

        self.logger.info(f"Starting a new game as {color_name} against {opponent_name}.")
        self.boards = {board.copy()}
        self.priority_boards = set()
        self.color = color
        self.turn_num = 0
        self.board = board

    
    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.turn_num += 1
        self.logger.debug(f"Starting turn {self.turn_num}.")

        # Do not "handle_opponent_move_result" if no one has moved yet
        if self.turn_num == 1 and self.color == chess.WHITE:
            return

        if captured_my_piece:
            self.logger.debug(f"Opponent captured my piece at {chess.SQUARE_NAMES[capture_square]}.")
            self.board.remove_piece_at(capture_square)
        else:
            self.logger.debug("Opponent's move was not a capture.")

        print('Number of boards after last turn ', self.color, 'IS ', len(self.boards))

        # If creation of new board set didn't complete during op's turn (self.boards will not be empty)
        if self.boards:
            new_board_set, boards_in_check = populate_next_board_set(
                self.boards, self.color, rc_disable_pbar=self.rc_disable_pbar
            )
            self.priority_boards |= boards_in_check
            for square in new_board_set.keys():
                self.next_turn_boards[square] |= new_board_set[square]

        # Get this turn's board set from a dictionary keyed by the possible capture squares
        self.boards = self.next_turn_boards[capture_square]

        self.logger.debug(
            "Finished expanding and filtering the set of possible board states. "
            f"There are {len(self.boards)} possible boards "
            f"at the start of our turn {self.turn_num}."
        )

        print('Finished expanding and filtering the set of possible board states. There are', len(self.boards), 'possible boards at the start of our turn', self.turn_num)


    def choose_sense(
        self,
        sense_actions: List[Square],
        move_actions: List[chess.Move],
        seconds_left: float,
    ) -> Optional[Square]:

        self.logger.debug(
            f"Choosing a sensing square for turn {self.turn_num} "
            f"with {len(self.boards)} boards and {seconds_left:.0f} seconds remaining."
        )

        # The option to pass isn't included in the reconchess input
        move_actions += [chess.Move.null()]

        with Timer(self.logger.debug, "choosing sense location"):
            # Pass the needed information to the decision-making function to choose a sense square
            sense_choice = self.sense_strategy(self.board, self.color, sense_actions, move_actions, self.capture_square, self.sense_history, seconds_left)

        self.logger.debug(f"Chose to sense {chess.SQUARE_NAMES[sense_choice] if sense_choice else 'nowhere'}")

        self.sense_history.insert(0, sense_choice)
        self.sense_history.pop()
        print('sense history ', self.sense_history)

        return sense_choice

    def sense_strategy(
        self,
        sense_actions: List[Square],
        move_actions: List[chess.Move],
        seconds_left: float,
    ) -> Optional[Square]:
        raise NotImplementedError

    
    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):

        # Filter the possible board set to only boards which would have produced the observed sense result
        num_before = len(self.boards)
        i = tqdm(
            self.boards,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} Filtering {len(self.boards)} boards by sense results",
            unit="boards",
        )
        self.boards = {
            board for board in map(partial(board_matches_sense, sense_result=sense_result), i) if board is not None
        }
        self.logger.debug(f"There were {num_before} possible boards before sensing " f"and {len(self.boards)} after.")
        print('There were', num_before, 'possible boards before sensing and', len(self.boards) ,'after.')

        # how many board were eliminated
        board_set_reduction = ( num_before - len(self.boards)) / num_before 
        print('percentage of eliminated boards ', board_set_reduction)

        # add the pieces in the sense result to our board for partial sensing strategy
        if len(self.boards) == 1:
            board: chess.Board = list(self.boards)[0]
            self.board.set_fen(board.fen())
        else:
            for square, piece in sense_result:
                self.board.set_piece_at(square, piece)

        return board_set_reduction

    
    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # Currently, move_actions is passed by reference, so if we add the null move here it will be in the list twice
        #  since we added it in choose_sense also. Instead of removing this line altogether, I'm leaving a check so we
        #  are prepared in the case that reconchess is updated to pass a copy of the move_actions list instead.
        if chess.Move.null() not in move_actions:
            move_actions += [chess.Move.null()]
        
        print('Choosing move for turn', self.turn_num ,' from ', len(move_actions), ' moves over', len(self.boards), 'boards with', seconds_left ,'seconds remaining.')

        self.logger.debug(
            f"Choosing move for turn {self.turn_num} "
            f"from {len(move_actions)} moves over {len(self.boards)} boards "
            f"with {seconds_left:.2f} seconds remaining."
        )

        with Timer(self.logger.debug, "choosing move"):
            # Pass the needed information to the decision-making function to choose a move
            move_choice = self.move_strategy(move_actions, seconds_left)

        self.logger.debug(f"The chosen move was {move_choice}")

        # reconchess uses None for the null move, so correct the function output if that was our choice
        return move_choice if move_choice != chess.Move.null() else None


    def move_strategy(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        raise NotImplementedError


    def handle_move_result(
        self,
        requested_move: Optional[chess.Move],
        taken_move: Optional[chess.Move],
        captured_opponent_piece: bool,
        capture_square: Optional[Square],
    ):

        print('The requested move was', requested_move, 'and the taken move was', taken_move)
        self.logger.debug(f"The requested move was {requested_move} and the taken move was {taken_move}.")
        if captured_opponent_piece:
            self.logger.debug(f"Move {taken_move} was a capture!")

        num_boards_before_filtering = len(self.boards)

        if requested_move is None:
            requested_move = chess.Move.null()
        if taken_move is None:
            taken_move = chess.Move.null()

        # Filter the possible board set to only boards on which the requested move would have resulted in the taken move
        # invalid board if --> the move is a capture but would not have happened in this board 
        #                  --> the move is capture and the capture piece is king 
        #                  --> the move is not capture but capture happened
        #                  --> if the requested move would have not resulted in the taken move
        i = tqdm(
            self.boards,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} Filtering {len(self.boards)} boards by move results",
            unit="boards",
        )
        self.boards = {
            board
            for board in map(
                partial(
                    update_board_by_move,
                    requested_move=requested_move,
                    taken_move=taken_move,
                    captured_opponent_piece=captured_opponent_piece,
                    capture_square=capture_square,
                ),
                i,
            )
            if board is not None
        }

        # Update board for partial sensing strategy 
        self.board.turn = self.color
        self.board.push(taken_move)

        self.logger.debug(
            f"There were {num_boards_before_filtering} possible boards "
            f"before filtering and {len(self.boards)} after."
        )

        print('There were', num_boards_before_filtering, 'possible boards before filtering with taken move and', len(self.boards), 'after.')

        # Re-initialize the set of boards for next turn (filled in while_we_wait and/or handle_opponent_move_result)
        self.next_turn_boards = defaultdict(set)
        self.next_turn_boards_unsorted = set()
        self.priority_boards = set()



    def while_we_wait(self):
        start_time = time()
        self.logger.debug(
            "Running the `while_we_wait` method. " f"{len(self.boards)} boards left to expand for next turn."
        )

        our_king_square = tuple(self.boards)[0].king(self.color) if len(self.boards) else None

        while time() - start_time < 1:  # Rate-limit server queries by looping here for at least 1 second

            # If there are still boards in the set from last turn, remove one and expand it by all possible moves
            if len(self.boards):
                new_board_set, priority_boards = populate_next_board_set(
                    {self.boards.pop()},
                    self.color,
                    rc_disable_pbar=True,
                )
                self.priority_boards |= priority_boards
                for square in new_board_set.keys():
                    self.next_turn_boards[square] |= new_board_set[square]
                    if square != our_king_square:
                        self.next_turn_boards_unsorted |= new_board_set[square]

            # If all of last turn's boards have been expanded, pass to the sense/move function's waiting method
            else:
                self.downtime_strategy()


    def downtime_strategy(self):
        sleep(0.1)

    def handle_game_end(
        self,
        winner_color: Optional[Color],
        win_reason: Optional[WinReason],
        game_history: GameHistory,
    ):
        self.logger.info(
            f"I {'won' if winner_color == self.color else 'lost'} "
            f"by {win_reason.name if hasattr(win_reason, 'name') else win_reason}"
        )
        self.gameover_strategy()

    def gameover_strategy(self):
        pass
