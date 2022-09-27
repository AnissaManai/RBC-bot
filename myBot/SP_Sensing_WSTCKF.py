import random
from dataclasses import dataclass
from time import sleep, time
from typing import List, Optional, Tuple
from chess import Board, Move
import torch

import chess.engine
from stockfish import Stockfish
from reconchess import *

from stable_baselines3 import PPO

from myBot.utilities import stockfish
from myBot.utilities.utils import (
    generate_input_for_model, 
    revert_sense_square
)
import os

STOCKFISH_ENV_VAR = "FAIRYSTOCKFISH_EXECUTABLE"


class selfPlaySensingWSTCKF(Player):
    """
    TODO: add description of the strategy
    """

    def __init__(self, train: bool):
        self.board = None
        self.color = None
        self.train = train
        self.capture_square = None
        self.sense_history = [None, None, None, None, None]


    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        # initialize the stockfish engine
        # self.engine = chess.engine.SimpleEngine.popen_uci(stockfish, setpgrp=True)
        self.engine = stockfish.create_engine(STOCKFISH_ENV_VAR)
        self.board: Board = board
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[Move], seconds_left: float) -> \
            Optional[Square]:
        if self.train: 
            return sense_actions[0]
        else:
            obs = generate_input_for_model(self.board, self.capture_square, self.sense_history)
            
            # print('input shape ', obs.shape)
            
            # obs = torch.from_numpy(obs)
            # obs = obs.float()
            # obs = obs[None, :, :, :]
            # model_dir = 'models/sl-model'
            # model_path = os.path.join(model_dir, "top3-partial")
            # if torch.cuda.is_available():
            #     model = torch.load(model_path)
            # else: 
            #     model = torch.load(model_path, map_location=torch.device('cpu'))

            # _ , prediction = torch.max(model(obs) , 1) 
            # prediction = prediction.item()
            
            model_dir = 'models/self_play_models'
            model_path = os.path.join(model_dir, "final_model_w_strangefish_g0")

            model = PPO.load(model_path)
            prediction, _states = model.predict(obs, deterministic = True)

            flip = self.color == chess.BLACK
            sense = revert_sense_square(prediction, flip=flip)
            # print('sense ', sense)

            self.sense_history.insert(0, sense)
            self.sense_history.pop()
            
            return int(sense)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(self, move_actions: List[Move], seconds_left: float) -> Optional[Move]:
        # if we might be able to take the king, try to
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return Move(attacker_square, enemy_king_square)

        # otherwise, try to move with the stockfish chess engine
        try:
            self.board.turn = self.color
            self.board.clear_stack()
            result = self.engine.play(self.board, chess.engine.Limit(time=1))

            # # Assign the postion in Stockfish based on FEN
            # board_fen = self.board.fen()
            # stockfish.set_fen_position(board_fen)
            # # Get the top moves
            # top_moves=stockfish.get_top_moves()
            # #Get the best move
            # move =top_moves[0]['Move']

            # print('move ', result.move)
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
            self.engine = stockfish.create_engine(STOCKFISH_ENV_VAR)
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))

        # if all else fails, choose random move
        move = random.choice(move_actions + [None])
        # print('random move ', move)
        return move

    def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is None:
            taken_move = Move.null()

        # print('requested move ', requested_move)
        # print('taken move ', taken_move)
        self.board.turn = self.color
        self.board.push(taken_move)

        
    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
