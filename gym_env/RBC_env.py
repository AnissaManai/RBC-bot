import os
from re import S
import numpy as np
from stable_baselines3 import PPO
import gym
from gym import spaces
import chess
from typing import List, Optional
from datetime import datetime

from reconchess import LocalGame, Player, Game, Square
from myBot import selfPlaySensingWSTCKF, StrangefishWStockfish
# from myBot import SelfPlaySensingWSTRGF
from myBot.utilities.utils import generate_input_for_model, revert_sense_square

class RBCEnv(gym.Env):

    def __init__(self, model_dir):

        self.action_space = spaces.Discrete(36)
        self.observation_space = spaces.Box(low=0, high=1, shape=(21, 8,8), dtype = np.uint8)
        self.model_dir = model_dir

        self.reward_range = (-1100, 1100)

        self.num_episode = 0
        self.num_step = 0
        self.current_learning_player = chess.WHITE
        self.success_episode = []

        # self.white_player:Player = selfPlaySensingWSTCKF(train = True)
        # self.black_player:Player = selfPlaySensingWSTCKF(train = True)

        self.done = False

    def reset(self):
        
        self.sense_result = None
        self.opponent_capture_square = None
        self.opt_enemy_capture_square = None

        self.num_captured_white_pieces = 0
        self.num_captured_black_pieces = 0

        self.black_sense_history = [None, None, None, None, None]
        self.white_sense_history = [None, None, None, None, None]

        self.board_before_sense = None
        self.board_after_step = None 
        self.board_after_sense = None
        self.taken_move_from_square = None

        self.done = False

        self.best_model = PPO.load(os.path.join(self.model_dir, "self_play_best_model"))

        #Alternate between players after each episode
        if self.num_episode % 2 == 0: 
            self.current_learning_player = chess.WHITE 

            self.white_player:Player = selfPlaySensingWSTCKF(train = True)
            # self.black_player:Player = StrangefishWStockfish()
            self.black_player:Player = selfPlaySensingWSTCKF(train = True)

            self.players = [self.black_player, self.white_player]

            self.white_name = self.white_player.__class__.__name__
            self.black_name = self.black_player.__class__.__name__
            
            self.game = LocalGame()
            self.game.store_players(self.white_name, self.black_name)

            self.board_white_obs= chess.Board(self.game.board.copy().epd(en_passant='xfen'))
            self.board_black_obs= chess.Board(self.game.board.copy().epd(en_passant='xfen'))

            # setting the turn in the observation board of black player to black. 
            self.board_black_obs.turn = chess.BLACK

            self.white_player.handle_game_start(chess.WHITE, self.board_white_obs, self.black_name)
            self.black_player.handle_game_start(chess.BLACK, self.board_black_obs, self.white_name)
            self.game.start()
            sense_history = self.white_sense_history
        else: 
            self.current_learning_player = chess.BLACK

            # self.white_player:Player = StrangefishWStockfish() 
            self.white_player:Player = selfPlaySensingWSTCKF(train = True)
            self.black_player:Player = selfPlaySensingWSTCKF(train = True)

            self.players = [self.black_player, self.white_player]

            self.white_name = self.white_player.__class__.__name__
            self.black_name = self.black_player.__class__.__name__
            
            self.game = LocalGame()
            self.game.store_players(self.white_name, self.black_name)

            self.board_white_obs= chess.Board(self.game.board.copy().epd(en_passant='xfen'))
            self.board_black_obs= chess.Board(self.game.board.copy().epd(en_passant='xfen'))

            # setting the turn in the observation board of black player to black. 
            self.board_black_obs.turn = chess.BLACK

            self.white_player.handle_game_start(chess.WHITE, self.board_white_obs, self.black_name)
            self.black_player.handle_game_start(chess.BLACK, self.board_black_obs, self.white_name)
            self.game.start()

            sense_history = self.black_sense_history

            # if current player is black player, play the first turn of the white player before calling step function 
            obs = self._next_observation(chess.WHITE, self.opponent_capture_square, self.white_sense_history)
            sense, _states = self.best_model.predict(obs)
            sense_action = revert_sense_square(sense, flip=False)
            sense_actions = [sense_action]
            # sense_actions = self.game.sense_actions()
            self.play_turn(self.game, self.players[self.game.turn], sense_actions, end_turn_last=True)
            self.white_sense_history.insert(0, sense)
            self.white_sense_history.pop()

        # self.board_set_reduction = 0
        print('+++++++++++++ EPISODE ', self.num_episode ,'as ', self.current_learning_player, '+++++++++++')

        return self._next_observation(self.current_learning_player, self.opponent_capture_square, sense_history)


    def notify_opponent_move_results(self, game: Game, player: Player):
        """
        Passes the opponents move results to the player. Does the following sequentially:

        #. Get the results of the opponents move using :meth:`Game.opponent_move_results`.
        #. Give the results to the player using :meth:`Player.handle_opponent_move_result`.

        :param game: The :class:`Game` that `player` is playing in.
        :param player: The :class:`Player` whose turn it is.
        """
        self.opponent_capture_square = game.opponent_move_results()
        player.handle_opponent_move_result(self.opponent_capture_square is not None, self.opponent_capture_square)

    def play_move(self, game: Game, player: Player, move_actions: List[chess.Move], end_turn_last=False):
        """
        Runs the move phase for `player` in `game`. Does the following sequentially:

        #. Get the moving action using :meth:`Player.choose_move`.
        #. Apply the moving action using :meth:`Game.move`.
        #. Ends the current player's turn using :meth:`Game.end_turn`.
        #. Give the result of the moveaction to player using :meth:`Player.handle_move_result`.

        If `end_turn_last` is True, then :meth:`Game.end_turn` is called last instead of before
        :meth:`Player.handle_move_result`.

        :param game: The :class:`Game` that `player` is playing in.
        :param player: The :class:`Player` whose turn it is.
        :param move_actions: The possible move actions for `player`.
        :param end_turn_last: Flag indicating whether to call :meth:`Game.end_turn` before or after :meth:`Player.handle_move_result`
        """
        move = player.choose_move(move_actions, game.get_seconds_left())
        requested_move, taken_move, self.opt_enemy_capture_square = game.move(move)
        if taken_move != None:
            self.taken_move_from_square = taken_move.from_square

        # print('taken move', taken_move)
        if not end_turn_last:
            game.end_turn()

        player.handle_move_result(requested_move, taken_move,
                                self.opt_enemy_capture_square is not None, self.opt_enemy_capture_square)

        if  self.opt_enemy_capture_square is not None: 
            if player.color == chess.WHITE:
                self.num_captured_black_pieces += 1  
                # print('num captured black pieces ', self.num_captured_black_pieces)
            else:
                self.num_captured_white_pieces += 1 
                # print('num captured white pieces ', self.num_captured_white_pieces)


        

        if end_turn_last:
            game.end_turn()


    def play_sense(self, game: Game, player: Player, sense_actions: List[Square], move_actions: List[chess.Move]):
        """
        Runs the sense phase for `player` in `game`. Does the following sequentially:

        #. Get the sensing action using :meth:`Player.choose_sense`.
        #. Apply the sense action using :meth:`Game.sense`.
        #. Give the result of the sense action to player using :meth:`Player.handle_sense_result`.

        :param game: The :class:`Game` that `player` is playing in.
        :param player: The :class:`Player` whose turn it is.
        :param sense_actions: The possible sense actions for `player`.
        :param move_actions: The possible move actions for `player`.
        """
        sense = player.choose_sense(sense_actions, move_actions, game.get_seconds_left())
        self.sense_result = game.sense(sense)
        self.board_set_reduction = player.handle_sense_result(self.sense_result)


    def play_turn(self, game:Game, player: Player, sense_action, end_turn_last = False):
        move_actions = game.move_actions()

        self.notify_opponent_move_results(game, player)

        # print('board before sense ','\n', self.board_before_sense)

        self.play_sense(game, player, sense_action, move_actions)

        # get board after sense 
        self.board_after_sense =  self.board_white_obs.copy() if self.current_learning_player == chess.WHITE else self.board_black_obs.copy()
        # print('board after sense ', '\n',  self.board_after_sense)

        self.play_move(game, player, move_actions, end_turn_last)


    def get_reward(self, player_color): 
        piece_at_sense_square: Optional[chess.Piece] = self.sense_result[4][1]
        reward = 0
        
        # if we sensed on our piece 
        if piece_at_sense_square != None:
            if piece_at_sense_square.color == player_color:
                reward -= 5

        # Add reward if there is a change in the sensed squares from before to after sensing
        # reward based on sensed pieces values
        # print('sense result ', self.sense_result)
        for square, piece in self.sense_result:
            if self.board_before_sense.piece_at(square) != self.board_after_sense.piece_at(square):
                # if it's not a change caused by current player move
                # print('piece', piece)
                if piece == None and square != self.taken_move_from_square:
                    reward += 1
                elif piece != None:
                    piece_color = piece.color
                    piece_type = piece.piece_type
                    if piece_color != player_color: 
                        if piece_type == chess.PAWN:
                            reward += 5  
                        elif piece_type == chess.KNIGHT:
                            reward += 5
                        elif piece_type == chess.BISHOP:
                            reward += 5
                        elif piece_type == chess.ROOK: 
                            reward += 5
                        elif piece_type == chess.QUEEN: 
                            reward += 10
                        elif piece_type == chess.KING: 
                                # Encourage the agent to sense the king as the number of opt pieces is decreasing
                                num_pieces_left = self.num_captured_white_pieces if player_color == chess.WHITE else self.num_captured_black_pieces
                                # print('KING REWARD ', 1 - (num_pieces_left / 16) ** 0.5)
                                reward += (1 - (num_pieces_left / 16) ** 0.5)

                
                # print('there is a change in the board and reward is ', reward)
        
        # print('reward board set reduction ',  self.board_set_reduction)
        # Reward for board set reduction 
        # reward += self.board_set_reduction

        return reward

    def _next_observation(self, player, opt_capture_square, sense_history): 
        self.board_before_sense = self.board_white_obs.copy() if player == chess.WHITE else self.board_black_obs.copy()

        # Use current player's move result to update next player's observation
        if opt_capture_square != None: 
            self.board_before_sense.remove_piece_at(opt_capture_square)

        obs = generate_input_for_model(self.board_before_sense, opt_capture_square, sense_history)
        return obs
    
    def play_next_step(self, action, player_color, player_sense_history, enemy_sense_history, flip):
        sense_action = revert_sense_square(action, flip)
        reward = 0
        # print('Starting turn of ',self.game.turn)
        # print('Sense action ', sense_action)
        self.play_turn(self.game, self.players[self.game.turn], [sense_action], end_turn_last=True)
        # print('Board after move ', '\n', self.board_white_obs if player_color == chess.WHITE else self.board_black_obs)
        reward = self.get_reward(player_color)
        player_sense_history.insert(0, action)
        player_sense_history.pop()

        obs = self._next_observation(self.game.turn, self.opt_enemy_capture_square, self.black_sense_history)
        sense, _states = self.best_model.predict(obs)
        sense_action = revert_sense_square(sense, not flip)
        sense_actions = [sense_action]
        # print('Starting turn of ',self.game.turn)
        # print('Sense action ', sense_action)
        # sense_actions = self.game.sense_actions()
        self.play_turn(self.game, self.players[self.game.turn], sense_actions, end_turn_last=True)
        # print('Board after move ', '\n', self.board_black_obs if player_color == chess.WHITE else self.board_white_obs)
        enemy_sense_history.insert(0, sense)
        enemy_sense_history.pop()
        obs = self._next_observation(player_color, self.opt_enemy_capture_square, self.white_sense_history)

        return obs, reward


    
    def step(self, action):
        
        if self.current_learning_player == chess.WHITE: 
            obs, reward = self.play_next_step(action, chess.WHITE, self.white_sense_history, self.black_sense_history, False)

        elif self.current_learning_player == chess.BLACK: 
            obs, reward = self.play_next_step(action, chess.BLACK, self.black_sense_history, self.white_sense_history, True)

        # print('Reward ', reward)
        

        if self.game.is_over():
            self.done = True
            if self.game.get_winner_color() == self.current_learning_player: 
                print(f'Current Player {self.current_learning_player} won')
                reward += 100
            else:
                print(f'Current Player {self.current_learning_player} Lost')
                reward -= 100
            

        if self.done: 
            winner_color = self.game.get_winner_color()
            win_reason = self.game.get_win_reason()
            game_history = self.game.get_game_history()
            # print('final reward ', reward)
            self.render_episode(winner_color)
            self.num_episode += 1
            self.white_player.handle_game_end(winner_color, win_reason, game_history)
            self.black_player.handle_game_end(winner_color, win_reason, game_history)

            
        self.num_step += 1

        return obs, reward, self.done, {}



    def render_episode(self, winner_color):
        self.success_episode.append('Winner is ' + str(winner_color))

        file = open('C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester4\\RBC\\RBC-Agent\\self-play-log\\render.txt', 'a')
        file.write('-------------------------------------------\n')
        file.write(f'Episode number {self.num_episode}\n')
        file.write(f'{self.success_episode[-1]} in {self.num_step} steps\n')
        file.close()
