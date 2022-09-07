import numpy as np
import gym
from gym import spaces
import chess
from typing import List, Optional

from reconchess import LocalGame, Player, Game, Square
from myBot import SelfPlayBot
from myBot.utilities import generate_input_for_model, revert_sense_square

class RBCEnv(gym.Env):

    def __init__(self):

        self.action_space = spaces.Discrete(36)
        self.observation_space = spaces.Box(low=0, high=1, shape=(21, 8,8), dtype = np.uint8)

        self.reward_range = (-1,1)

        self.current_episode = 0
        self.success_episode = []


        self.white_player:Player = SelfPlayBot(rc_disable_pbar=True)
        self.black_player:Player = SelfPlayBot(rc_disable_pbar=True)

        self.players = [self.black_player, self.white_player]


        self.white_name = self.white_player.__class__.__name__
        self.black_name = self.black_player.__class__.__name__

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

    def reset(self):
        # print('reset')
        self.game = LocalGame()
        self.game.store_players(self.white_name, self.black_name)
        self.board_white_obs= chess.Board(self.game.board.copy().epd(en_passant='xfen'))
        self.board_black_obs= chess.Board(self.game.board.copy().epd(en_passant='xfen'))
        self.board_black_obs.turn = chess.BLACK
        # self.board = self.game.board.copy()

        self.white_player.handle_game_start(chess.WHITE, self.board_white_obs, self.black_name)
        self.black_player.handle_game_start(chess.BLACK, self.board_black_obs, self.white_name)
        self.game.start()

        self.current_step = 0

        self.board_set_reduction = 0

        return self._next_observation(self.opponent_capture_square, self.white_sense_history, chess.WHITE)


    def _next_observation(self, opt_capture_square, sense_history, next_player): 
        
        self.board_before_sense = self.board_white_obs.copy() if next_player == chess.WHITE else self.board_black_obs.copy()


        # Use current player's move result to update next player's observation
        if opt_capture_square != None: 
            self.board_before_sense.remove_piece_at(opt_capture_square)

        obs = generate_input_for_model(self.board_before_sense, opt_capture_square, sense_history)
        return obs


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

        if not end_turn_last:
            game.end_turn()

        player.handle_move_result(requested_move, taken_move,
                                self.opt_enemy_capture_square is not None, self.opt_enemy_capture_square)

        if  self.opt_enemy_capture_square is not None: 
            if player.color == chess.WHITE:
                self.num_captured_black_pieces += 1  
            else:
                self.num_captured_white_pieces += 1 

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

        self.play_sense(game, player, sense_action, move_actions)

        # get board after sense 
        self.board_after_sense =  self.board_white_obs if self.game.turn == chess.WHITE else self.board_black_obs
        # print('board after sense ', '\n',  self.board_after_sense)

        self.play_move(game, player, move_actions, end_turn_last)


    def get_reward(self, opponent_color): 
        piece_at_sense_square: Optional[chess.Piece] = self.sense_result[4][1]
        reward = 0
        
        # if we sensed on our piece 
        if piece_at_sense_square != None:
            if piece_at_sense_square.color != opponent_color:
                reward -= 0.1


        # Add reward if there is a change in the sensed squares from before to after sensing
        # reward based on sensed piece's values
        for square, piece in self.sense_result:
            if self.board_before_sense.piece_at(square) != self.board_after_sense.piece_at(square):
                # if it's not a change caused by current player move
                if piece == None and square != self.taken_move_from_square:
                    reward += 1
                elif piece != None:
                    piece_color = piece.color
                    piece_type = piece.piece_type
                    if piece_color == opponent_color: 
                        match piece_type: 
                            case chess.PAWN:
                                reward += 1
                            case chess.KNIGHT:
                                reward += 3
                            case chess.BISHOP:
                                reward += 3
                            case chess.ROOK: 
                                reward += 5
                            case chess.QUEEN: 
                                reward += 9
                            case chess.KING: 
                                # Encourage the agent to sense the king as the number of opt pieces is decreasing
                                num_pieces_left = self.num_captured_white_pieces if opponent_color == chess.WHITE else self.num_captured_black_pieces
                                print('KING REWARD ', 1 - (num_pieces_left / 16) ** 0.5)
                                reward += (1 - (num_pieces_left / 16) ** 0.5)

                
                # print('there is a change in board and reward is ', reward)
        
        # print('reward board set reduction ',  self.board_set_reduction)
        # Reward for board set reduction 
        reward += self.board_set_reduction

        

        return reward



    def step(self, action):
        current_player: SelfPlayBot = self.players[self.game.turn]
        current_player_color = self.game.turn
        opponent_color = not self.game.turn
        action = revert_sense_square(action, current_player_color == chess.BLACK)
        sense_action = [action]
        
        print('PLAY TURN ', current_player_color)
        print('sense action ', action)

        # board_before_step = self.board_white_obs if current_player_color == chess.WHITE else self.board_black_obs

        # print('board before sense of ', current_player_color, '\n', self.board_before_sense)
        
        self.play_turn(self.game, current_player, sense_action, end_turn_last=True)
        self.board_after_step =  self.board_white_obs if current_player_color == chess.WHITE else self.board_black_obs

        # print('board after turn of ', current_player_color, '\n',  self.board_after_step)

        # update the sense history of current player 
        # get the observation for the next player 
        if current_player_color == chess.WHITE: 
            self.white_sense_history.insert(0, action)
            self.white_sense_history.pop()
            obs = self._next_observation(self.opt_enemy_capture_square, self.black_sense_history, chess.BLACK)

            # print('opt enemy capture square ', self.opt_enemy_capture_square)
            # print('sense history white ', self.white_sense_history)
        else: 
            self.black_sense_history.insert(0, action)
            self.black_sense_history.pop()
            obs = self._next_observation(self.opt_enemy_capture_square, self.white_sense_history, chess.WHITE)

            # print('opt enemy capture square ', self.opt_enemy_capture_square)
            # print('sense history white ', self.black_sense_history)

        # print('sense result ', self.sense_result)
        reward = self.get_reward(opponent_color)

        print('final reward ', reward)
        

        
        done = False
        if self.game.is_over():
            done = True
            if self.game.get_winner_color() == current_player_color: 
                print(f'Player {current_player_color} won')
                reward += 100
            elif self.game.get_winner_color() == opponent_color:
                print(f'Player {current_player_color} Lost')
                reward -= 100
            else:
                reward -= 50
            

        if done: 
            winner_color = self.game.get_winner_color()
            win_reason = self.game.get_win_reason()
            game_history = self.game.get_game_history()
            print('done! winner is  ', winner_color)
            self.render_episode(winner_color)
            self.current_episode += 1
            
            self.white_player.handle_game_end(winner_color, win_reason, game_history)
            self.black_player.handle_game_end(winner_color, win_reason, game_history)


        return obs, reward, done, {}



    def render_episode(self, winner_color):
        self.success_episode.append('Winner is ' + str(winner_color))

        file = open('..\\self-play-log\\render.txt', 'a')
        file.write('-------------------------------------------\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(f'{self.success_episode[-1]} in {self.current_step} steps\n')
        file.close()
