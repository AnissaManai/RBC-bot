""" 
    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
    and https://github.com/ginoperrotta/reconchess-strangefish2
    Copyright (c) 2021, The Johns Hopkins University Applied Physics Laboratory LLC
"""

import traceback
from datetime import datetime

import chess
from reconchess import LocalGame, play_local_game
from reconchess.bots.trout_bot import TroutBot
from reconchess.bots.random_bot import RandomBot
from myBot import EruditeBot

from myBot import myBot

def main(): 
    white_bot_name, black_bot_name = 'RBC-bot', 'TroutBot',

    game = LocalGame()

    try: 
        winner_color, win_reason, history = play_local_game( 
            TroutBot(),
            EruditeBot(), 
            game = game
        )

        winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    except: 
        traceback.print_exc()
        game.end()

        winner = 'ERROR'
        history = game.get_game_history()

    print('Game Over!')
    print('Winner: {}!'.format(winner))

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')    

    replay_path = '{}-{}-{}-{}.json'.format(white_bot_name, black_bot_name, winner, timestamp)
    print('Saving replay to {}...'.format(replay_path))
    history.save(replay_path)


if __name__ == '__main__':
    main()