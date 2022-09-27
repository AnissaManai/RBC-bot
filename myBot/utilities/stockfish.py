""" 
    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
    and https://github.com/ginoperrotta/reconchess-strangefish2
    Copyright (c) 2021, The Johns Hopkins University Applied Physics Laboratory LLC
"""

import os

import chess.engine



def create_engine(engine_name):
    # engine_name = "FAIRYSTOCKFISH_EXECUTABLE"
    # engine_name = "STOCKFISH_11_EXECUTABLE"
    # make sure stockfish environment variable exists
    if engine_name not in os.environ:
        raise KeyError('This bot requires an environment variable called "STOCKFISH_EXECUTABLE"'
                    ' pointing to the Stockfish executable')
    # make sure there is actually a file
    STOCKFISH_EXECUTABLE = os.getenv(engine_name)
    if not os.path.exists(STOCKFISH_EXECUTABLE):
        raise ValueError('No stockfish executable found at "{}"'.format(STOCKFISH_EXECUTABLE))
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_EXECUTABLE)
    return engine
