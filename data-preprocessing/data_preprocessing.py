import json
from operator import contains
from re import M
from traceback import print_tb
from unittest import result
import chess
import chess.engine
import random
import numpy as np
from myBot.utilities import stockfish
import os
from utils import get_board_position_index, get_row_col
import h5py
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import pickle

WEAK_BOTS = ['ai_games_cvi','random', 'armandli', 'URChIn', 'attacker', 'callumcanavan', 'trout', 'GarrisonNRL', 'Frampt', 'wbernar5', 'DynamicEntropy' ]


# H5_TRAIN = "RBC-data-2021-train-sample.hdf5"
# H5_VAL = "RBC-data-2021-val-sample.hdf5"
# H5_TEST = "RBC-data-2021-test-sample.hdf5"


H5_TRAIN = "C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester 4\\RBC\\RBC-Agent\\data\\RBC-data-2021-train.hdf5"
H5_VAL = "C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester 4\\RBC\\RBC-Agent\\data\\RBC-data-2021-val.hdf5"
H5_TEST = "C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester 4\\RBC\\RBC-Agent\\data\\RBC-data-2021-test.hdf5"


def get_file_names():
    path = 'C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester 4\\RBC\\neurips2021_RBC_game_logs\\neurips2021_histories\\'
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    return json_files


def get_game_data(boards, senses):
    x = []
    y = []
    break_out_flag = False
    for player in boards:
        for idx, board in enumerate(boards[player]):
            sense = senses[player][idx]
            board = chess.Board(board)

            # this is the 3d matrix
            board3d = np.zeros((20, 8, 8), dtype=np.float64)

            me = board.turn
            you = not board.turn

            # Add the pieces's view on the matrix x 6 x 2
            # Plane for each piece type for each player 
            for piece in chess.PIECE_TYPES:
                for square in board.pieces(piece, chess.WHITE):
                    row, col = get_row_col(square)
                    board3d[piece - 1][row][col] = 1
                for square in board.pieces(piece, chess.BLACK):
                    row, col = get_row_col(square)
                    board3d[piece + 5][row][col] = 1

            # Color x 1
            if board.turn == chess.WHITE: board3d[12, :, :] = 1

            # En Passant Square x 1
            if board.ep_square is not None: 
                row, col = get_row_col(board.ep_square)
                board3d[13, row, col] = 1
            
            
            # Castling Rights for both sides x 2 x 2
            if board.has_kingside_castling_rights(me): board3d[14, :, :] = 1
            if board.has_queenside_castling_rights(me): board3d[15, :, :] = 1
            if board.has_kingside_castling_rights(you): board3d[16, :, :] = 1
            if board.has_queenside_castling_rights(you): board3d[17, :, :] = 1

            
            for square in chess.PIECE_TYPES:
                # Plane representing White Pieces  
                for square in board.pieces(piece, chess.WHITE): 
                    row, col = get_row_col(square)
                    board3d[18][row][col] = 1
                # Plane representing Black Pieces
                for square in board.pieces(piece, chess.BLACK):
                    row, col = get_row_col(square)
                    board3d[19][row][col] = 1

            sense_square = 0
            sense_plane =  np.zeros((8, 8), dtype=np.float64)
            if sense != None: 
                row, col = get_row_col(sense)
                offset = 0
                if row >= 2: offset = row + (row - 2)
                sense_square = sense - (8 + offset)
                if sense_square not in range(0, 37):
                    break_out_flag = True    
                    break
                # else: 
                #     for delta_rank in [1, 0, -1]:
                #         for delta_file in [-1, 0, 1]:
                #             sense_plane[row + delta_rank][col + delta_file] = 1
                # print('sense ', sense)
                # print('plane ', sense_plane)

            x.append(board3d)
            y.append(sense_square)
        if break_out_flag: 
            x, y = [], []
            break
    return x, y


def create_dataset_file(filenames, x_data, y_data):
    script_dir = os.path.dirname(__file__)
    num_omitted_files = 0
    for f1 in tqdm(filenames): 
        path = 'C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester 4\\RBC\\neurips2021_RBC_game_logs\\neurips2021_histories\\' + str(f1)
        file_path = os.path.join(script_dir, path)
        with open(file_path, 'r') as infile: # open file 
            values = json.load(infile) # values in the file 

            # Omit data from weak bots
            if (( values['white_name'] in WEAK_BOTS) and (values['black_name'] in WEAK_BOTS)):
                num_omitted_files += 1
                continue

            boards_bm = values['fens_before_move'] # boards positions before move for both players in one game
            senses = values['senses'] # all sense made by both players in one game
            
            boards_am = values['fens_after_move'] # boards positions after move for both players in one game
            requested_moves = values['requested_moves'] # all requested moves by both players in one game
            taken_moves = values['taken_moves'] # all taken moves by both players in one game
            capture_squares = values['capture_squares'] # all captures made by both players after each move in one game

            x, y = get_game_data(boards_bm, senses)

            x = np.asarray(x)
            y = np.asarray(y)
            # print('x ', x.shape)
            # print('y ', y.shape )    

            # Check whether the arrays are empty
            if (x.shape != (0,)) and (y.shape != (0,)):
                x_data.resize(x_data.shape[0] + x.shape[0], axis= 0)
                x_data[-x.shape[0]:] = x

                y_data.resize(y_data.shape[0] + y.shape[0], axis = 0) 
                y_data[-y.shape[0]:] = y
    print('x data shape ', x_data.shape[0])
    print('y data shape ', y_data.shape[0])
    print('{} files omitted from {} '.format(len(filenames), num_omitted_files))
                    

def split_dataset(val_file, test_file): 
    f = h5py.File(H5_TRAIN, 'r+')
    x_train_data = f['data'][...]
    y_train_data = f['labels'][...]

    X_train, X_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.2)
    X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, test_size= 0.1)

    f['data'].resize(X_train.shape[0], axis = 0)
    f['data'][...] = X_train
    f['labels'].resize(y_train.shape[0], axis = 0)
    f['labels'][...] = y_train

    
    val_file.create_dataset('data', data= X_val)
    val_file.create_dataset('labels', data= y_val)

   
    test_file.create_dataset('data', data= X_test)
    test_file.create_dataset('labels', data= y_test)

    print('x data shape ',f['data'].shape[0])
    print('y data shape ', f['labels'].shape[0])


train_file = h5py.File(H5_TRAIN, 'w')
val_file = h5py.File(H5_VAL, 'w')
test_file = h5py.File(H5_TEST, 'w')

# train_data = train_file.create_group('train_data')
# val_data = val_file.create_group('val_data')
# test_data = test_file.create_group('test_data')

x_train_data = train_file.create_dataset('data', shape=(0,20,8,8), maxshape=(None, None, None, None))
y_train_data = train_file.create_dataset('labels', shape=(0,), maxshape=(None, ))


json_files = get_file_names()

start = datetime.now()
create_dataset_file(json_files, x_train_data, y_train_data)
split_dataset(val_file, test_file)
end = datetime.now()
print('create dataset duration ', end - start)
train_file.close()



