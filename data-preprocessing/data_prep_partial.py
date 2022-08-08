from asyncio.windows_events import NULL
import chess
import chess.engine
import numpy as np
import os
from utils import get_row_col
import h5py
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json



# WEAK_BOTS = ['ai_games_cvi','random', 'armandli', 'URChIn', 'attacker', 'callumcanavan', 'trout', 'GarrisonNRL', 'Frampt', 'wbernar5', 'DynamicEntropy' ]

TOP3_BOTS = ['Fianchetto', 'StrangeFish2', 'penumbra']
# TOP2_BOTS = ['Fianchetto', 'StrangeFish2']

H5_TRAIN = "C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester4\\RBC\\RBC-Agent\\data\\top3-partial-train.hdf5"
H5_VAL = "C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester4\\RBC\\RBC-Agent\\data\\top3-partial-val.hdf5"
H5_TEST = "C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester4\\RBC\\RBC-Agent\\data\\top3-partial-test.hdf5"


def get_file_names():
    path = 'C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester4\\RBC\\neurips2021_RBC_game_logs\\neurips2021_histories\\'
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    return json_files


def get_game_data(boards, senses, sense_results, capture_squares, player):
    '''
        Board3d: 
        0 - 5: curr player pieces 
        6 - 11 : opp player pieces 
        12 : Color of player( 0 for black, 1 for white)
        13 : White pieces position
        14 : Black pieces position 
        15 : capture square of opp in last move
        16 - 20 : sense history 
    '''
    x = []
    y = []
    skip_board = False
    

    
    opponent_pieces_planes = np.zeros((6, 8, 8), dtype=np.float64)

    for idx, board in enumerate(boards):
        board3d = np.zeros((21, 8, 8), dtype=np.float64)

        sense = senses[idx]
        board = chess.Board(board)
        me = board.turn


        # TODO: add step counter for each position update 
        # Plane for each piece type for current player
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, me):
                row, col = get_row_col(square, mirror = True)
                if me == chess.WHITE: 
                    board3d[piece - 1][row][col] = 1
                    # Plane representing current player pieces 
                    board3d[12][row][col] = 1
                else: 
                    board3d[piece + 5][row][col] = 1
                    # Plane representing current player pieces 
                    board3d[13][row][col] = 1

        # Color x 1
        if board.turn == chess.WHITE: board3d[14, :, :] = 1

        # capture square after opponent move 
        opponent_capture_square = None
        if player == "false":
            opponent = "true"
            opponent_capture_square = capture_squares[opponent][idx]
        elif player == "true" and idx != 0:
            opponent = "false"
            opponent_capture_square = capture_squares[opponent][idx - 1]
        if opponent_capture_square != None:
            row, col = get_row_col(opponent_capture_square, mirror = True)
            board3d[15][row][col] = 1
        
        # Add history of sense locations of the last 5 steps
        if idx > 0:
            plane_index = 16
            for i in range(1, 6): 
                prev_sens_idx = idx - i
                if(prev_sens_idx >= 0): 
                    prev_sense = senses[prev_sens_idx]

                    if prev_sense != None: 
                        row, col = get_row_col(prev_sense, mirror = True)
                        offset = 0
                        if row >= 2: offset = row + (row - 2)
                        sense_square = prev_sense - (8 + offset)
                        if sense_square in range(0, 37):
                            for delta_rank in [1, 0, -1]:
                                for delta_file in [-1, 0, 1]:
                                    board3d[plane_index][row + delta_rank][col + delta_file] = 1
                    plane_index += 1

            # Update opponent pieces locations based on previous sense result
            prev_sense_result = sense_results[idx -1]
            for square, piece in prev_sense_result: 
                piece = chess.PIECE_SYMBOLS.index(piece['value'].lower()) if piece != None else None
                if piece != None:
                    row, col = get_row_col(square, mirror = True)
                    opponent_pieces_planes[piece - 1][:][:] = 0
                    opponent_pieces_planes[piece - 1][row][col] = 1

                    if me == chess.WHITE: 
                        board3d[6:12][:][:] = opponent_pieces_planes
                    else: 
                        board3d[:6][:][:] = opponent_pieces_planes

        sense_square = 0
        sense_plane =  np.zeros((8, 8), dtype=np.float64)
        if sense != None: 
            row, col = get_row_col(sense, mirror = True)
            offset = 0
            if row >= 2: offset = row + (row - 2)
            sense_square = sense - (8 + offset)
            if sense_square not in range(0, 37):
                skip_board = True

        if not skip_board:
            x.append(board3d)
            y.append(sense_square)
        skip_board = False
            
    return x, y

def append_to_file(x_data, y_data, x, y):
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



def create_dataset_file(filenames, x_data, y_data):
    script_dir = os.path.dirname(__file__)
    for f1 in tqdm(filenames): 
        path = 'C:\\Users\\aniss\\OneDrive\\Dokumente\\Uni\\Semester4\\RBC\\neurips2021_RBC_game_logs\\neurips2021_histories\\' + str(f1)
        file_path = os.path.join(script_dir, path)
        with open(file_path, 'r') as infile: # open file 
            values = json.load(infile) # values in the file 
            boards_bm = []
            senses = []
            player = ''

            capture_squares = values['capture_squares'] # all captures made by both players after each move in one game

            # Get data of top 3 bots
            if values['white_name'] in TOP3_BOTS:
                player = 'true'  
                boards_bm = values['fens_before_move'][player] # boards positions before move for White player in one game
                senses = values['senses'][player] # all sense made by White player in one game  
                sense_results = values['sense_results'][player] # sense results 
                moves = values['requested_moves'][player]
                x, y = get_game_data(boards_bm, senses, sense_results, capture_squares, player)
                append_to_file(x_data, y_data, x, y)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


            if values['black_name'] in TOP3_BOTS:
                player = 'false'
                boards_bm = values['fens_before_move'][player] # boards positions before move for Black player in one game
                senses = values['senses'][player] # all sense made by Black player in one game
                sense_results = values['sense_results'][player] # sense results 
                x, y = get_game_data(boards_bm, senses, sense_results, capture_squares, player)
                append_to_file(x_data, y_data, x, y)


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

    print('train data shape ',f['data'].shape[0])
    print('train label shape ', f['labels'].shape[0])

    print('val data shape ', val_file['data'].shape[0])
    print('val label shape ', val_file['labels'].shape[0])

    print('test data shape', test_file['data'].shape[0])
    print('test label shape ', test_file['labels'].shape[0])


train_file = h5py.File(H5_TRAIN, 'w')
val_file = h5py.File(H5_VAL, 'w')
test_file = h5py.File(H5_TEST, 'w')

x_train_data = train_file.create_dataset('data', shape=(0,21,8,8), maxshape=(None, None, None, None))
y_train_data = train_file.create_dataset('labels', shape=(0,), maxshape=(None, ))


json_files = get_file_names()

start = datetime.now()
create_dataset_file(json_files, x_train_data, y_train_data)
split_dataset(val_file, test_file)
end = datetime.now()
print('create dataset duration ', end - start)
train_file.close()