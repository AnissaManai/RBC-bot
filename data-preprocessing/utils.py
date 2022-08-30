import numpy as np
import itertools


def get_row_col(position, flip = False, width = 8):
    """
    Maps a value [0,63] to its row and column index
    :param position: Position id which is an integer [0,63]
    :param flip: Returns the indices for the flipped board
    :return: Row and columns index
    """
    # returns the column and row index of a given position
    row = position // width
    col = position % width

    if flip: 
        row = width - 1 - row
        col = width - 1 - col

    return row, col

def get_sense_square(position, flip = False):
    row, col = get_row_col(position, flip)
    offset = 0
    sense_square = None
    if row >= 2: offset = row + (row - 2) 
    if flip: 
        sense_square = (63 - position) - offset - 9
    else: 
        sense_square = position - offset - 9
    return sense_square

# pos = list(itertools.chain(range(9, 15), range(17, 23), range(25, 31), range(33, 39), range(41, 47), range(49, 55)))


def get_board_position_index(row, col, flip=False, width = 8):
    """
    Maps a row and column index to the integer value [0, 63].
    :param row: Row index of the square
    :param col: Column index of the square
    :param flip: Returns integer value for a flipped board
    :return:
    """
    if flip:
        row = (width - 1) - row

    return (row * width) + col



def get_sense_plane(sense_pos):
    labels = np.zeros((37, 8, 8))

    for sense_pos in range(1, 37): 
        row = (sense_pos - 1) // 6
        col = (sense_pos - 1) % 6

        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                labels[sense_pos][row + 1 + delta_rank][col + 1 + delta_file] = 1

    return labels[sense_pos]
