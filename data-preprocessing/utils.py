from cProfile import label
from turtle import position
import numpy as np


def get_row_col(position, mirror=False):
    """
    Maps a value [0,63] to its row and column index
    :param position: Position id which is an integer [0,63]
    :param mirror: Returns the indices for the mirrored board
    :return: Row and columns index
    """
    # returns the column and row index of a given position
    row = position // 8
    col = position % 8

    if mirror:
        row = 7 - row

    return row, col


def get_board_position_index(row, col, mirror=False):
    """
    Maps a row and column index to the integer value [0, 63].
    :param row: Row index of the square
    :param col: Column index of the square
    :param mirror: Returns integer value for a mirrored board
    :return:
    """
    if mirror:
        row = 7 - row

    return (row * 8) + col


def get_sense_plane(sense_pos):
    labels = np.zeros((37, 8, 8))

    for sense_pos in range(1, 37): 
        row = (sense_pos - 1) // 6
        col = (sense_pos - 1) % 6

        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                labels[sense_pos][row + 1 + delta_rank][col + 1 + delta_file] = 1

    return labels[sense_pos]
