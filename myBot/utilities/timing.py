""" 
    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
    and https://github.com/ginoperrotta/reconchess-strangefish2
    Copyright (c) 2021, The Johns Hopkins University Applied Physics Laboratory LLC
"""

from time import time


class Timer:
    def __init__(self, log_func, message: str):
        self.log_func = log_func
        self.message = message
        self.start = None

    def __enter__(self):
        self.start = time()
        # self.log_func('Starting ' + self.message)

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time() - self.start
        self.log_func("Finished " + self.message + f" in {duration:,.4g} seconds.")