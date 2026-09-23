"""
Conversions between a robot's (x, y, orientation) on the hex-oriented grid
and its flat state index into the transition matrix:

    state = 6 * (x + grid_size * y) + orientation

orientation is one of 6 headings, 60 degrees apart (0, 60, 120, 180, 240, 300).
"""


def info_to_state(x, y, orientation, grid_size):
    return 6 * (x + grid_size * y) + orientation


def state_to_info(state, grid_size):
    orientation = state % 6
    state //= 6
    y = state // grid_size
    x = state % grid_size
    return x, y, orientation
