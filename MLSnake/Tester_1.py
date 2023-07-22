from . import config as cfg
import numpy as np
import math


def get_state(snake_list, food):
    '''Gets the current visual state of the game in pixels, and returns it as a 2D NumPy array.'''

    # Begin with all pixels as unoccupied, empty pixels.
    state = np.zeros((cfg.game_size, cfg.game_size), dtype=cfg.memtype)

    # Set the snake pixels.
    for seg in snake_list:
        state[seg[1]][seg[0]] = 1

    # Set the food pixel.
    state[food[1]][food[0]] = 2

    return state


test_state = []

for i in range(cfg.stack_size):
    snake_list = []
    for x in range(5):
        snake_list.append([math.floor(cfg.game_size * 3/8) - x + i,
                           math.floor(cfg.game_size * 1 / 2)])
    food = [0, 0]

    test_state.append(get_state(snake_list, food))
