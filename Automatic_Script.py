# Import packages.
import queue
import random
import threading
import math
import time
import pickle
import tensorflow as tf             # Tensorflow.
# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
import tkinter as tkinter           # Required to control the game.
import matplotlib.pyplot as plt     # Plotting statistics.

from MLSnake import Game_2 as snake
from MLSnake import Agent_2 as agent
from MLSnake import config as cfg

game = snake.game()
snake_agent = agent.agent(game)
x = 0

while True:
    snake_agent.start()

    if x % cfg.autosave_period == 0:
        snake_agent.save_all()
        print("Average score: " +
              str(sum(snake_agent.stats["score"]) / len(snake_agent.stats["score"])))
        print("Average loss: " +
              str(sum(snake_agent.stats["loss"]) / len(snake_agent.stats["loss"])))
    x += 1

    # Trackthe score/performance of the agent.
    snake_agent.stats["score"].append(game.score)
