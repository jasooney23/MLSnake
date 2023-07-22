# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
import time

from MLSnake import Game_2 as snake
from MLSnake import Agent_5 as agent
from MLSnake import config as cfg
from MLSnake import Tester_1 as test

game = snake.game()
snake_agent = agent.agent(game)
x = 0

while True:
      snake_agent.start()

      if x % cfg.autosave_period == 0:
            print()
            print()

            print("SAVING IN 5 SECONDS")
            time.sleep(5)
            snake_agent.save_all(cfg.save_path)

            print("BACKING UP IN 5 SECONDS")
            time.sleep(5)
            snake_agent.save_all(cfg.backup_path)

            print("Frames played: " + str(snake_agent.train_data["frames_played"]))
            print("Average score: " +
                  str(sum(snake_agent.stats["score"]) / len(snake_agent.stats["score"])))
            print(f"Average score (Last {cfg.autosave_period} episodes): " 
                  + str(sum(snake_agent.stats["score"][-cfg.autosave_period:]) / cfg.autosave_period))
            print("Average loss: " +
                  str(sum(snake_agent.stats["loss"]) / len(snake_agent.stats["loss"])))
            print("Average Q Value: " +
                  str(sum(snake_agent.stats["testing"]) / len(snake_agent.stats["testing"])))

            print()
            print()
      x += 1
