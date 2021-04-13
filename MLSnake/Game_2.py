'''This is a special version of the snake game that allows for
    easy adjustment of the size of the game. Unlike the original
    snake file, this one cannot take human input (yet), and is
    for the agent's environment only.'''

'''Changes to v2:
    See Agent_2.py.'''


# Import packages.
import numpy as np
import random
import time
import tkinter as tk
import math
from . import config as cfg
class game:
    '''Creates an instance of a game. The game shows a window,
       which advances one frame with every call of step().'''

    def __init__(self):
        '''Init function.'''

        # Create the window of appropriate size. 200 physical pixels are added to the window, giving space for debug info.
        self.w = tk.Tk()
        self.w.geometry(str(cfg.screen_size + 200) +
                        "x" + str(cfg.screen_size))
        self.w.resizable(0, 0)

        # Define the canvas, which the pixels are drawn on.
        self.canvas = tk.Canvas(self.w, bg="#000000",
                                width=cfg.screen_size + 200, height=cfg.screen_size)
        self.canvas.pack()

    # ====================================================================== #
    # ====================================================================== #
    '''State reporting functions for the agent.'''

    def get_state(self):
        '''Gets the current visual state of the game in pixels, and returns it as a 2D NumPy array.'''

        # Begin with all pixels as unoccupied, empty pixels.
        state = np.zeros((cfg.game_size, cfg.game_size))

        # Set the snake pixels.
        for seg in self.snake_list:
            state[seg[1]][seg[0]] = 0.5

        # Set the food pixel.
        state[self.food[1]][self.food[0]] = 1

        return state

    def distance_to_food(self, seg):
        '''Calculates the total distance to the food from the head of the snake in pixels (diagonals count as 2).'''

        diff_x = abs(seg[0] - self.food[0])
        diff_y = abs(seg[1] - self.food[1])

        return diff_x + diff_y

    # ====================================================================== #
    # ====================================================================== #
    '''GUI drawing functions.'''

    def draw_pixel(self, pos, fill):
        '''Draws a pixel at a given position with a given fill colour.

           Function Parameters:
           pos <int[2]> = position for the pixel to be drawn in.'''

        self.canvas.create_rectangle(pos[0] * cfg.snake_size, pos[1] * cfg.snake_size,
                                     (pos[0] + 1) * cfg.snake_size, (pos[1] + 1) * cfg.snake_size, fill=fill)

    def draw_snake(self):
        '''Draws the entire snake.'''

        for seg in self.snake_list:
            self.draw_pixel(seg, "#888888")

    def draw_food(self):
        '''Draws the food.'''

        self.draw_pixel(self.food, "#ffffff")

    def draw_sidebar(self, values):
        '''Draws a sidebar to the right of the game that displays debug values.

           Function Parameters:
           values <float[5]> = contains the model's confidence for each action.'''

        # Draw it 5 * 50 physical pixels wide, and 18 * 50 tall.
        for x in range(0, 5):
            for y in range(0, 18):
                self.canvas.create_rectangle(cfg.screen_size + x * 50, y * 50,
                                             cfg.screen_size + (x + 1) * 50, (y + 1) * 50, fill="white", outline="white")

        self.canvas.create_text(cfg.screen_size + 100, 60, fill="black", text="UP: " +
                                str(values[0][0]), font=("Arial", 13))
        self.canvas.create_text(cfg.screen_size + 100, 80, fill="black", text="DOWN: " +
                                str(values[0][1]), font=("Arial", 13))
        self.canvas.create_text(cfg.screen_size + 100, 100, fill="black", text="LEFT: " +
                                str(values[0][2]), font=("Arial", 13))
        self.canvas.create_text(cfg.screen_size + 100, 120, fill="black", text="RIGHT: " +
                                str(values[0][3]), font=("Arial", 13))
        self.canvas.create_text(cfg.screen_size + 100, 140, fill="black", text="NONE: " +
                                str(values[0][4]), font=("Arial", 13))

    def draw_update(self, values):
        '''Updates the entire GUI, and animates a frame.

           Function Parameters:
           values <float[5]> = the values to be passed into draw_sidebar().'''

        # Clear the screen.
        self.canvas.delete("all")
        self.canvas.create_text(30, 10, fill="white", text="Score: " +
                                str(self.score))

        # Draw all the pixels.
        self.draw_food()
        self.draw_snake()
        self.draw_sidebar(values)

        # Show the changes.
        self.w.update()

    # ====================================================================== #
    # ====================================================================== #
    '''These functions control the internal state of the snake and game.'''

    def place_food(self):
        '''Places the food in a random position that isn't occupied by the snake.'''

        # Keep trying to place the food until it's placed in a valid spot.
        while True:
            self.food[0] = random.randint(0, cfg.game_size - 1)
            self.food[1] = random.randint(0, cfg.game_size - 1)

            # Make sure that the food doesn't occupy the same space as the snake.
            for seg in self.snake_list:
                if seg[0] == self.food[0] and seg[1] == self.food[1]:
                    break

            # If the food isn't occupying the same space as the snake, return.
            return

    def eat_food(self):
        '''Eats the food, increases the length of the snake, and places it somewhere random.'''

        # Increase length of snake.
        self.snake_list.insert(0, [self.food[0], self.food[1]])

        # Place the food.
        self.place_food()

        # Increase the score once placed.
        self.score += 1

    def detect_dead(self, segment):
        '''Determines whether the snake is out of bounds, or crashing
           into itself. If it is, then it's dead.

           Function Parameters:
           segment <int[2]> = the segment (always the current head segment) to see if it's crashing with the body.'''

        # If the snake would be dead locally with the current head segment. If it is, then the game won't draw the snake with its head *inside* the body.
        loc_alive = True

        # If out-of-bounds.
        if segment[0] < 0 or segment[0] >= cfg.game_size or segment[1] < 0 or segment[1] >= cfg.game_size:
            loc_alive = False

        # If colliding (in the same space) with any snake segments.
        for i in range(0, len(self.snake_list)):
            # Make sure that it's not detecting the head overlapping with itself.
            if i != 0:
                if segment == self.snake_list[i]:
                    loc_alive = False

        return loc_alive

    def update(self, direction, values):
        '''Processes a frame update for the game. Moves the snake, updates score, detects death, etc.
           The way movement works is not by moving all of the segments, but rather by taking the last
           segment of the snake and placing it ahead of the current head segment, so the last segment
           becomes the new head.

           Function Parameters:
           direction <string> = the direction that the snake will move.'''

        self.reward = 0

        # The current head segment of the snake.
        first_segment = self.snake_list[0]

        # The last segment of the snake, which will become the new head.
        last_segment = self.snake_list.pop(
            len(self.snake_list) - 1)

        # The change in coordinate that the snake will move.
        x_mod = 0
        y_mod = 0

        if direction == "UP":
            y_mod = -1
        elif direction == "DOWN":
            y_mod = 1
        elif direction == "LEFT":
            x_mod = -1
        elif direction == "RIGHT":
            x_mod = 1

        # The location of the new head segment.
        new_coords = [first_segment[0] + x_mod, first_segment[1] + y_mod]

        # Check if the new head segment overlaps with with the snake's body.
        if self.alive:
            self.alive = self.detect_dead(new_coords)

        # If the snake is still alive after moving.
        if self.alive:
            # Reward the agent for moving closer towards the food; punish it for moving further away.
            if self.distance_to_food(new_coords) < self.distance_to_food(first_segment):
                self.reward = cfg.reward_closer
            elif self.distance_to_food(new_coords) > self.distance_to_food(first_segment):
                self.reward = cfg.reward_further

            # If the new head position would also overlap with the position of the food, increase the length of the snake.
            if new_coords[0] == self.food[0] and new_coords[1] == self.food[1]:
                self.eat_food()
                # Reward the agent for eating the food.
                self.reward = cfg.reward_eat
                # Place the removed tail segment back to the end, because the new head segment has instead been placed by eat_food().
                self.snake_list.insert(len(self.snake_list), last_segment)
            else:
                # Move and place the tail segment to where the new head should be, thus completing the "movement" of the snake.
                last_segment[0] = first_segment[0] + x_mod
                last_segment[1] = first_segment[1] + y_mod
                self.snake_list.insert(0, last_segment)

            # Update the GUI.
            self.draw_update(values)

        # If the new snake after movement causes death.
        else:
            # Punish the agent for dying.
            self.reward = cfg.reward_death
            self.running = False

            # Replace the tail segment to the end. This way, the game doesn't show the snake fusing into itself after it's died.
            self.snake_list.insert(len(self.snake_list), last_segment)

            # Update the GUI.
            self.draw_update(values)

        # Return the current state of the game, and the reward that the agent receives for its action.
        return np.array([self.get_state(), self.reward])

    def step(self, action, values):
        '''The agent calls this function when it is ready to advance the game.
           This function takes the raw input from the agent, and processes the
           action to make sure it cannot move backwards into itself.

           Function Parameters:
           action <string> = the action the agent wants to take.
           values <float[5]> = the prediction values for draw_sidebar().'''

        # If the agent decides to not input to the game, the snake continues moving in the same direction.
        # This way, the snake never stops moving (which is how snake usually functions).
        if action == "NONE":
            action = self.last_action

        # If the agent tries moving backwards into the snake, force it to move in the opposite direction.
        if self.last_action == "UP" and action == "DOWN":
            action = "UP"
        elif self.last_action == "DOWN" and action == "UP":
            action = "DOWN"
        elif self.last_action == "LEFT" and action == "RIGHT":
            action = "LEFT"
        elif self.last_action == "RIGHT" and action == "LEFT":
            action = "RIGHT"

        # Update the game.
        self.last_action = action
        return self.update(action, values)

    def reset(self):
        '''Start the game, resetting all values. It also contains the game mainloop, which
           calls the agent to make a decision then learn from that decision.'''

        self.running = True
        self.alive = True
        self.reward = 0
        self.score = 0

        self.snake_list = []

        # Initialize the snake with length 5 near the middle of the screen.
        for x in range(5):
            self.snake_list.append([math.floor(cfg.game_size * 3/8 - x),
                                    math.floor(cfg.game_size * 1 / 2)])

        # Initialize the food.
        self.food = [math.floor(cfg.game_size * 2/3),
                     math.floor(cfg.game_size * 1/2)]

        # Randomize first food placement.
        self.place_food()

        # Make the snake move right by default.
        self.last_action = "RIGHT"

        self.draw_update([["n/a", "n/a", "n/a", "n/a", "n/a"]])
