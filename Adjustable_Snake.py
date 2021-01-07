import random
import queue
import time
import numpy as np
import tkinter as tk
import math


class game:
    '''
    Creates an instance of a game. Will not advance on its own,
        only advances one frame every time step() is called.
        Do not call any of the methods here from a thread that
        isn't the main thread, except step().
    '''

    def __init__(self):
        # ============================================== #
        self.game_size = 32
        self.screen_size = 700
        self.snake_size = self.screen_size / self.game_size
        #============================================== #
        self.w = tk.Tk()

        self.w.geometry(str(self.screen_size + 200) +
                        "x" + str(self.screen_size))
        self.w.resizable(0, 0)

        self.canvas = tk.Canvas(self.w, bg="#000000",
                                width=self.screen_size + 200, height=self.screen_size)
        self.canvas.pack()
        #============================================== #

    def get_state(self):
        state = np.zeros((self.game_size, self.game_size, 3))
        for seg in self.snake_list:
            state[seg[1]][seg[0]] = [1, 1, 1]

        state[self.food[1]][self.food[0]] = [1, 0, 0]
        return state

    def distance_to_food(self, seg):
        diff_x = abs(seg[0] - self.food[0])
        diff_y = abs(seg[1] - self.food[1])
        return diff_x + diff_y

    # ============================================================================= #
    '''
    GUI drawing functions.
    '''

    def draw_pixel(self, pos, fill):
        '''
        Draws a single pixel at a point, with a given colour.
        '''
        self.canvas.create_rectangle(pos[0] * self.snake_size, pos[1] * self.snake_size,
                                     (pos[0] + 1) * self.snake_size, (pos[1] + 1) * self.snake_size, fill=fill)

    def draw_snake(self):
        '''
        Draws the entire snake.
        '''
        for seg in self.snake_list:
            self.draw_pixel(seg, "#ffffff")

    def draw_food(self):
        '''
        Draws the food.
        '''
        self.draw_pixel(self.food, "#ff0000")

    def draw_sidebar(self, values):
        for x in range(0, 5):
            for y in range(0, 18):
                self.canvas.create_rectangle(self.screen_size + x * 50, y * 50,
                                             self.screen_size + (x + 1) * 50, (y + 1) * 50, fill="white", outline="white")

        self.canvas.create_text(self.screen_size + 100, 60, fill="black", text="UP: " +
                                str(values[0][0]), font=("Arial", 13))
        self.canvas.create_text(self.screen_size + 100, 80, fill="black", text="DOWN: " +
                                str(values[0][1]), font=("Arial", 13))
        self.canvas.create_text(self.screen_size + 100, 100, fill="black", text="LEFT: " +
                                str(values[0][2]), font=("Arial", 13))
        self.canvas.create_text(self.screen_size + 100, 120, fill="black", text="RIGHT: " +
                                str(values[0][3]), font=("Arial", 13))
        self.canvas.create_text(self.screen_size + 100, 140, fill="black", text="NONE: " +
                                str(values[0][4]), font=("Arial", 13))

        # self.canvas.create_text(self.screen_size + 100, 180, fill="black", text="Reward: " +
        #                         str(training[0]), font=("Arial", 13))
        # self.canvas.create_text(self.screen_size + 100, 200, fill="black", text="Expected Q: " +
        #                         str(training[1]), font=("Arial", 13))

    def draw_update(self, values):
        '''
        Updates the entire GUI, and animates a frame.
        '''
        self.canvas.delete("all")
        self.canvas.create_text(30, 10, fill="white", text="Score: " +
                                str(self.score))
        self.draw_food()
        self.draw_snake()
        self.draw_sidebar(values)

        self.w.update()

    # ============================================================================= #
    '''
    These functions run the game. step() indirectly calls update(),
        which calls both eat_food() and detect_dead().
    '''

    def eat_food(self):
        '''
        Eats the food, increases length, and repositions the food.
        '''
        self.snake_list.insert(0, [self.food[0], self.food[1]])

        placed_food = False
        while not placed_food:
            self.food[0] = random.randint(0, self.game_size - 1)
            self.food[1] = random.randint(0, self.game_size - 1)
            for seg in self.snake_list:
                if not seg[0] == self.food[0] or not seg[1] == self.food[1]:
                    placed_food = True
                    break

        self.score += 1
        self.reward = 1

    def detect_dead(self, segment):
        '''
        Determines whether the snake is out of bounds, or crashing
            into itself.
        '''
        loc_alive = True
        if segment[0] < 0 or segment[0] >= self.game_size or segment[1] < 0 or segment[1] >= self.game_size:
            loc_alive = False

        for i in range(0, len(self.snake_list)):
            if i != 0:
                if segment == self.snake_list[i]:
                    loc_alive = False

        return loc_alive

    def update(self, direction, values):
        '''
        Will be called if a frame step has been queued. Moves the
            snake, and detects if the food has been eaten, or if
            the snake has collided with itself.
        '''
        self.reward = 0
        first_segment = self.snake_list[0]
        last_segment = self.snake_list.pop(
            len(self.snake_list) - 1)

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

        new_coords = [first_segment[0] + x_mod, first_segment[1] + y_mod]

        if self.alive:
            self.alive = self.detect_dead(new_coords)

        if self.alive:
            if self.distance_to_food(new_coords) < self.distance_to_food(first_segment):
                self.reward = 0.1
            elif self.distance_to_food(new_coords) > self.distance_to_food(first_segment):
                self.reward = -0.1

            if new_coords[0] == self.food[0] and new_coords[1] == self.food[1]:
                self.eat_food()
                self.reward = 1
                self.snake_list.insert(len(self.snake_list), last_segment)
            else:
                last_segment[0] = first_segment[0] + x_mod
                last_segment[1] = first_segment[1] + y_mod
                self.snake_list.insert(0, last_segment)

            self.draw_update(values)
        else:
            self.reward = 0
            self.running = False
            self.snake_list.insert(len(self.snake_list), last_segment)

            self.draw_update(values)

        return np.array([self.get_state(), self.reward])

    def step(self, action, values):
        '''
        Queues the game to step forwards a frame, with the given action.
            Can be called from a thread, and the game will not proceed
            until this is called.
        '''
        if action == "NONE":
            action = self.last_action

        if self.last_action == "UP" and action == "DOWN":
            action = "UP"
        elif self.last_action == "DOWN" and action == "UP":
            action = "DOWN"
        elif self.last_action == "LEFT" and action == "RIGHT":
            action = "LEFT"
        elif self.last_action == "RIGHT" and action == "LEFT":
            action = "RIGHT"

        self.last_action = action
        return self.update(action, values)

    # ============================================================================= #

    def start(self, agent):
        '''
        Starts the game. Initializes the GUI, the Tkinter window
            mainloop, and checks if there are any queued frame
            steps.
        '''

        # ==============================================#
        self.running = True
        self.alive = True
        self.reward = 0.1
        self.score = 0

        self.snake_list = []
        for x in range(5):
            self.snake_list.append([math.floor(self.game_size * 3/8 - x),
                                    math.floor(self.game_size * 1 / 2)])

        self.food = [math.floor(self.game_size * 2/3),
                     math.floor(self.game_size * 1/2)]

        self.agent = agent
        self.last_action = "RIGHT"
        #==============================================#

        self.draw_update([["n/a", "n/a", "n/a", "n/a", "n/a"]])

        time.sleep(0.01)

        while self.running:
            agent.step()
            agent.learn()
