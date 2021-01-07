import random
import queue
import time
import numpy as np
import tkinter as tk


class game:
    '''
    Creates an instance of a game. Will not advance on its own,
        only advances one frame every time step() is called.
        Do not call any of the methods here from a thread that
        isn't the main thread, except step().
    '''

    def __init__(self):
        # ============================================== #
        self.running = True
        self.alive = True
        self.reward = 0.1
        self.score = 0

        self.snake_list = [[3, 7], [2, 7], [1, 7]]
        self.food = [11, 7]
        self.snake_size = 50

        self.last_action = "RIGHT"
        #============================================== #
        self.w = tk.Tk()
        self.w.geometry("750x750")
        self.w.resizable(0, 0)

        self.canvas = tk.Canvas(self.w, bg="#000000", width=750, height=750)
        self.canvas.pack()
        #============================================== #

    def get_state(self):
        state = np.zeros((15, 15, 3))
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

    def draw_update(self):
        '''
        Updates the entire GUI, and animates a frame.
        '''
        self.canvas.delete("all")
        self.canvas.create_text(30, 10, fill="white", text="Score: " +
                                str(self.score))
        self.draw_food()
        self.draw_snake()

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
            self.food[0] = random.randint(0, 14)
            self.food[1] = random.randint(0, 14)
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
        if segment[0] < 0 or segment[0] >= 15 or segment[1] < 0 or segment[1] >= 15:
            loc_alive = False

        for i in range(0, len(self.snake_list)):
            if i != 0:
                if segment == self.snake_list[i]:
                    loc_alive = False

        return loc_alive

    def update(self, direction):
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
            # if self.distance_to_food(new_coords) < self.distance_to_food(first_segment):
            #     self.reward = 0.2
            # elif self.distance_to_food(new_coords) > self.distance_to_food(first_segment):
            #     self.reward = -0.2

            if new_coords[0] == self.food[0] and new_coords[1] == self.food[1]:
                self.eat_food()
                self.reward = 1
                self.snake_list.insert(len(self.snake_list), last_segment)
            else:
                last_segment[0] = first_segment[0] + x_mod
                last_segment[1] = first_segment[1] + y_mod
                self.snake_list.insert(0, last_segment)

            self.draw_update()
        else:
            self.reward = -1
            self.running = False

        return np.array([self.get_state(), self.reward])

    def step(self, action):
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
        return self.update(action)

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

        self.snake_list = [[5, 7], [4, 7], [3, 7], [2, 7], [1, 7]]
        self.food = [10, 7]

        self.agent = agent
        self.last_action = "RIGHT"
        #==============================================#

        self.draw_update()

        time.sleep(0.01)

        while self.running:
            agent.step()
            agent.learn()
