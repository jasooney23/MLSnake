import tensorflow as tf
import numpy as np
import tkinter as tk
from . import Game_2 as snake
import pickle
import math

#============================================#

stack_size = 2

#============================================#

tf.autograph.set_verbosity(0)
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPUs found")

save_path = "."
print(physical_devices)

q = tf.keras.models.load_model(save_path + "/model1")

snake_lists = []
foods = []

game_size = 32
screen_size = 900
snake_size = screen_size / game_size

for x in range(0, stack_size):
    snake_lists.append(
        [])
    foods.append([])

w = tk.Tk()
w.geometry(str(screen_size + 200) + "x" + str(screen_size))
w.resizable(0, 0)

canvas = tk.Canvas(w, bg="#000000", width=screen_size +
                   200, height=screen_size)
canvas.pack(side="left")

value = [["n/a", "n/a", "n/a", "n/a", "n/a"]]


def draw_pixel(pos, fill):
    '''
    Draws a single pixel at a point, with a given colour.
    '''
    canvas.create_rectangle(pos[0] * snake_size, pos[1] * snake_size,
                            (pos[0] + 1) * snake_size, (pos[1] + 1) * snake_size, fill=fill)


def draw_snake(snake_list):
    '''
    Draws the entire snake.
    '''
    for seg in snake_list:
        draw_pixel(seg, "#ffffff")


def draw_food(food):
    '''
    Draws the food.
    '''
    for f in food:
        draw_pixel(f, "#ff0000")


def draw_sidebar():
    for x in range(0, 5):
        for y in range(0, 18):
            canvas.create_rectangle(screen_size + x * 50, y * 50,
                                    screen_size + (x + 1) * 50, (y + 1) * 50, fill="white", outline="white")

    canvas.create_text(screen_size + 100, 20, fill="black", text="Frame " +
                       str(frame), font=("Arial", 25))

    canvas.create_text(screen_size + 100, 60, fill="black", text="UP: " +
                       str(value[0][0]), font=("Arial", 13))
    canvas.create_text(screen_size + 100, 80, fill="black", text="DOWN: " +
                       str(value[0][1]), font=("Arial", 13))
    canvas.create_text(screen_size + 100, 100, fill="black", text="LEFT: " +
                       str(value[0][2]), font=("Arial", 13))
    canvas.create_text(screen_size + 100, 120, fill="black", text="RIGHT: " +
                       str(value[0][3]), font=("Arial", 13))
    canvas.create_text(screen_size + 100, 140, fill="black", text="NONE: " +
                       str(value[0][4]), font=("Arial", 13))

    canvas.create_text(screen_size + 100, 220, fill="gray", text='''
HOW TO USE:
Click on a pixel to change its state
Black = no pixel
Gray = snake segment
White = food

CONTROLS:
Arrow Keys to select frame
Enter to evaluate frames
Backspace to clear current frame
''', font=("Arial", 8))


def draw_update():
    '''
    Updates the entire GUI, and animates a frame.
    '''
    canvas.delete("all")

    if len(foods[frame - 1]) > 0:
        draw_food(foods[frame - 1])
    if len(snake_lists[frame - 1]) > 0:
        draw_snake(snake_lists[frame - 1])
    draw_sidebar()

    w.update()


frame = 1


def frame_left(event):
    global frame
    if frame > 1:
        frame -= 1
        draw_update()


def frame_right(event):
    global frame
    if frame < stack_size:
        frame += 1
        draw_update()


def change_pixel(event):
    global frame
    coord = [math.floor(event.x / snake_size),
             math.floor(event.y / snake_size)]

    if coord[0] >= 0 and coord[0] <= game_size - 1 and coord[1] >= 0 and coord[1] <= game_size - 1:
        if coord in snake_lists[frame - 1]:
            snake_lists[frame - 1].remove(coord)
            foods[frame - 1].append(coord)
        elif coord in foods[frame - 1]:
            foods[frame - 1].remove(coord)
        else:
            snake_lists[frame - 1].append(coord)

    draw_update()


def get_state(snake_list, food):
    state = np.zeros((game_size, game_size, 3))
    for seg in snake_list:
        state[seg[1]][seg[0]] = [1, 1, 1]

    state[food[1]][food[0]] = [1, 0, 0]
    return state


def stack(frames):
    fstack = np.stack(frames, axis=2)

    return fstack


def update_value(event):
    global value
    states = []
    for x in range(0, stack_size):
        states.append(
            get_state(snake_lists[x], foods[x][0]))
    value = q.predict(np.expand_dims(stack(states), axis=0))

    draw_update()


def clear_frame(event):
    snake_lists[frame - 1] = []
    foods[frame - 1] = []
    draw_update()


draw_update()
canvas.bind("<Left>", frame_left)
canvas.bind("<Right>", frame_right)
canvas.bind("<Button 1>", change_pixel)
canvas.bind("<Return>", update_value)
canvas.bind("<BackSpace>", clear_frame)
canvas.focus_set()

while True:
    w.update()
