import tensorflow as tf
import numpy as np
import tkinter as tk
import snake_one as snake
import pickle
import math

#============================================#

stack_size = 4

#============================================#

tf.autograph.set_verbosity(0)
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPUs found")

save_path = "."
print(physical_devices)

q = tf.keras.models.load_model(save_path + "/model")

snake_lists = []
foods = []
snake_size = 50

for x in range(0, stack_size):
    snake_lists.append(
        [])
    foods.append([])

w = tk.Tk()
w.geometry("950x750")
w.resizable(0, 0)

canvas = tk.Canvas(w, bg="#000000", width=950, height=750)
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
        draw_pixel(seg, "#808080")


def draw_food(food):
    '''
    Draws the food.
    '''
    for f in food:
        draw_pixel(f, "#ffffff")


def draw_sidebar():
    for x in range(15, 20):
        for y in range(0, 15):
            canvas.create_rectangle(x * snake_size, y * snake_size,
                                    (x + 1)*snake_size, (y + 1)*snake_size, fill="white", outline="white")

    canvas.create_text(850, 20, fill="black", text="Frame " +
                       str(frame), font=("Arial", 25))

    canvas.create_text(850, 60, fill="black", text="UP: " +
                       str(value[0][0]), font=("Arial", 13))
    canvas.create_text(850, 80, fill="black", text="DOWN: " +
                       str(value[0][1]), font=("Arial", 13))
    canvas.create_text(850, 100, fill="black", text="LEFT: " +
                       str(value[0][2]), font=("Arial", 13))
    canvas.create_text(850, 120, fill="black", text="RIGHT: " +
                       str(value[0][3]), font=("Arial", 13))
    canvas.create_text(850, 140, fill="black", text="NONE: " +
                       str(value[0][4]), font=("Arial", 13))

    canvas.create_text(850, 220, fill="gray", text='''
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

    if coord[0] >= 0 and coord[0] <= 14 and coord[1] >= 0 and coord[1] <= 14:
        if coord in snake_lists[frame - 1]:
            snake_lists[frame - 1].remove(coord)
            foods[frame - 1].append(coord)
        elif coord in foods[frame - 1]:
            foods[frame - 1].remove(coord)
        else:
            snake_lists[frame - 1].append(coord)

    draw_update()


def get_state(snake_list, food):
    state = np.zeros((15, 15))
    for seg in snake_list:
        state[seg[1]][seg[0]] = 0.5

    state[food[1]][food[0]] = 1
    return state


def stack(frames):
    fstack = frames[0]
    for x in range(1, len(frames)):
        fstack = np.dstack((fstack, frames[x]))

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
