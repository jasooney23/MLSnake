import tensorflow as tf             # Tensorflow.
import numpy as np
import tkinter as tk
import math
from MLSnake import Game_2 as snake
from MLSnake import Agent_5 as agent
from MLSnake import config as cfg


tf.autograph.set_verbosity(0)
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPUs found")

print(physical_devices)

q = tf.keras.models.load_model(cfg.save_path + "/DQN1")
target_q = tf.keras.models.load_model(cfg.save_path + "/DQN2")
# q.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate,
#                                                 momentum=cfg.rms_momentum), loss=tf.keras.losses.Huber(), run_eagerly=True)

# target_q.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate,
                                                        # momentum=cfg.rms_momentum), loss=tf.keras.losses.Huber(), run_eagerly=True)

snake_lists = []
foods = []

for x in range(0, cfg.stack_size + 1):
    snake_lists.append(
        [])
    foods.append([])

w = tk.Tk()
w.geometry(str(cfg.screen_size + 300) + "x" + str(cfg.screen_size))
w.resizable(0, 0)

canvas = tk.Canvas(w, bg="#000000", width=cfg.screen_size +
                   300, height=cfg.screen_size)
canvas.pack(side="left")

value = [["n/a", "n/a", "n/a", "n/a", "n/a"]]


def draw_pixel(pos, fill):
    '''
    Draws a single pixel at a point, with a given colour.
    '''
    canvas.create_rectangle(pos[0] * cfg.snake_size, pos[1] * cfg.snake_size,
                            (pos[0] + 1) * cfg.snake_size, (pos[1] + 1) * cfg.snake_size, fill=fill)


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
    for x in range(0, 7):
        for y in range(0, 18):
            canvas.create_rectangle(cfg.screen_size + x * 50, y * 50,
                                    cfg.screen_size + (x + 1) * 50, (y + 1) * 50, fill="white", outline="white")

    canvas.create_text(cfg.screen_size + 100, 20, fill="black", text="Frame " +
                       str(frame), font=("Arial", 25))

    canvas.create_text(cfg.screen_size + 100, 60, fill="black", text="UP: " +
                       str(value[0][0]), font=("Arial", 13))
    canvas.create_text(cfg.screen_size + 100, 80, fill="black", text="DOWN: " +
                       str(value[0][1]), font=("Arial", 13))
    canvas.create_text(cfg.screen_size + 100, 100, fill="black", text="LEFT: " +
                       str(value[0][2]), font=("Arial", 13))
    canvas.create_text(cfg.screen_size + 100, 120, fill="black", text="RIGHT: " +
                       str(value[0][3]), font=("Arial", 13))
    canvas.create_text(cfg.screen_size + 100, 140, fill="black", text="NONE: " +
                       str(value[0][4]), font=("Arial", 13))

    canvas.create_text(cfg.screen_size + 150, 400, fill="gray", text='''
HOW TO USE:
Click on a pixel to change its state
Black = no pixel
Gray = snake segment
White = food

CONTROLS:
Arrow Keys to select frame
Enter to evaluate frames
Numbers 1-5 to perform gradient descent on frames
(Numbers => actions)
RShift to reset model
Backspace to clear current frame

EVALUATION:
Basic evaluation stacks frames from
the first frame and ignores the last
frame
Learning evaluation uses the first
frames as the current timestep phi,
and uses a stack starting from the
second frame as the transition

IT WILL THROW AN ERROR IF THERE IS
NO FOOD OR SNAKE PIXELS
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
    if frame < cfg.stack_size + 1:
        frame += 1
        draw_update()


def change_pixel(event):
    global frame
    coord = [math.floor(event.x / cfg.snake_size),
             math.floor(event.y / cfg.snake_size)]

    if coord[0] >= 0 and coord[0] <= cfg.game_size - 1 and coord[1] >= 0 and coord[1] <= cfg.game_size - 1:
        if coord in snake_lists[frame - 1]:
            snake_lists[frame - 1].remove(coord)
            foods[frame - 1].append(coord)
        elif coord in foods[frame - 1]:
            foods[frame - 1].remove(coord)
        else:
            snake_lists[frame - 1].append(coord)

    draw_update()


def get_state(snake_list, food):
    state = np.zeros((cfg.game_size, cfg.game_size))
    for seg in snake_list:
        state[seg[1]][seg[0]] = 1

    state[food[1]][food[0]] = 2
    return state


def stack(frames):
    # fstack = np.stack(frames, axis=2)
    # return fstack

    # This code takes the frame stack and lays it out into a 1-D tensor.
    fstack = np.array([])
    for state in frames:  # Iterate through each frame.
        for i in state:  # Iterate through and append each row in a the current frame.
            fstack = np.append(fstack, i)

    return fstack

def update_value(event):
    global value
    states = []
    for x in range(0, cfg.stack_size):
        frame = get_state(snake_lists[x], foods[x][0])
        states.append(frame)

    value = q(np.expand_dims(stack(states), axis=0), training=False).numpy()
    value = q.predict(np.array([stack(states), ]))

    draw_update()


def clear_frame(event):
    snake_lists[frame - 1] = []
    foods[frame - 1] = []
    draw_update()

def learn(action):
    with tf.GradientTape() as tape:
        # Assemble phi_t and phi_t+1
        states = []
        for x in range(0, cfg.stack_size):
            frame = get_state(snake_lists[x], foods[x][0])
            states.append(frame)
        states = np.array([stack(states), ])
        transitions = []
        for x in range(1, cfg.stack_size + 1):
            frame = get_state(snake_lists[x], foods[x][0])
            transitions.append(frame)
        transitions = np.array([stack(states), ])

        action = action - 1
        # reward = 0 # nothing happens
        # done = 0 # not done

        reward = -10 # death
        done = 1 # done

        q_phi_next = target_q(transitions, training=False)
        masks = tf.one_hot(action, cfg.num_actions)

        # print(q_phi_next)
        # print(tf.reduce_max(q_phi_next, axis=1))

        # Determine y_j; the target Q value
        # y_j = reward + discount * best Q value of next state
        # target_q_values = cfg.discount * tf.reduce_max(q_phi_next, axis=1)
        # INSTEAD OF TAKING MAX VALUE, TAKE Q(s',a)
        target_q_values = cfg.discount * tf.reduce_sum(tf.multiply(q_phi_next, masks), axis=1)
        target_q_values = reward + target_q_values * (1 - done)


        q_phi = q(states, training=False)
        q_action = tf.reduce_sum(tf.multiply(q_phi, masks), axis=1)

        # CLIP TD ERROR -1 < E < 1
        td_error = target_q_values - q_action
        # td_error = tf.clip_by_value(td_error, -1, 1)

        loss = tf.math.reduce_mean(tf.math.square(td_error)) # MSE

        print(f"target_q: {target_q_values}")
        print(f"q_action: {q_action}")
        print(f"q_phi: {q_phi}")
        print(f"q_phi_next: {q_phi_next}")
        print(f"td_error:{td_error}")
        print(f"Masks: {masks}")
        print(f"Loss: {loss}")
        print()
        
    gradients = tape.gradient(loss, q.trainable_variables)
    q.optimizer.apply_gradients(
        zip(gradients, q.trainable_variables))
    
    global value
    value = q.predict(np.array([stack(states), ]))
    draw_update()

def reset_model(event):
    global q
    q = tf.keras.models.load_model(cfg.save_path + "/DQN1")
    draw_update()

def learn_1(event):
    learn(1)
def learn_2(event):
    learn(2)
def learn_3(event):
    learn(3)
def learn_4(event):
    learn(4)
def learn_5(event):
    learn(5)

draw_update()
canvas.bind("1", learn_1)
canvas.bind("2", learn_2)
canvas.bind("3", learn_3)
canvas.bind("4", learn_4)
canvas.bind("5", learn_5)

canvas.bind("<Left>", frame_left)
canvas.bind("<Right>", frame_right)
canvas.bind("<Button 1>", change_pixel)
canvas.bind("<Return>", update_value)
canvas.bind("<Shift_R>", reset_model)
canvas.bind("<BackSpace>", clear_frame)
canvas.focus_set()

while True:
    w.update()
