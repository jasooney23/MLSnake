# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import queue
import random
import threading
import math
import time
import pickle
import snake_one as snake
import tensorflow as tf
import numpy as np
import tkinter as tkinter
import matplotlib.pyplot as plt

tf.autograph.set_verbosity(0)
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.debugging.set_log_device_placement(True)

save_path = "./save"
print(physical_devices)


# %%
stack_size = 8
epsilon = 0.1
discount = 0.95
learning_rate = 0.1
memory_size = 100000
batch_size = 5000


# %%
try:
    with open(save_path + "/optimizer.dat", "rb") as openfile:
        optimizer = pickle.load(openfile)
    q = tf.keras.models.load_model(save_path + "/model")
except:
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # Q-Network
    q = tf.keras.Sequential()

    input_size = (15, 15, stack_size)

    q.add(tf.keras.layers.Conv2D(15, 5,
                                 activation="relu", input_shape=input_size))
    q.add(tf.keras.layers.Conv2D(11, 3,
                                 activation="relu"))
    q.add(tf.keras.layers.Conv2D(9, 3,
                                 activation="relu"))
    q.add(tf.keras.layers.Flatten())
    q.add(tf.keras.layers.Dense(27, activation="relu"))
    q.add(tf.keras.layers.Dense(4))
    q.compile(optimizer=optimizer, loss="mse")

q.summary()


# %%
try:
    with open(save_path + "/update_index.dat", "rb") as openfile:
        update_index = int(pickle.load(openfile))
    with open(save_path + "/filled_memory.dat", "rb") as openfile:
        filled_memory = int(pickle.load(openfile))
    with open(save_path + "/states_memory.dat", "rb") as openfile:
        states_memory = pickle.load(openfile)
    with open(save_path + "/action_memory.dat", "rb") as openfile:
        action_memory = pickle.load(openfile)
    with open(save_path + "/reward_memory.dat", "rb") as openfile:
        reward_memory = pickle.load(openfile)
    with open(save_path + "/transitions_memory.dat", "rb") as openfile:
        transitions_memory = pickle.load(openfile)
except:
    update_index = 0
    filled_memory = 0
    # Replay Memory
    states_memory = np.ndarray((memory_size, 15, 15, stack_size))
    action_memory = np.ndarray((memory_size))
    reward_memory = np.ndarray((memory_size))
    transitions_memory = np.ndarray((memory_size, 15, 15, stack_size))


# %%
# class experience:
#     states = None
#     action = None
#     reward = None
#     transitions = None

#     def __init__(self, states, action, reward, transitions):
#         self.states = states
#         self.action = action
#         self.reward = reward
#         self.transitions = transitions


# %%
class agent:
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    phi = queue.deque()

    def __init__(self, game):
        self.game = game

    def update_memory(self, states, action, reward, transitions):
        global update_index, filled_memory
        if update_index >= memory_size:
            update_index = 0

        states_memory[update_index] = states
        action_memory[update_index] = action
        reward_memory[update_index] = reward
        transitions_memory[update_index] = transitions

        update_index += 1
        if filled_memory < batch_size:
            filled_memory += 1

    def stack(self, frames):
        fstack = frames[0]
        for x in range(1, len(frames)):
            fstack = np.dstack((fstack, frames[x]))

        return fstack

    def epsilon_action(self):
        if random.uniform(0, 1) <= epsilon or len(self.phi) < stack_size:
            action = self.directions[random.randint(0, 3)]
        else:
            action = self.directions[
                np.argmax(
                    q.predict(
                        np.expand_dims(self.stack(self.phi), axis=0)))]
        return action

    def step(self):
        action = self.epsilon_action()
        state_reward = self.game.step(action)

        phi_last = list(self.phi)
        self.phi.appendleft(state_reward[0])

        if len(self.phi) > stack_size:
            phi_last = self.stack(phi_last)
            self.phi.pop()
            phi_current = self.stack(self.phi)

            self.update_memory(phi_last, self.directions.index(
                action), state_reward[1], phi_current)

    def get_batch_indices(self, memory):
        indices = []
        for x in range(batch_size):
            indices.append(random.randint(0, filled_memory - 1))

        return indices

    def losses(self):
        loss_tensor = np.ndarray((batch_size))

        indices = self.get_batch_indices(states_memory)
        states = states_memory[indices]
        action = action_memory[indices]
        reward = reward_memory[indices]
        transitions = transitions_memory[indices]

        q_phi = q.predict(states)
        q_phi_next = q.predict(transitions)

        for t in range(batch_size):
            yj = reward[t] + discount * np.amax(q_phi_next[t])
            loss_tensor[t] = math.pow(yj - q_phi[t][int(action[t])], 2)

        return loss_tensor

    def learn(self):
        if filled_memory != 0:
            losses = self.losses()

            gradient = optimizer.get_gradients(losses, ())
            optimizer.apply_gradients(gradient)


# %%
game = snake.game()

for x in range(500):
    print("Training agent on episode " + str(x))
    dqn = agent(game)

    game.start(dqn)

    if x % 10 == 0:
        q.save(save_path + "/model", overwrite=True, include_optimizer=True)
        with open(save_path + "/optimizer.dat", "wb") as openfile:
            pickle.dump(optimizer, openfile)
        with open(save_path + "/update_index.dat", "wb") as openfile:
            pickle.dump(update_index, openfile)
        with open(save_path + "/filled_memory.dat", "wb") as openfile:
            pickle.dump(filled_memory, openfile)
        with open(save_path + "/states_memory.dat", "wb") as openfile:
            pickle.dump(states_memory, openfile)
        with open(save_path + "/action_memory.dat", "wb") as openfile:
            pickle.dump(action_memory, openfile)
        with open(save_path + "/reward_memory.dat", "wb") as openfile:
            pickle.dump(reward_memory, openfile)
        with open(save_path + "/transitions_memory.dat", "wb") as openfile:
            pickle.dump(transitions_memory, openfile)

    print("Finished episode " + str(x) +
          ", agent scored " + str(game.score) + " points.")

game.w.destroy()


# %%
