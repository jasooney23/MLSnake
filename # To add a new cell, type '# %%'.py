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
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
# tf.debugging.set_log_device_placement(True)

save_path = "./save"
model_path = "./model"
print(physical_devices)


# %%
stack_size = 1
epsilon = 0.1
discount = 0.99
learning_rate = 0.0001
memory_size = 10000
batch_size = 32


# %%
def make_model():
    # Q-Network
    q1 = tf.keras.Sequential()

    input_size = (15, 15, stack_size, 3)

    q1.add(tf.keras.layers.Conv3D(15, (5, 5, 1),
                                  activation="relu", input_shape=input_size))
    q1.add(tf.keras.layers.Conv3D(11, (3, 3, 1),
                                  activation="relu"))
    q1.add(tf.keras.layers.Conv3D(9, (3, 3, 1),
                                  activation="relu"))

    q1.add(tf.keras.layers.Flatten())

    q1.add(tf.keras.layers.Dense(27, activation="relu"))
    q1.add(tf.keras.layers.Dense(5))

    q1.compile(optimizer=tf.keras.optimizers.SGD(
        learning_rate=learning_rate), loss="mse")
    return q1


# %%
try:
    q1 = tf.keras.models.load_model(save_path + "/model1")
    q2 = tf.keras.models.load_model(save_path + "/model2")

    print("Loaded models")
except Exception as e:
    print(e)
    q1 = make_model()
    q2 = make_model()
    print("Created models")

q1.summary()
q2.summary()


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

    with open(save_path + "/scores.dat", "rb") as openfile:
        scores = pickle.load(openfile)
    with open(save_path + "/optimal_value.dat", "rb") as openfile:
        optimal_value = pickle.load(openfile)
    print("Loaded replay memory")
except Exception as e:
    print(e)
    update_index = 0
    filled_memory = 0
    # Replay Memory
    states_memory = np.ndarray((memory_size, 15, 15, stack_size, 3))
    action_memory = np.ndarray((memory_size))
    reward_memory = np.ndarray((memory_size))
    transitions_memory = np.ndarray((memory_size, 15, 15, stack_size, 3))
    scores = []
    optimal_value = []
    print("Created replay memory")


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
    directions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]

    def __init__(self, game):
        self.game = game
        self.phi = queue.deque()

        self.max_q = 0
        self.min_q = 0

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

        fstack = np.stack(frames, axis=2)
        # fstack = frames[0]
        # for x in range(1, len(frames)):
        #     frame = np.expand_dims(frames[x], axis=2)
        #     print(fstack.shape)
        #     print(frame.shape)
        #     # frame = frames[x]
        #     fstack = np.stack((fstack, frame), axis=2)

        return fstack

    def epsilon_action(self):
        # print(len(self.phi))
        stack = np.expand_dims(self.stack(self.phi), axis=0)

        if random.uniform(0, 1) <= epsilon or len(self.phi) < stack_size:
            action = self.directions[random.randint(0, 4)]

            if random.uniform(0, 1) <= 0.5:
                self.q_selector = 1
            else:
                self.q_selector = 2

            # print("random")
        else:
            if random.uniform(0, 1) <= 0.5:
                self.q_selector = 1
                prediction = q1.predict(stack)
            else:
                self.q_selector = 2
                prediction = q2.predict(stack)

            maxq = np.argmax(prediction)
            action = self.directions[maxq]
            # print(prediction)
            # print(action)
            # print(prediction[0][maxq])
        return action

    def step(self):
        if len(self.phi) == 0:
            for x in range(stack_size):
                self.phi.append(self.game.get_state())

        action = self.epsilon_action()
        state_reward = self.game.step(action)

        phi_last = list(self.phi)
        self.phi.append(state_reward[0])

        if len(self.phi) > stack_size:
            phi_last = self.stack(phi_last)
            self.phi.popleft()
            phi_current = self.stack(self.phi)

            # if state_reward[1] != 0 or random.randint(0, 4) != 0:
            self.update_memory(phi_last, self.directions.index(
                action), state_reward[1], phi_current)

    def get_batch_indices(self, memory):
        indices = []
        for x in range(batch_size):
            indices.append(random.randint(0, filled_memory - 1))

        return indices

    def losses(self):
        yj_tensor = np.ndarray((batch_size))

        indices = self.get_batch_indices(states_memory)
        states = states_memory[indices]
        action = action_memory[indices]
        reward = reward_memory[indices]
        transitions = transitions_memory[indices]

        if self.q_selector == 1:
            q_phi_next = q2.predict(transitions)
        else:
            q_phi_next = q1.predict(transitions)

        for t in range(batch_size):
            if reward[t] < 0:
                yj = reward[t]
            else:
                yj = reward[t] + (discount * np.amax(q_phi_next[t]))
            yj_tensor[t] = yj

            if reward[t] == 1 or reward[t] == -1:
                print_data = True
            else:
                print_data = False

        return states, yj_tensor, print_data

    def learn(self):
        if filled_memory == memory_size:
            state_data, expected_data, print_data = self.losses()

            # before1 = q1.predict(np.expand_dims(self.stack(self.phi), axis=0))
            # before2 = q1.predict(np.expand_dims(self.stack(self.phi), axis=0))
            # before1 = q1.get_weights()[0][0]
            # before2 = q2.get_weights()[0][0]

            if self.q_selector == 1:
                q1.train_on_batch(state_data, expected_data)
            else:
                q2.train_on_batch(state_data, expected_data)

            # gradient = tape.gradient()
            # optimizer.apply_gradients(zip(self.grad(), model.trainable_variables))
            # gradient = optimizer.get_gradients(losses, q.trainable_variables)
            # optimizer.apply_gradients(gradient)

            after1 = q1.predict(np.expand_dims(self.stack(self.phi), axis=0))
            # after2 = q1.predict(np.expand_dims(self.stack(self.phi), axis=0))
            # # # after1 = q1.get_weights()[0][0]
            # # # after2 = q2.get_weights()[0][0]

            if math.isnan(after1[0][0]):
                print("NaN")

            # if print_data:
            #     print("Before 1: " + str(before1))
            #     print("After 1:  " + str(after1) + "\n--------------------------------")
            #     print("Before 2: " + str(before2))
            #     print("After 2:  " + str(after2) + "\n================================")

            if np.amax(after1) > self.max_q:
                self.max_q = np.amax(after1)
            if np.amin(after1) > self.min_q:
                self.min_q = np.amin(after1)


# %%
def plot():
    # print("==============================")
    # print("AVERAGE LOSS v. FRAME")
    # plt.plot(average_loss)
    # plt.xlabel("Frame")
    # plt.ylabel("Average Loss")
    # plt.show()
    # print("==============================")
    print("SCORE v. EPISODE")
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    print("Q RANGE v. EPISODE")
    plt.plot(optimal_value)
    plt.xlabel("EPISODE")
    plt.ylabel("Range of Q values")
    plt.show()
    print("==============================")


# %%
game = snake.game()
x = 0

while True:
    # print("Training agent on episode " + str(x))
    dqn = agent(game)

    game.start(dqn)

    if x % 1000 == 0:
        q1.save(save_path + "/model1", overwrite=True, include_optimizer=True)
        q2.save(save_path + "/model2", overwrite=True, include_optimizer=True)
        q1.save(model_path + "1", overwrite=True, include_optimizer=True)
        q2.save(model_path + "2", overwrite=True, include_optimizer=True)
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

        with open(save_path + "/scores.dat", "wb") as openfile:
            pickle.dump(scores, openfile)
        with open(save_path + "/optimal_value.dat", "wb") as openfile:
            pickle.dump(optimal_value, openfile)

        plot()
    x += 1

    # print("Finished episode " + str(x) + ", agent scored " + str(game.score) + " points.")
    scores.append(game.score)
    print(dqn.max_q)
    print(dqn.min_q)
    optimal_value.append(dqn.max_q - dqn.min_q)


# %%
