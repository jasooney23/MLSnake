'''Changes in v2:
    agent.step() and agent.learn() are now calledfrom the agent's own mainloop,
    instead of the game's mainloop. No significant feature or performance 
    changes were made, just reorganization.
    
    This version also now uses a different method of cloning the target net,
    which might work better.'''

# Import packages.
import queue
import random
import math
import tensorflow as tf

# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
from . import config as cfg
from . import Memory_1 as mem


class agent:
    '''This creates an agent that plays and learns snake. All of the functions
       used for training are contained in this class.'''

    def __init__(self, game):
        '''Sets the game window for the agent to control, and initializes the frame stack (phi).

           Function Parameters:
           game <snake.game> = the game window that the agent will use as its environment'''
        self.game = game
        self.phi = queue.deque()

        self.q, self.target_q = mem.load_models()
        self.train_data, self.stats = mem.load_data()
        self.states_memory, self.action_memory, self.reward_memory, self.transitions_memory = mem.load_memory()

        self.q.summary()
        self.target_q.summary()

    def save_all(self):
        mem.save_models(self.q, self.target_q)
        mem.save_data(self.train_data, self.stats)
        mem.save_memory(self.states_memory, self.action_memory,
                        self.reward_memory, self.transitions_memory)

    def update_memory(self, states, action, reward, transitions):
        '''Creates a new experience in the replay memory. Each experience is stored between 4
           seperate NumPy arrays, at the same index.

           Function Parameters:
           states <np.ndarray; shape=(game_size^2 * stack_size)> = the state/frame stack for the experience
           action <int> = the action taken
           reward <int> = the reward received for taking said action in state
           transitions <np.ndarray; shape=(game_size^2 * stack_size)> = the frame stack of the frame AFTER taking said action'''

        # Start replacing experiences in the memory from the beginning again.
        if self.train_data["update_index"] >= cfg.memory_size:
            self.train_data["update_index"] = 0

        # Insert the experience into replay memory.
        self.states_memory[self.train_data["update_index"]] = states
        self.action_memory[self.train_data["update_index"]] = action
        self.reward_memory[self.train_data["update_index"]] = reward
        self.transitions_memory[self.train_data["update_index"]] = transitions

        self.train_data["update_index"] += 1
        # Keep track of how much of the memory has been filled.
        if self.train_data["filled_memory"] < cfg.memory_size:
            self.train_data["filled_memory"] += 1

    def stack(self, frames):
        '''Formats a frame stack (phi) to be inputtable to the model.

           Function Parameters:
           frames <np.ndarray; shape=(game_size, game_size, stack_size)> = the frame stack to convert to a single input tensor'''

        # This code is for when I need input to an RGB convolutional model with Conv3D.
        #
        # fstack = np.stack(frames, axis=2)

        # This code is for when I need input to a black/white convolutional model with Conv2D.
        #
        # fstack = frames[0]
        # for x in range(1, len(frames)):
        #     frame = np.expand_dims(frames[x], axis=2)
        #     print(fstack.shape)
        #     print(frame.shape)
        #     # frame = frames[x]
        #     fstack = np.stack((fstack, frame), axis=2)

        # This code takes the frame stack and lays it out into a 1-D tensor.
        fstack = np.array([])
        for state in frames:  # Iterate through each frame.
            for i in state:  # Iterate through and append each row in a the current frame.
                fstack = np.append(fstack, i)

        return fstack

    def epsilon_action(self):
        '''The epsilon-greedy policy decides whether the agent will
           choose the action it thinks is best, or choose a random
           action. Currently, there is an epsilon slope, where the
           agent will take less random actions the better it performs.'''

        # Call np.expand_dims so that it can be used as input to the model.
        stack = np.expand_dims(self.stack(self.phi), axis=0)

        # Calculate the percentage of episodes played in relation to explore_count, and
        # cap it at 100% (stored as 1.0f in code).
        explore_count_percent = len(self.stats["score"]) / cfg.explore_count
        if explore_count_percent > 1:
            explore_count_percent = 1

        # Calculate what the current epsilon balue should be. The epsilon decreases from
        # start_epsilon to end_epsilon over explore_count episodes.
        current_epsilon = cfg.start_epsilon - \
            ((cfg.start_epsilon - cfg.end_epsilon) * explore_count_percent)

        # If a uniformly drawn random float is less than or equal the current epsilon,
        # take a random action.
        if random.uniform(0, 1) <= current_epsilon:
            prediction = [["n/a", "n/a", "n/a", "n/a", "n/a"]]

            action = cfg.directions[random.randint(0, 4)]
        else:
            # Select the best action, according to the model.
            prediction = self.q.predict(stack)

            maxq = np.argmax(prediction)
            action = cfg.directions[maxq]
        return action, prediction

    def step(self):
        '''Advances the game by one frame. It is called by the game, and passes an action to
           the game when it advances a frame.'''

        # Initialize phi entirely with starting frames.
        if len(self.phi) == 0:
            for _ in range(cfg.stack_size):
                self.phi.append(self.game.get_state())

        # Get action according to epsilon-greedy policy, and the Q-values
        # for all actions (debug purposes).
        action, values = self.epsilon_action()

        # Step the game forward one frame, and receive the next state and the reward.
        state_reward = self.game.step(action, values)

        phi_last = list(self.phi)
        # Update the frame stack with the latest frame.
        self.phi.append(state_reward[0])

        # Update replay memory when there are enough frames in the frame stack.
        if len(self.phi) > cfg.stack_size:
            phi_last = self.stack(phi_last)
            self.phi.popleft()
            phi_current = self.stack(self.phi)

            # Update the memory with the last state, the action taken in the last state, the reward for doing so,
            # and the resulting state.
            self.update_memory(phi_last, cfg.directions.index(
                action), state_reward[1], phi_current)

    def get_batch_indices(self):
        '''Gets a list of experiences from the replay memory. In reality,
           returns a list of indexes that are used to access the parts of  
           each experience in each array.'''

        indices = []
        for _ in range(cfg.batch_size):
            indices.append(random.randint(
                0, self.train_data["filled_memory"] - 1))

        return indices

    def learn(self):
        '''This function is also called by the game mainloop, once self.step() has been called.
           As the name suggests, it performs a gradient descent step, and also resets the
           target Q-network if needed.'''

        # Create the mini-batch of experiences.
        indices = self.get_batch_indices()
        states = self.states_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        transitions = self.transitions_memory[indices]

        # Set the target Q-network value to zero if the state is terminal.
        dones = []
        for t in range(cfg.batch_size):
            if rewards[t] < 0:
                dones.append(1.0)
            else:
                dones.append(0.0)
        dones = tf.convert_to_tensor(dones)

        targets = self.q.predict(states)
        q_phi_next = self.target_q.predict(transitions)
        yj = cfg.discount * tf.reduce_max(q_phi_next, axis=1)
        yj = rewards + yj * (1 - dones)

        for i in range(0, cfg.batch_size):
            targets[i][int(actions[i])] = yj[i]

        self.q.train_on_batch(states, targets)

        # If the learning rate is too large and causes the model to diverge to infinity, let the user know.
        if math.isnan(targets[0][0]):
            print("NaN")
        if math.isinf(targets[0][0]):
            print("inf")

        if self.train_data["reset_steps"] >= cfg.c:
            # Reset the target Q-network to the behaviour Q-network.
            self.train_data["reset_steps"] = 0
            self.target_q = tf.keras.models.clone_model(self.q)
        self.train_data["reset_steps"] += 1

    def start(self):
        self.game.reset()
        while self.game.running:
            self.step()
            self.learn()
