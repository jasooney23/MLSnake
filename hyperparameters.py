# Save directory, includes model and replay memory.
save_path = "./save"


# Hyperparameters below control the training of the agent.

stack_size = 3  # Phi; amount of frames the agent sees e.g. stack_size 4 means the agent sees the
# last 4 frames.

game_size = 15  # The dimensions of the game (it is square).


# The epsilon-greedy slope stops changing after this many episodes.
explore_count = 1000
start_epsilon = 1           # The epsilon slope begins at this float.
end_epsilon = 0.1           # The epsilon slope stops at this float.


# Discount factor. A higher discount factor determines how much the agent
# should care about prioritizing the future vs. the present.
discount = 0.9

learning_rate = 0.01   # AKA step size.

# The target Q-network (Q-hat) is reset to the behaviour Q-net after this
# many frames.
c = 5000


memory_size = 50000  # The size of the replay memory.
batch_size = 32     # The mini-batch size used for a gradient descent step.
