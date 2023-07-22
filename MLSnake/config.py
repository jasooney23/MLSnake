import numpy as np
import math

# Save directory, includes model and replay memory.
save_path = "./save"
backup_path = "./backup_save"


# Hyperparameters below control the training of the agent.

stack_size = 5  # Phi; amount of frames the agent sees e.g. stack_size 4 means the agent sees the
# last 4 frames.


# The epsilon-greedy slope stops changing after this many frames.
explore_count = 100000
start_epsilon = 1           # The epsilon slope begins at this float.
end_epsilon = 0.1           # The epsilon slope stops at this float.


# Discount factor. A higher discount factor determines how much the agent
# should care about prioritizing the future vs. the present.
discount = 0.99

learning_rate = 0.0001   # AKA step size.

# The target Q-network (Q-hat) is reset to the behaviour Q-net after this
# many frames.
c = 1000


memory_size = 100000  # The size of the replay memory.
batch_size = 32     # The mini-batch size used for a gradient descent step.

# The possible actions that the agent can take each frame.
directions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
num_actions = len(directions)


# ============================================== #

# Size of one side of the game, in snake_size pixels (e.g. game_size 20 is 20x20 snake_size pixels).
# snake_size pixels refers to an in-game pixel, whose size varies based on game_size.
game_size = 15

# The size of the window, in physical pixels (e.g. screen_size 1000 is 1000x1000 physical pixels).
# Physical pixels refers to the literal pixels in a display. Add 200 to this number because of the
# sidebar.
screen_size = 720

# The reward for moving nearer to the food.
reward_closer = 0
# The punishment for moving further from the food.
reward_further = 0
# The reward for eating the food.
reward_eat = 1
# The punishment for dying.
reward_death = -1

# The size of one in-game pixel.
snake_size = screen_size / game_size

#============================================== #

autosave_period = 1000

rms_momentum = 0.95

memtype = np.int8

slowmode = False
slowdelay = 0.2


# Disable epsilon, always take the best action.
# Use to evaluate the best possible performance.
performance_mode = False

#============================================== #
