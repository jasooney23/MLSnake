# Import packages.
import pickle
import tensorflow as tf     # Tensorflow.
import numpy as np          # I chose NumPy instead of Pandas because it uses less RAM.
from . import config as cfg

# =================================================================== #

def make_model():
    '''Creates a tf.keras.Sequential model with numerous hidden layers.'''
    q = tf.keras.Sequential()

    input_size = (cfg.game_size * cfg.game_size * cfg.stack_size,)  # The input tensor to the model.

    neuron_count = (input_size[0] + 5) / 4
    # The network. Hidden layers were determined with this source: 
    # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    q.add(tf.keras.layers.Dense(units=neuron_count, input_shape=input_size, activation="relu"))
    q.add(tf.keras.layers.Dense(cfg.num_actions)) # 5 outputs for 5 different actions.

    q.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss="mse", run_eagerly=True)

    return q

def make_data():
    # Create the replay memory if non-existent.
    train_data = {
        "update_index" : 0,     # Which index of the replay memory to update.
        "filled_memory" : 0,    # The amount of experiences in the memory.
        "reset_steps" : 0       # The amount of frames that have passed since the last target model reset.
    }

    # Stats.
    stats = {
        "score" : [0]     # Score per episode.
    }

    return train_data, stats

def make_memory():

    # Replay Memory.
    states_memory = np.ndarray((cfg.memory_size, cfg.game_size * cfg.game_size * cfg.stack_size))
    action_memory = np.ndarray((cfg.memory_size))
    reward_memory = np.ndarray((cfg.memory_size))
    transitions_memory = np.ndarray((cfg.memory_size, cfg.game_size * cfg.game_size * cfg.stack_size))

    return states_memory, action_memory, reward_memory, transitions_memory

# =================================================================== #

def load_models():
    try:
        # Try to load a saved model.
        q = tf.keras.models.load_model(cfg.save_path + "/model")
        target_q = tf.keras.models.load_model(cfg.save_path + "/target_model")
    except Exception as e:
        print(e)
        q = make_model()
        target_q = make_model()

    return q, target_q

def load_data():
    try:
        # Try to load replay memory and statistics.
        with open(cfg.save_path + "/train_data.dat", "rb") as openfile:
            train_data = int(pickle.load(openfile))
        with open(cfg.save_path + "/stats.dat", "rb") as openfile:
            stats = int(pickle.load(openfile))
    except Exception as e:
        print(e)
        train_data, stats = make_data()
    
    return train_data, stats

def load_memory():
    try:
        # Replay memory isn't stored using Pickle because it uses too much RAM.
        states_memory = np.load(cfg.save_path + "/states_memory.npy")
        action_memory = np.load(cfg.save_path + "/action_memory.npy")
        reward_memory = np.load(cfg.save_path + "/reward_memory.npy")
        transitions_memory = np.load(cfg.save_path + "/transitions_memory.npy")
    except Exception as e:
        print(e)
        states_memory, action_memory, reward_memory, transitions_memory = make_memory()
    
    return states_memory, action_memory, reward_memory, transitions_memory

# =================================================================== #

def save_models(q, target_q):
    q.save(cfg.save_path + "/model1", overwrite=True, include_optimizer=True)
    target_q.save(cfg.save_path + "/target_q", overwrite=True, include_optimizer=True)

def save_data(train_data, stats):
    with open(cfg.save_path + "/train_data.dat", "wb") as openfile:
        pickle.dump(train_data, openfile)
    with open(cfg.save_path + "/stats.dat", "wb") as openfile:
        pickle.dump(stats, openfile)

def save_memory(states_memory, action_memory, reward_memory, transitions_memory):
    np.save(cfg.save_path + "/states_memory", states_memory)
    np.save(cfg.save_path + "/action_memory", action_memory)
    np.save(cfg.save_path + "/reward_memory", reward_memory)
    np.save(cfg.save_path + "/transitions_memory", transitions_memory)