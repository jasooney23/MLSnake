'''
Changes in v2:
- Manual model architecture, hopefully improved.
- Try Softmax?
'''

# Import packages.
import pickle
from pickletools import optimize
import tensorflow as tf     # Tensorflow.
# I chose NumPy instead of Pandas because it uses less RAM.
import numpy as np
from . import config as cfg

# =================================================================== #


def make_model(name):
    '''Creates a tf.keras.Sequential model with numerous hidden layers.'''
    q = tf.keras.Sequential(name=name)

    # The input tensor to the model.
    input_size = (cfg.game_size * cfg.game_size * cfg.stack_size,)

    # The network. Hidden layers were determined with this source:
    # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    q.add(tf.keras.layers.InputLayer(input_shape=input_size))
    q.add(tf.keras.layers.Dense(units=225, activation="relu"))
    # q.add(tf.keras.layers.Dense(units=neuron_count, activation="relu"))
    # 5 outputs for 5 different actions.
    q.add(tf.keras.layers.Dense(units=cfg.num_actions))

    # q.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate,
    #                                                 momentum=cfg.rms_momentum), loss=tf.keras.losses.Huber(), run_eagerly=True)
    q.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
                loss=tf.keras.losses.Huber(), run_eagerly=True)

    return q


def make_data():
    # Create the replay memory if non-existent.
    train_data = {
        "update_index": 0,     # Which index of the replay memory to update.
        "filled_memory": 0,    # The amount of experiences in the memory.
        # The amount of frames that have passed since the last target model reset.
        "reset_steps": 0,
        "frames_played": 0,
    }

    # Stats.
    stats = {
        "score": [0],     # Score per episode.
        "loss": [0],      # Loss per frame.
        "testing": [0]
    }

    return train_data, stats


def make_memory():

    # Replay Memory.
    states_memory = np.ndarray(
        (cfg.memory_size, cfg.game_size * cfg.game_size * cfg.stack_size), dtype=cfg.memtype)
    action_memory = np.ndarray((cfg.memory_size), dtype=cfg.memtype)
    reward_memory = np.ndarray((cfg.memory_size), dtype=cfg.memtype)
    transitions_memory = np.ndarray(
        (cfg.memory_size, cfg.game_size * cfg.game_size * cfg.stack_size), dtype=cfg.memtype)

    return states_memory, action_memory, reward_memory, transitions_memory

# =================================================================== #


def load_models():
    try:
        # Try to load a saved model.
        q = tf.keras.models.load_model(cfg.save_path + "/model")
        target_q = tf.keras.models.load_model(cfg.save_path + "/target_model")
    except Exception as e:
        print(e)
        try:
            # Try to load a saved model.
            q = tf.keras.models.load_model(cfg.backup_path + "/model")
            target_q = tf.keras.models.load_model(
                cfg.backup_path + "/target_model")
        except Exception as e:
            print(e)
            q = make_model("DQN")
            target_q = make_model("Target_DQN")

    # q.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate,
    #                                                 momentum=cfg.rms_momentum), loss=tf.keras.losses.Huber(), run_eagerly=True)

    # target_q.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate,
    #                                                        momentum=cfg.rms_momentum), loss=tf.keras.losses.Huber(), run_eagerly=True)

    return q, target_q


def load_data():
    try:
        # Try to load replay memory and statistics.
        with open(cfg.save_path + "/train_data.dat", "rb") as openfile:
            train_data = pickle.load(openfile)
        with open(cfg.save_path + "/stats.dat", "rb") as openfile:
            stats = pickle.load(openfile)
    except Exception as e:
        print(e)
        try:
            # Try to load replay memory and statistics.
            with open(cfg.backup_path + "/train_data.dat", "rb") as openfile:
                train_data = pickle.load(openfile)
            with open(cfg.backup_path + "/stats.dat", "rb") as openfile:
                stats = pickle.load(openfile)
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
        try:
            # Replay memory isn't stored using Pickle because it uses too much RAM.
            states_memory = np.load(cfg.backup_path + "/states_memory.npy")
            action_memory = np.load(cfg.backup_path + "/action_memory.npy")
            reward_memory = np.load(cfg.backup_path + "/reward_memory.npy")
            transitions_memory = np.load(
                cfg.backup_path + "/transitions_memory.npy")
        except Exception as e:
            print(e)
            states_memory, action_memory, reward_memory, transitions_memory = make_memory()

    return states_memory, action_memory, reward_memory, transitions_memory

# =================================================================== #


def save_models(path, q, target_q):
    q.save(path + "/model", overwrite=True, include_optimizer=True)
    target_q.save(path + "/target_model",
                  overwrite=True, include_optimizer=True)


def save_data(path, train_data, stats):
    with open(path + "/train_data.dat", "wb") as openfile:
        pickle.dump(train_data, openfile)
    with open(path + "/stats.dat", "wb") as openfile:
        pickle.dump(stats, openfile)


def save_memory(path, states_memory, action_memory, reward_memory, transitions_memory):
    np.save(path + "/states_memory", states_memory)
    np.save(path + "/action_memory", action_memory)
    np.save(path + "/reward_memory", reward_memory)
    np.save(path + "/transitions_memory", transitions_memory)
