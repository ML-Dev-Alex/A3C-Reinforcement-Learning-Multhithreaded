import logging
import os
import pickle
import queue
import sys
import threading
import time
from time import sleep

import numpy as np
import tensorflow as tf

from agent import Agent
from environment import Environment
from learner import learner

if __name__ == "__main__":
    TEST = False

    GAME = 'Pong-Atari2600'
    START_STATE = 'Start'
    TIME_MAX = 100000000
    NUM_THREADS = 8
    OPTIMIZER_INITIAL_LEARNING_RATE = 1e-4
    DISCOUNT_FACTOR = 0.99
    # States refer to a couple of preprocessed frames
    FRAME_SKIP = 4
    STATES_SIZE = 4
    ASYNC_UPDATE = 5
    SAVER_PATH = './models/{}/'.format(GAME)
    LOGGER_PATH = './logs/{}/'.format(GAME)
    SCORE_PATH = './score/{}/'.format(GAME)

    training_finished = False
    training_start = time.time()

    if not os.path.exists(SAVER_PATH):
        os.makedirs(SAVER_PATH)
        LOAD = False
        counter = 0
        episode_counter = 0
        total_minutes = 0
        save_number = 0
        recorded_episode = 0
        with open('{}variables.pkl'.format(SAVER_PATH), 'wb') as f:
            pickle.dump([counter, episode_counter, recorded_episode, total_minutes, save_number], f)
    else:
        with open('{}variables.pkl'.format(SAVER_PATH), 'rb') as f:
            counter, episode_counter, recorded_episode, total_minutes, save_number = pickle.load(f)
        LOAD = True

    if not os.path.exists(LOGGER_PATH):
        os.makedirs(LOGGER_PATH)

    # Retro saves over older files is the save directory if we stop and re-start training
    RECORDER_PATH = './recordings/{}-{}/'.format(GAME, save_number)

    if not os.path.exists('{}'.format(RECORDER_PATH)):
        os.makedirs('{}'.format(RECORDER_PATH))
        save_number += 1
        with open('{}variables.pkl'.format(SAVER_PATH), 'wb') as f:
            pickle.dump([counter, episode_counter, recorded_episode, total_minutes, save_number], f)

    processes = []
    envs = []
    for i in range(NUM_THREADS):
        if i is 0:
            env = Environment(GAME, START_STATE, FRAME_SKIP, STATES_SIZE, RECORDER_PATH, i, TEST)
        else:
            env = Environment(GAME, START_STATE, FRAME_SKIP, STATES_SIZE, RECORDER_PATH, i, False)
        envs.append(env)

    # Allows tf to use cpu if an operation is not gpu friendly
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as session:
        agent = Agent(session=session, action_size=env.action_size, width=env.width,
                      height=env.height, states_size=env.states_size,
                      optimizer=tf.train.AdamOptimizer(OPTIMIZER_INITIAL_LEARNING_RATE))

        saver = tf.train.Saver(max_to_keep=2)

        # start queues to save information between threads
        counter_q = queue.Queue()
        episode_q = queue.Queue()

        if os.path.exists(SAVER_PATH) and LOAD:
            saver = tf.train.import_meta_graph(SAVER_PATH + 'model-{}.meta'.format(recorded_episode))
            saver.restore(session, tf.train.latest_checkpoint(SAVER_PATH))
        else:
            session.run(tf.global_variables_initializer())

        counter_q.put(counter)
        episode_q.put(episode_counter)

        logging.basicConfig(filename=LOGGER_PATH + 'episodes.log', level=logging.DEBUG)
        logger = logging.getLogger("THREAD 0")
        logger.addHandler(logging.StreamHandler(sys.stdout))

        for i in range(NUM_THREADS):
            processes.append(
                threading.Thread(target=learner,
                                 args=(
                                     agent, i, logger, envs[i], counter_q, episode_q, recorded_episode, saver, session,
                                     total_minutes, training_start, ASYNC_UPDATE, TIME_MAX,
                                     save_number, DISCOUNT_FACTOR, SAVER_PATH)))

        for process in processes:
            # Make all processes daemonic to exit them all if one does
            process.daemon = True
            process.start()

        while not training_finished:
            sleep(0.01)

        # Cleanly exit after all processes are terminated
        for process in processes:
            process.join()

