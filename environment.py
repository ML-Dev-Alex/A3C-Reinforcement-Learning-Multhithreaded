import time

import numpy as np
import retrowrapper


class Environment:
    def __init__(self, game, start_state, states_size, frame_skip, path, thread, test):
        self.game = game

        self.test = test

        # Thread 0 is used for logging/debugging, saving the model and rendering one env when necessary
        if thread is 0:
            self.env = retrowrapper.RetroWrapper(game=self.game, state=start_state, record=path)
        else:
            self.env = retrowrapper.RetroWrapper(game=self.game, state=start_state)

        self.frame_skip = frame_skip

        self.states_size = states_size
        self.states_counter = 0

        # Accelerate training by making the action space fit the actual actions for the game
        if game == 'SpaceInvaders-Atari2600':
            self.action_space = [1, 2, 3]
        elif game == 'Breakout-Atari2600':
            self.action_space = [1, 2, 3]
        elif game == 'Pong-Atari2600':
            self.action_space = [1, 4, 5]
        else:
            # Otherwise, use the actions specified by Open AI.
            self.action_space = range(self.env.action_space.n)

        self.action_size = len(self.action_space)

        self.observation_shape = np.array(self.env.observation_space.shape)

        # Account for the fact that we halve the size of each frame on the pre-processing step
        self.width = np.uint32(self.env.observation_space.shape[0] / 2)
        self.height = np.uint32(self.env.observation_space.shape[1] / 2)

        # Initialize vectors of the right shapes and types to speed up training
        self.action = np.zeros(self.env.action_space.n, dtype=np.uint8)
        self.states = np.zeros((1, self.width, self.height, self.states_size), dtype=np.float32)
        self.frame_temp1 = np.zeros((self.observation_shape[0], self.observation_shape[1], self.observation_shape[2]),
                                    dtype=np.int8)
        self.frame_temp2 = np.zeros((self.observation_shape[0], self.observation_shape[1], self.observation_shape[2]),
                                    dtype=np.int8)

    def step(self, action_index):
        # Account for the fact that some games have different action spaces
        action_index = self.action_space[action_index]

        # action is a one hot encoded array
        self.action = np.zeros_like(self.action)
        self.action[action_index] = 1

        reward_sum = 0
        done = False

        # Skip a few frames for better simulating human behavior and faster convergence,
        for i in range(self.frame_skip):
            if i % 2 is 0:
                self.frame_temp1[:], reward, terminal, info = self.env.step(self.action)
                # if game ends after taking only one frame, simply copy it to pass as the second frame
                if terminal and i is 0:
                    self.frame_temp2[:] = self.frame_temp1
            else:
                self.frame_temp2[:], reward, terminal, info = self.env.step(self.action)

            if self.test:
                self.render()

            reward_sum += reward
            done += terminal
            # preprocess the last couple of frames as a state and append it to a list of a causally related states
            if done:
                break

        self.states[:, :, :, self.states_counter] = self.preprocess(self.frame_temp1, self.frame_temp2)[0, :, :, 0]
        self.states_counter += 1
        if self.states_counter >= 4:
            self.states_counter = 0

        return np.copy(self.states), reward_sum, done, info

    def preprocess(self, frame1, frame2):
        # First axis is the batch axis for passing multiple frames at once to the model, second and third are the
        # reduced image dimensions, and the last axis is a temporal axis rather than rgb channels for the conv model
        preprocesed_state = np.zeros((self.width, self.height, 1), dtype=np.float32)
        # Take the max between two frames to eliminate problems with atari flickering on some atari environments,
        # Halve the size, remove color information (channels) and normalize
        preprocesed_state[:, :, 0] = np.mean(np.fmax(np.float32(frame1[::2, ::2]) * (1.0 / 255.0), np.float32(frame2[::2, ::2])
                                             * (1.0 / 255.0)), axis=2)

        return np.expand_dims(preprocesed_state[:, :, :], axis=0)

    def render(self):
        self.env.render()
        time.sleep(0.005)

    def reset(self):
        # Some environments require 'fire' (action = 1) to start
        # action is a one hot encoded array
        self.action = np.zeros_like(self.action)
        self.action[1] = 1

        self.states = np.zeros_like(self.states)

        self.env.reset()
        self.states_counter = 0

        # Creates a list of a few causally related states
        for i in range(self.frame_skip):
            if i % 2 is 0:
                self.frame_temp1[:], _, _, info = self.env.step(self.action)
            else:
                self.frame_temp2[:], _, _, info = self.env.step(self.action)

            if self.test:
                self.render()

        self.states[:, :, :, :] = np.repeat(self.preprocess(self.frame_temp1, self.frame_temp2), self.states_size,
                                            axis=3)
        return np.copy(self.states)
