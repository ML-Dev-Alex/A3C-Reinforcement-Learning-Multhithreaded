import pickle
import time

import numpy as np

global training_finished
training_finished = False


def learner(agent, thread, logger, env, counter_q, episode_q, recorded_episode, saver, session, total_minutes,
            training_start, async_update, time_max, save_number, discount_factor, save_path):
    counter = counter_q.get()
    counter_q.put(counter + 1)
    learner_counter_start = 0

    episode_reward = 0
    episode_start = 0
    last_step = 0
    terminal = True

    started = False

    try:
        while counter < time_max:
            batch_states = []
            batch_rewards = []
            batch_actions = []
            batch_base_values = []

            if terminal:
                terminal = False
                state = env.reset()

                # Logging
                if started:
                    episode_counter = episode_q.get()
                    counter = counter_q.get()
                    episode_q.put(episode_counter + 1)
                    counter_q.put(counter + 1)

                minutes = total_minutes + np.uint32((time.time() - training_start) / 60)
                if thread is 0:
                    episode_end = time.time()
                    episode_time = np.uint32(episode_end - episode_start)
                    sps = np.uint32((counter - last_step) / episode_time)
                    if started:
                        logger.debug(
                            "Recorded episode: {}, Reward: {}, Frames per Second: {}, Seconds elapsed: {}, "
                            "Total frames: {}, Total minutes: {}, Total Episodes: {}".format(
                                recorded_episode, np.round(episode_reward, 2), sps * 4, np.uint32(episode_time),
                                                                               counter * 4, minutes, episode_counter))

                        if recorded_episode % 10 is 0:
                            saver.save(session, save_path + 'model', global_step=recorded_episode)
                            with open('{}variables.pkl'.format(save_path), 'wb') as f:
                                pickle.dump([counter, episode_counter, recorded_episode, minutes, save_number],
                                            f)
                        recorded_episode += 1
                    elif recorded_episode is 0:
                        logger.debug("Training starting.")
                    else:
                        logger.debug("Training resuming")
                        recorded_episode += 1
                    last_step = counter

                    episode_start = time.time()

                if not started:
                    started = True
                # Logging end

                episode_reward = 0

            # take a few actions without updating the model, saving them to a batch for later
            while not terminal and len(batch_states) < async_update:
                batch_states.append(state)

                # policy is the potential value of each action at the current state
                # value is just the estimated value of the state
                policy, value = agent.get_policy_and_value(state)

                # Select an action based on the policy
                action_index = np.random.choice(agent.action_size, p=policy)

                # Act accordingly, observe states and actions, and check if the game has ended
                state, reward, terminal, _ = env.step(action_index)

                # clip the reward to make it more consistent across different games
                reward = np.clip(reward, -1, 1)

                # Logging
                counter = counter_q.get()
                counter_q.put(counter + 1)
                episode_reward += reward
                learner_counter_start += 1
                # Logging end

                # Append observations to a mini batch
                batch_rewards.append(reward)
                batch_actions.append(action_index)
                batch_base_values.append(value[0])

            # After the batch is big enough, or we reach a terminal state, we can update the model
            # Its important to set the target value of an ending state to 0
            target_value = 0
            if not terminal:
                # otherwise get the estimate value for the current state
                target_value = agent.get_value(state)[0]

            batch_target_values = []
            # We basically use the last state in this chain as the end of a causal link, and its target value gets
            # propagated back in time into the states that lead to it
            for reward in reversed(batch_rewards):
                # The target value for each state is the reward observed plus the discounted value of the final state on
                # a chain, which is used to try to estimate how much the current state impacted the value of the chain
                target_value = reward + discount_factor * target_value
                batch_target_values.append(target_value)

            # Compute the estimated advantage value of each state by subtracting the base value observed at the state,
            # from our estimate of how impactfull the state was in order to reach the end of the causal chain
            # This gives us a clearer function to optimize in order to select the most impactfull actions
            batch_advantages = np.array(batch_target_values) - np.array(batch_base_values)

            # Apply asynchronous gradient update
            agent.train(np.vstack(batch_states), batch_actions, batch_target_values, batch_advantages)

        training_finished = True

    except KeyboardInterrupt:
        training_finished = True
