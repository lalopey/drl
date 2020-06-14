from collections import deque
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def trainer(env, agent, n_episodes, max_t, solved_score):

    brain_name = env.brain_names[0]

    all_scores = []
    scores_window = deque(maxlen=100)

    all_avg_score = []

    for i_episode in range(n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        num_agents = agent.num_agents

        states = env_info.vector_observations[0:num_agents]
        scores = np.zeros(num_agents)

        for steps in range(max_t):

            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done

            agent.step(states, actions, rewards, next_states, done, i_episode)

            scores += rewards
            states = next_states
            if np.any(done):
                break

        episode_score = np.max(scores)
        all_scores.append(episode_score)
        scores_window.append(episode_score)
        avg_score = np.mean(scores_window)

        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tEpisode score (max over agents): {:.2f}'.format(i_episode, avg_score,
                                                                                                  episode_score),
            end="")
        if i_episode > 0 and i_episode % 100 == 0:
            print(
                '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            agent.report()
            all_avg_score.append(avg_score)

        if (i_episode > 99) and (avg_score >= solved_score):
            print('\n\rEnvironment solved in {} episodes with an Average Score of {:.2f}'.format(i_episode, avg_score))
            agent.report()
            return all_scores

    return all_scores


def get_env_size(env):

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    return action_size, state_size


def plot_results(scores, title):

    scores_rolling = pd.Series(scores).rolling(100).mean()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(scores_rolling, "-", c="red", linewidth=3)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    ax.legend(["Score of Each episode", "100-episode Score Moving Average "])
    plt.title(title)
    plt.show()






