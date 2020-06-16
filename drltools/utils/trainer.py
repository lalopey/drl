import numpy as np
from collections import deque

import matplotlib.pyplot as plt


def trainer(env, config, agent_class, n_episodes, max_t, solved_score, title=''):
    """
    :param env: Unity environment
    :param config: Dictionary with config specifics, for examples look at drtools.utils.config.py
    :param agent_class: Agent class. For examples look at drltools/agent/agent.py
    :param n_episodes: Maximum number of episodes to train the agent
    :param max_t: Maximum number of steps per episode
    :param solved_score: Sufficient score as an average of the previous 100 episodes
    :param title: (str) Plot's title
    :return:
    """

    config['action_size'], config['state_size'] = get_env_size(env)
    agent = agent_class(config)
    scores = _trainer(env, agent, n_episodes=n_episodes, max_t=max_t, solved_score=solved_score)
    plot_results(scores, title)

    env.close()


def _trainer(env, agent, n_episodes, max_t, solved_score):
    """
    Wrapper to train unity agents with reinforcement learning algorithms
    :param env: Unity environment
    :param agent: Agent class. For examples look at drltools/agent/agent.py
    :param n_episodes: Maximum number of episodes to train the agent
    :param max_t: Maximum number of steps per episode
    :param solved_score: Sufficient score as an average of the previous 100 episodes
    """

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
        # Step until a maximum number of steps is achieved or the episode is done
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
    """
    Wrapper to output action and state size from a unity environment
    :param env: Unity environment
    :return(int,int): action and state size
    """

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    return action_size, state_size


def moving_average(data, window=100):
    """
    Calculates moving average for a given list. If the number of data points is less than window.
    calculates the average up until that point. The returned list will be the same size as the original
    :param data: (list) list of data points
    :param window: (int) moving average window
    :return: (list) list of moving averages
    """

    queue = deque(maxlen=window)
    ma = []

    for d in data:
        queue.append(d)
        ma.append(sum(queue) / len(queue))

    return ma


def plot_results(scores, title):
    """
    Wrapper to plot results of training
    :param scores: (list[float]) List of scores
    :param title: (str) Plot's title
    """

    scores_rolling = moving_average(scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(scores_rolling, "-", c="red", linewidth=3)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    ax.legend(["Score of Each episode", "100-episode Score Moving Average "])
    plt.title(title)
    plt.show()






