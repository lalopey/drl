from drltools.utils import dqn_config, trainer
from drltools.agent import DQNPriorityAgent
from unityagents import UnityEnvironment


env = UnityEnvironment(file_name="unity_environments/Banana_mac.app", worker_id=1)
config = dqn_config
agent_class = DQNPriorityAgent
n_episodes = 2000
max_t = 1000
solved_score = 13
title = 'DQN'

if __name__ == "__main__":

    trainer(env, config, agent_class, n_episodes, max_t, solved_score, title)
