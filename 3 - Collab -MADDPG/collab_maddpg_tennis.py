from drltools.utils import trainer, maddpg_config
from drltools.agent.agent import MaDDPGAgent
from unityagents import UnityEnvironment


env = UnityEnvironment(file_name="unity_environments/Tennis_mac.app", worker_id=1)
config = maddpg_config
agent_class = MaDDPGAgent
n_episodes = 10000
max_t = 1000
solved_score = 0.5
title = 'MADDPG Tennis'

if __name__ == "__main__":

    trainer(env, config, agent_class, n_episodes, max_t, solved_score, title)

