from drltools.utils import trainer, ddpg_config
from drltools.agent.agent import DDPGAgent
from unityagents import UnityEnvironment


env = UnityEnvironment(file_name="unity_environments/Reacher_mac.app", worker_id=1)
config = ddpg_config
agent_class = DDPGAgent
n_episodes = 2000
max_t = 1000
solved_score = 30
title = 'DDPG Reacher'

if __name__ == "__main__":

    trainer(env, config, agent_class, n_episodes, max_t, solved_score, title)


