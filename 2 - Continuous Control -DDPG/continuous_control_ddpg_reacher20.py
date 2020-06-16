from drltools.utils import trainer, ddpgmulti_config
from drltools.agent.agent import DDPGMultiAgent
from unityagents import UnityEnvironment


env = UnityEnvironment(file_name="unity_environments/Reacher20_mac.app", worker_id=1)
config = ddpgmulti_config
agent_class = DDPGMultiAgent
n_episodes = 2000
max_t = 1000
solved_score = 30
title = 'DDPG Reacher 20 agents'

if __name__ == "__main__":

    trainer(env, config, agent_class, n_episodes, max_t, solved_score, title)


