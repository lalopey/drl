from drltools.utils import trainer, ddpgmulti_config, get_env_size, plot_results
from drltools.agent.agent import DDPGMultiAgent

from unityagents import UnityEnvironment


env = UnityEnvironment(file_name="unity_environments/Reacher20_mac.app", worker_id=1)
config = ddpgmulti_config
config['action_size'], config['state_size'] = get_env_size(env)

agent = DDPGMultiAgent(config)

scores = trainer(env, agent, n_episodes=2000, max_t=1000, solved_score=30)

plot_results(scores, 'DDPG Reacher 20 agents')

env.close()
