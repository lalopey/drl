from drltools.utils import trainer, ddpg_config, get_env_size, plot_results
from drltools.agent.agent import DDPGAgent

from unityagents import UnityEnvironment


env = UnityEnvironment(file_name="unity_environments/Reacher_mac.app", worker_id=1)
config = ddpg_config
config['action_size'], config['state_size'] = get_env_size(env)

agent = DDPGAgent(config)

scores = trainer(env, agent, n_episodes=2000, max_t=1000, solved_score=30)

plot_results(scores, 'DDPG Reacher')

env.close()
