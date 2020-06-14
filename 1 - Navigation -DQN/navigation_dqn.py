from drltools.utils import trainer, dqn_config, get_env_size, plot_results
from drltools.agent import DQNAgent
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="unity_environments/Banana_mac.app", worker_id=1)
config = dqn_config
config['action_size'], config['state_size'] = get_env_size(env)

agent = DQNAgent(config)

scores = trainer(env, agent, n_episodes=2000, max_t=1000, solved_score=13)

plot_results(scores, 'DQN')

env.close()
