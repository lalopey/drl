from drltools.model import Actor, Critic, QNetwork
from drltools.utils.agent_utils import ReplayBuffer, OUNoise

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'

class BaseAgent:
    """Base class for agents. Interacts with the environment"""
    def __init__(self, config):
        """
        :param config: Dictionary containing configuration parameters. A sample configuration can
        be found in utils/config.py
        """
        self.num_agents = config['num_agents']
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.seed = config['seed']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.update_every_steps = config['update_every_steps']
        self.learns_per_update = config['learns_per_update']
        self.t_step = 0

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DQNAgent(BaseAgent):
    """Implementation of Deep Q-learning"""

    def __init__(self, config):
        """
        :param config: Dictionary containing configuration parameters. The current implementation requires
        the number of units in each of two layers for the q-network, fc1_units and fc2_units, as well as
        values a eps_start, eps_end, and eps_decay to define an epsilon greedy policy.  A sample configuration can
        be found in utils/config.py
        """
        super(DQNAgent, self).__init__(config)

        # Q-network parameters
        self.lr = config['lr']
        self.fc1_units = config['fc1_units']
        self.fc2_units = config['fc2_units']
        # Epsilon-greedy parameters
        self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        self.eps_decay = config['eps_decay']

        self.eps = self.eps_start
        self.current_episode = 1

        # Local Q-network
        self.qnetwork_local = QNetwork(self.state_size, 
                                       self.action_size,
                                       self.fc1_units,
                                       self.fc2_units).to(DEVICE)
        # Target Q-network
        self.qnetwork_target = QNetwork(self.state_size,
                                        self.action_size,
                                        self.fc1_units,
                                        self.fc2_units).to(DEVICE)
        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), 
                                    lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, 
                                   self.buffer_size, 
                                   self.batch_size, 
                                   self.seed)

    def step(self, state, action, reward, next_state, done, i_episode):
        """
        Receives an experience (state, action, reward, next_state, done), adds the tuple to the memory buffer and
        learns by sampling from the replay buffer
        :param i_episode: episode number, used to update epsilon greedy policy
        """

        # Check if its the first step of an episode and if so, update the epsilon of epsilon-greedy policy
        if self.current_episode != i_episode:
            self.current_episode = i_episode
            self.eps = max(self.eps_end, self.eps_decay * self.eps)

        # Add tuple to buffer
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = self.t_step + 1
        # Play from buffer learns_per_update times every update_every_steps steps.
        if (len(self.memory) > self.batch_size) and (self.t_step % self.update_every_steps == 0):

            for _ in range(self.learns_per_update):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_target_max = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        TD_update = rewards + (self.gamma * Q_target_max * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())

        # Compute loss
        loss = F.mse_loss(Q_expected, TD_update)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def report(self):
        """Save model parameters to a pth file"""
        torch.save(self.qnetwork_local.state_dict(), 'trained_agents/dqn_checkpoint.pth')


class DoubleDQNAgent(DQNAgent):
    """Implementation of Deep Q-learning"""

    def __init__(self, config):
        """
        :param config: Dictionary containing configuration parameters. The current implementation requires
        the number of units in each of two layers for the q-network, fc1_units and fc2_units, as well as
        values a eps_start, eps_end, and eps_decay to define an epsilon greedy policy.  A sample configuration can
        be found in utils/config.py
        """
        super(DoubleDQNAgent, self).__init__(config)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        states, actions, rewards, next_states, dones = experiences

        # Evaluate local network in next states
        Q_local = self.qnetwork_local(next_states)

        # Get max predicted Q values (for next states) from local model
        max_action = torch.max(Q_local, dim=1, keepdim=True)[1]

        # Evaluate target network in next states, at the maximum action obtained from
        # the local network
        Q_target = self.qnetwork_target(next_states)
        Q_target_max = Q_target.gather(1, max_action)

        TD_update = rewards + (self.gamma * Q_target_max * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions.long())


        # Compute loss
        loss = F.mse_loss(Q_expected, TD_update)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def report(self):
        """Save model parameters to a pth file"""
        torch.save(self.qnetwork_local.state_dict(), 'trained_agents/ddqn_checkpoint.pth')


class DDPGAgent(BaseAgent):
    """Implementation of Deep Deterministic Policy Gradient Agent"""
    def __init__(self, config):
        """
        :param config: Dictionary containing configuration parameters. The current implementation requires
        the number of units in each of two layers for the actor and critic networks, fc1_units and fc2_units, as well as
        other choices that can be found in utils/config.py
        """
        super(DDPGAgent, self).__init__(config)
        self.lr_actor = config['lr_actor']
        self.lr_critic = config['lr_critic']
        self.weight_decay = config['weight_decay']
        self.add_noise = config['add_noise']
        self.ou_mu = config['ou_mu']
        self.ou_theta = config['ou_theta']
        self.ou_sigma = config['ou_sigma']
        self.clip_gradient = config['clip_gradient']
        self.multi_critic = config['multi_critic']

        self.actor_fc1_units = config['actor_fc1_units']
        self.actor_fc2_units = config['actor_fc2_units']

        self.critic_fcs1_units = config['critic_fcs1_units']
        self.critic_fc2_units = config['critic_fc2_units']

        # Actor local and target networks
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.actor_fc1_units,
                                 self.actor_fc2_units).to(DEVICE)
        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.actor_fc1_units,
                                  self.actor_fc2_units).to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.lr_actor)

        # Have same initial parameters for actor and target networks
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)

        # Parameter for interacting multiple agents. If interacting, adjusts number of nodes in Critic
        # network by number of agents involved
        mulitcritic_adjust = 1
        if self.multi_critic:
            mulitcritic_adjust = self.num_agents

        # Critic local and target networks. multi_critic parameter for multiple interacting agents
        self.critic_local = Critic(self.state_size * mulitcritic_adjust,
                                   self.action_size * mulitcritic_adjust,
                                   self.critic_fcs1_units,
                                   self.critic_fc2_units,
                                   multi_critic=self.multi_critic).to(DEVICE)

        self.critic_target = Critic(self.state_size * mulitcritic_adjust,
                                    self.action_size * mulitcritic_adjust,
                                    self.critic_fcs1_units,
                                    self.critic_fc2_units,
                                    multi_critic=self.multi_critic).to(DEVICE)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.lr_critic,
                                           weight_decay=self.weight_decay)

        # Make sure the Target Network has the same weight values as the Local Network
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        # Noise process
        self.noise = OUNoise(size=self.action_size,
                             seed=self.seed,
                             mu=self.ou_mu,
                             theta=self.ou_theta,
                             sigma=self.ou_sigma)

        # Replay memory. If multiple interacting agents, this is done in a class that handles
        # all networks from multiple agents and how they sample from the buffer
        if not self.multi_critic:
            self.memory = ReplayBuffer(self.action_size,
                                       self.buffer_size,
                                       self.batch_size,
                                       self.seed)

    def step(self, states, actions, rewards, next_states, dones, i_episode):
        """
        Receives an experience (state, action, reward, next_state, done), adds the tuple to the memory buffer and
        learns by sampling from the replay buffer
        """
        # Add tuple to buffer
        self.memory.add(np.array(states).reshape(1,-1).squeeze(),
                        np.array(actions).reshape(1,-1).squeeze(),
                        rewards,
                        np.array(next_states).reshape(1,-1).squeeze(),
                        dones)

        self.t_step = self.t_step + 1
        # Play from buffer learns_per_update times every update_every_steps steps.
        if (len(self.memory) > self.batch_size) and (self.t_step % self.update_every_steps == 0):

            for _ in range(self.learns_per_update):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """Wrapper to reset noise"""
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss (using gradient clipping if input in config)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_gradient:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def report(self):
        """Save model parameters for both the actor and the critic to a pth file"""
        torch.save(self.actor_local.state_dict(), 'trained_agents/ddpg_reacher_checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'trained_agents/ddpg_reacher_checkpoint_critic.pth')


class DDPGMultiAgent(DDPGAgent):
    """Implementation of Deep Deterministic Policy Gradient Agent for multiple non-interacting agents"""

    def step(self, states, actions, rewards, next_states, dones, i_episode):
        """
        Receives an experience (state, action, reward, next_state, done), adds the tuple to the memory buffer and
        learns by sampling from the replay buffer
        """
        # Add tuples to replay buffer for all agents
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.t_step = self.t_step + 1

        if (len(self.memory) > self.batch_size) and (self.t_step % self.update_every_steps == 0):

            for _ in range(self.learns_per_update):
                experiences = self.memory.sample()
                self.learn(experiences)

    def report(self):
        """Save model parameters for both the actor and the critic to a pth file"""
        torch.save(self.actor_local.state_dict(), 'trained_agents/ddpg_reacher20_checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'trained_agents/ddpg_reacher20_checkpoint_critic.pth')


class MaDDPGAgent(BaseAgent):
    """Implementation of Deep Deterministic Policy Gradient Agent for multiple interacting agents"""
    def __init__(self, config):

        super(MaDDPGAgent, self).__init__(config)
        # Unlike in single DDPG, where we learn after a number of steps, in here we learn after update_every_episode
        # so that we get the full interaction between the agents
        self.update_every_episode = config['update_every_episode']
        # Instantiate one DDPGAgent for every interacting agent
        self.agents = [DDPGAgent(config) for i in range(self.num_agents)]
        # Add memory to replay buffer. This class manages all experience memories from all agents
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)

    def reset(self):
        """Wrapper to reset noise for all agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states):
        """Returns actions for given state as per current policy for all agents."""
        return [agent.act(state) for agent, state in zip(self.agents, states)]

    def step(self, states, actions, rewards, next_states, dones, i_episode):
        """
        Receives an experience (state, action, reward, next_state, done), adds the tuple to the memory buffer and
        learns by sampling from the replay buffer
        :param i_episode: episode number, used to keep track of when to learn
        """
        # Add tuples to replay buffer for all agents
        self.memory.add(np.array(states).reshape(1,-1).squeeze(),
                        np.array(actions).reshape(1,-1).squeeze(),
                        rewards,
                        np.array(next_states).reshape(1,-1).squeeze(),
                        dones)
        # Update after every update_every_episode episodes
        if (len(self.memory) > self.batch_size) and (i_episode% self.update_every_episode == 0):
            # Each player learns individually sampling from a common memory buffer
            for _ in range(self.learns_per_update):
                experiences = self.memory.sample()
                self.learn(experiences, player=0)
                experiences = self.memory.sample()
                self.learn(experiences, player=1)

    def cat_exps(self, x, dim):
        "Wrapper to concatenate elements of experience tuple for multiple agents"
        return torch.cat((torch.tensor(x[:, :dim]), torch.tensor(x[:, dim:])), dim=1).to(DEVICE)

    def learn(self, experiences, player):
        """
        Update policy and value parameters using given batch of experience tuples.
        Learning takes place for one player but need information on both players.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            player (int) Which player is learning from the experience
        """

        states, actions, rewards, next_states, done = experiences
        # If player 1 is learning, need to roll the DataFrame for the experience tuple
        # so that we can use the same framework for him to learn as player 0.
        # As if we were flipping the game screen depending on the player
        if player == 1:
            states = np.roll(states, self.state_size,1)
            actions = np.roll(actions, self.action_size,1)
            next_states = np.roll(next_states, self.state_size,1)
        # Split states from player learning and other player
        own_states = torch.tensor(states[:, :self.state_size]).to(DEVICE)
        other_states = torch.tensor(states[:, self.state_size:]).to(DEVICE)
        # Concatenate the state, action, and next_states for both players
        all_states = self.cat_exps(states, self.state_size)
        all_actions = self.cat_exps(actions, self.action_size)
        all_next_states = self.cat_exps(next_states, self.state_size)
        # Choose the DDPGAgent for the player learning
        agent = self.agents[player]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)),
                                     dim=1).to(DEVICE)
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - done))

        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()

        #torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim=1).to(DEVICE)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, self.tau)
        agent.soft_update(agent.actor_local, agent.actor_target, self.tau)

    def report(self):
        """Save model parameters for both the actor and the critic for each player to a pth file"""
        for i, agent in enumerate(self.agents):
            actor_local_filename = 'trained_agents/maddpg_tennis_checkpoint_actor_local_' + str(i) + '.pth'
            critic_local_filename = 'trained_agents/maddpg_tennis_checkpoint_critic_local_' + str(i) + '.pth'
            actor_target_filename = 'trained_agents/maddpg_tennis_checkpoint_actor_target_' + str(i) + '.pth'
            critic_target_filename = 'trained_agents/maddpg_tennis_checkpoint_critic_target_' + str(i) + '.pth'
            torch.save(agent.actor_local.state_dict(), actor_local_filename)
            torch.save(agent.critic_local.state_dict(), critic_local_filename)
            torch.save(agent.actor_target.state_dict(), actor_target_filename)
            torch.save(agent.critic_target.state_dict(), critic_target_filename)
