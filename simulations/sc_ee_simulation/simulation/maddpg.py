import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import attr
import gym



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_sigmoid=True):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.use_sigmoid = use_sigmoid
        self.action_dim = action_dim

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.action_dim == 34:
            splits = torch.split(x, (22, 8, 4), dim=1)
            a = F.gumbel_softmax(splits[0], tau=0.1, hard=False, dim=1)
            b = F.gumbel_softmax(splits[1], tau=0.1, hard=False, dim=1)
            c = F.gumbel_softmax(splits[2], tau=0.1, hard=False, dim=1)
            x = torch.concat([a, b, c], dim=1)
        else:
            x = F.softmax(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + n_agents*action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, state_dim, action_dims, n_agents, n_actions, max_size=int(1e6)):
        self.max_size = max_size
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0
        self.state_buffer = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((max_size, n_agents), dtype=np.float32)
        self.next_state_buffer = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((max_size, n_agents), dtype=np.float32)

        self.actor_state_buffer = []
        self.actor_next_state_buffer = []
        self.actor_action_buffer = []

        for i in range(n_agents):
            self.actor_state_buffer.append(
                            np.zeros((max_size, action_dims[i])))
            self.actor_next_state_buffer.append(
                            np.zeros((max_size, action_dims[i])))
            self.actor_action_buffer.append(
                            np.zeros((max_size, n_actions)))

    def add(self, raw_obs, state, action, reward, next_raw_obs, next_state, done):
        self.state_buffer[self.ptr] = state
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.done_buffer[self.ptr] = done

        for agent_idx in range(self.n_agents):
            self.actor_state_buffer[agent_idx][self.ptr] = raw_obs[agent_idx]
            self.actor_next_state_buffer[agent_idx][self.ptr] = next_raw_obs[agent_idx]
            self.actor_action_buffer[agent_idx][self.ptr] = action[agent_idx]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        actor_states = []
        actor_next_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_buffer[agent_idx][idx])
            actor_next_states.append(self.actor_next_state_buffer[agent_idx][idx])
            actions.append(self.actor_action_buffer[agent_idx][idx])
        return dict(
            actor_states=actor_states,
            state=torch.FloatTensor(self.state_buffer[idx]),
            action=actions,
            reward=torch.FloatTensor(self.reward_buffer[idx]),
            actor_next_states=actor_next_states,
            next_state=torch.FloatTensor(self.next_state_buffer[idx]),
            dones=torch.LongTensor(1 - self.done_buffer[idx]),
        )
    
    def save(self):
        # pass
        result = {}
        result["max_size"] = self.max_size
        result["ptr"] = self.ptr
        result["size"] = self.size
        result["state_buffer"] = self.state_buffer
        result["action_buffer"] = self.actor_action_buffer
        result["reward_buffer"] = self.reward_buffer
        result["next_state_buffer"] = self.next_state_buffer
        result["done_buffer"] = self.done_buffer
        result["actor_state"] = self.actor_state_buffer
        result["actor_next_state"] = self.actor_next_state_buffer
        return result
    
    def load(self, data):
        # pass
        self.max_size = data["max_size"]
        self.ptr = data["ptr"]
        self.size = data["size"]
        self.state_buffer = data["state_buffer"]
        self.actor_action_buffer = data["action_buffer"]
        self.reward_buffer = data["reward_buffer"]
        self.next_state_buffer = data["next_state_buffer"]
        self.done_buffer = data["done_buffer"]
        self.actor_state_buffer = data["actor_state"]
        self.actor_next_state_buffer = data["actor_next_state"]



@attr.s(kw_only=True)
class Hyperparameters:
    lr_a: float = attr.ib(default=0.001)
    lr_c: float = attr.ib(default=0.001)
    tau: float = attr.ib(default=0.01)
    gamma: float = attr.ib(default=0.9)
    use_sigmoid: bool = attr.ib(default=True)


@attr.s(kw_only=True)
class MaddpgHyperparameters:
    memory_size: int = attr.ib(default=100000)
    min_memory_size: int = attr.ib(default=128)
    batch_size: int = attr.ib(default=64)

@attr.s(kw_only=True)
class NoiseHyperparameters:
    use_noise: float = attr.ib(default=True)
    theta: float = attr.ib(default=0.15)
    sigma: float = attr.ib(default=0.2)
    decay: float = attr.ib(default=0.001)
    min_sigma: float = attr.ib(default=0.001)



class NoiseProcess:
    def __init__(self, action_space, theta, sigma, decay, min_sigma):
        action_shape = action_space
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = decay
        self.min_sigma = min_sigma

        self.dt = 0.5

        self.prev_x = np.zeros(action_shape)
        self.mean = np.zeros(action_shape)

    def sample(self):
        x = self.prev_x + self.theta * self.dt * (self.mean - self.prev_x) + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)

        self.prev_x = x
        return x

    def decay(self):
        self.sigma = max(self.min_sigma, self.sigma - self.sigma_decay)

    def save(self):
        result = {}
        result["theta"] = self.theta
        result["sigma"] = self.sigma
        result["sigma_decay"] = self.sigma_decay
        result["min_sigma"] = self.min_sigma
        result["prev_x"] = self.prev_x
        result["mean"] = self.mean
        result["dt"] = self.dt
        return result
    
    def load(self, data):
        self.theta = data["theta"]
        self.sigma = data["sigma"]
        self.sigma_decay = data["sigma_decay"]
        self.min_sigma = data["min_sigma"]
        self.prev_x = data["prev_x"]
        self.mean = data["mean"]
        self.dt = data["dt"]


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, hyperparameters: MaddpgHyperparameters = MaddpgHyperparameters()) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], n_actions, critic_dims, 
                            1, n_agents, agent_idx))
        self.memory = ReplayBuffer(critic_dims, actor_dims, 
                            n_agents, n_actions, max_size=hyperparameters.memory_size)
        self.hyperparameters = hyperparameters
            

    def train(self) -> (float, float):
        if self.memory.size < self.hyperparameters.min_memory_size:
            return 0.0, 0.0

        replay = self.memory.sample(self.hyperparameters.batch_size)

        states = torch.tensor(replay["state"], dtype=torch.float).to(self.device)
        actions = torch.tensor(replay["action"], dtype=torch.float).to(self.device)
        rewards = torch.tensor(replay["reward"]).to(self.device)
        states_ = torch.tensor(replay["next_state"], dtype=torch.float).to(self.device)
        dones = torch.tensor(replay["dones"]).to(self.device)
        actor_states = replay["actor_states"]
        actor_new_states = replay["actor_next_states"]

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_idx], 
                                 dtype=torch.float).to(self.device)

            new_pi = agent.actor_target(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = torch.tensor(actor_states[agent_idx], 
                                 dtype=torch.float).to(self.device)
            pi = agent.actor(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions],dim=1)
        critic_losses = []
        actor_losses = []

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.critic_target(states_, new_actions).flatten()
            test_dones = dones[:,0]
            critic_value_[test_dones] = 0.0
            critic_value = agent.critic(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.hyperparameters.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic_optimizer.step()

            actor_loss = agent.critic(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)

            agent.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor_optimizer.step()

            agent.update_target_networks()
            critic_losses.append(critic_loss.cpu().item())
            actor_losses.append(actor_loss.cpu().item())

        return np.average(actor_losses), np.average(critic_losses)
            
    def save(self, location):
        save_dict = {
            "memory": self.memory.save()
        }
        for agent_idx, agent in enumerate(self.agents):
            save_dict["agent_{}".format(agent_idx)] = agent.save()
        torch.save(save_dict, location)

    def load(self, location):
        data = torch.load(location)
        for agent_idx, agent in enumerate(self.agents):
            agent.load(data["agent_{}".format(agent_idx)])
        if data is not None:
            self.memory.load(data["memory"])

class Agent:
    def __init__(self, state_dim, action_dim, state_critic_dim, max_action, n_agents, agent_idx,
                 hyperparameters: Hyperparameters = Hyperparameters(),
                 noise_hyperparameters: NoiseHyperparameters = NoiseHyperparameters()):
        self.hyperparameters = hyperparameters
        self.noise_hyperparameters = noise_hyperparameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, self.max_action, self.hyperparameters.use_sigmoid).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action, self.hyperparameters.use_sigmoid).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hyperparameters.lr_a)

        self.critic = Critic(state_critic_dim, action_dim, n_agents).to(self.device)
        self.critic_target = Critic(state_critic_dim, action_dim, n_agents).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hyperparameters.lr_c, weight_decay=0.01)

        self.noise = None
        if self.noise_hyperparameters.use_noise:
            self.noise = NoiseProcess((action_dim,), self.noise_hyperparameters.theta, self.noise_hyperparameters.sigma,
                                      self.noise_hyperparameters.decay, self.noise_hyperparameters.min_sigma)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        selected_action = self.actor(state).cpu().data.numpy()
        if self.noise is not None:
            noise = self.noise.sample()
            selected_action = selected_action + noise

        return selected_action.clip((-self.max_action if not self.hyperparameters.use_sigmoid else 0), self.max_action).flatten()
    
    def select_unrandomized_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        selected_action = self.actor(state).cpu().data.numpy()
        return selected_action.clip((-self.max_action if not self.hyperparameters.use_sigmoid else 0), self.max_action).flatten()

    def noise_decay(self) -> None:
        if self.noise is None:
            return
        self.noise.decay()

    def update_target_networks(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.hyperparameters.tau * param.data + (1 - self.hyperparameters.tau) *
                                    target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.hyperparameters.tau * param.data + (1 - self.hyperparameters.tau) *
                                    target_param.data)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
            
    def save(self):
        return {
            "actor_model": self.actor.state_dict(),
            "critic_model": self.critic.state_dict(),
            "actor_model_target": self.actor_target.state_dict(),
            "critic_model_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "noise": self.noise.save() if self.noise is not None else None,
        }

    def load(self, data):
        if data is not None:
            self.actor.load_state_dict(data["actor_model"])
            self.actor_target.load_state_dict(data["actor_model_target"])
            self.critic.load_state_dict(data["critic_model"])
            self.critic_target.load_state_dict(data["critic_model_target"])
            self.actor_optimizer.load_state_dict(data["actor_optim"])
            self.critic_optimizer.load_state_dict(data["critic_optim"])
            if self.noise is not None and data["noise"] is not None:
                self.noise.load(data["noise"])