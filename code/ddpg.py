import torch
import torch.nn as nn
import random
from collections import deque
import copy
import numpy as np
import torch.nn.functional as F


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, max_action=1.0):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.max_action = max_action
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state * self.max_action


class ReplayBuffer:
    def __init__(self, buffer_size = 10000, batch_size=64):
        self.buffer = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
    
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def size(self):
        return len(self.buffer)
    

class RunningStatNorm(nn.Module):
    def __init__(self, shape, eps=1e-5, clip=5.0):
        super(RunningStatNorm, self).__init__()
        self.shape = shape
        self.eps = eps  
        self.clip = clip  
        # Register buffers for running stats (persisted in model state)
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))  

    def update(self, x):
        if not self.training:
            return
        x = torch.as_tensor(x, dtype=torch.float32)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        # Update running stats using Welford's algorithm
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_count / (self.count + batch_count)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        self.running_var = M2 / (self.count + batch_count)
        self.count += batch_count

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.training:
            self.update(x)
        # Normalize: (x - mean) / sqrt(var + eps)
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return torch.clamp(normalized, -self.clip, self.clip)
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0, norm_layer=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.norm_layer = norm_layer  # Shared normalization layer (or None)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        if self.norm_layer is not None:
            state = self.norm_layer(state)
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        return self.max_action * torch.tanh(self.fc3(out))



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, norm_layer=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer  # Shared normalization layer (or None)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        if self.norm_layer is not None:
            state = self.norm_layer(state)
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(torch.cat([out, action], dim=1)))
        return self.fc3(out)



class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1.0, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau= 0.001, noise_std=0.1, noise_decay=0.99, min_expl_noise=0.01, weight_decay=1e-2, normalize=False, device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.initial_std = noise_std
        self.noise_decay = noise_decay
        self.min_expl_noise = min_expl_noise
        self.step_count = 0
        self.prev_avg_reward = float('-inf')

        self.noise = OUNoise(action_dim, theta=0.15, sigma=noise_std, max_action=max_action)

        self.actor_norm = RunningStatNorm(shape=(state_dim,)).to(self.device) if normalize else None
        self.critic_norm = RunningStatNorm(shape=(state_dim,)).to(self.device) if normalize else None

        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action, self.actor_norm).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action, self.actor_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())  
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim, self.critic_norm).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, self.critic_norm).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())  
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)



    def select_action(self, state, deterministic=False):
        if isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        action = self.actor(state)

        if not deterministic:
            noise = torch.tensor(self.noise.sample(), dtype=torch.float32, device=self.device).reshape(1, -1)
            #noise = (torch.randn(self.actor.action_dim) * self.noise_std).to(self.device)
            action = (action + noise).clamp(-self.max_action, self.max_action)

        return action.detach().cpu().numpy().flatten()

    def reset_noise(self):
        self.noise.reset()
    
    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def exp_decay_noise(self, step):
        decayed_std = self.initial_std * (self.noise_decay ** step)
        self.noise_std = max(self.min_expl_noise, decayed_std)
        self.noise.sigma = max(decayed_std, self.min_expl_noise)

    def adaptive_decay_noise(self, avg_reward, threshold=3.0):
        improvement = avg_reward - self.prev_avg_reward
        if improvement < threshold:
            self.noise.sigma = max(self.noise.sigma * self.noise_decay, self.min_expl_noise)
            self.noise_std = max(self.min_expl_noise, self.noise_std*self.noise_decay)
        self.prev_avg_reward = avg_reward
        

    def update(self, replay_buffer):
        batch = replay_buffer.sample()
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.float32, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(np.array(done), dtype=torch.float32, device=self.device).unsqueeze(1)
        not_done = 1 - done


        with torch.no_grad():
            next_actions = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_actions)
            target = reward + (self.gamma * target_q * not_done)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()

        return actor_loss.item(), critic_loss.item(), target_q.mean().item(), current_q.mean().item(), target.mean().item()
    
    def eval(self):
        self.critic.eval()
        self.actor.eval()

    def train(self):
        self.critic.train()
        self.actor.train()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)