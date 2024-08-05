import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random
import torch.nn.functional as F
import math
from pathlib import Path
import logging
import scipy.sparse as sp
import torch.optim as optim

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Memory(object):
    def __init__(self, memory_size, batch_size):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*samples))
        
        state_batch = torch.vstack(batch.state)     
        action_batch = torch.vstack(batch.action).view(-1)
        reward_batch = torch.vstack(batch.reward).view(-1)
        next_state_batch = torch.vstack(batch.next_state)
        done_batch = torch.vstack(batch.done).view(-1)
   
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

    def __len__(self):
        return len(self.memory)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class Actor(nn.Module):
    def __init__(self, nb_states, hidden1=256 ,hidden2=16, init_w=0.2):
        super(Actor, self).__init__()
        self.nb_states = nb_states
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()
        # self.act = nn.Tanh()
        self.norm1 = nn.BatchNorm1d(nb_states)
        self.norm2 = nn.BatchNorm1d(hidden1)
        self.norm3 = nn.BatchNorm1d(hidden2)
        # self.init_weights(init_w)


    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data.uniform_(-init_w, init_w)


    
    def forward(self, x):
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.norm3(out)
        out = self.fc3(out)
        out = self.act(out)
        return out



class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions=1, hidden1=256, hidden2=16,init_w=0.2):
        super(Critic, self).__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2+nb_actions, 1) 
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(nb_states)
        self.norm2 = nn.BatchNorm1d(hidden1)
        self.norm3 = nn.BatchNorm1d(hidden2+nb_actions)
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data.uniform_(-init_w, init_w)

    
    def forward(self,x,a):
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.relu(out)  
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.relu(out) 
        out = self.fc3(self.norm3(torch.cat([out,a],dim=1)))

        return out


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class RandomProcess(object):
    def reset_states(self):
        pass

# class OUNoise(object):
#     def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()

#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x

#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class OUNoise(object):
    def __init__(self, gamma, decay_rate=0.75, init_noise=0.20):
        self.gamma = 1.0 / gamma
        self.decay_rate = decay_rate
        self.init_noise = init_noise
        self.rng = np.random.default_rng()
        
    def step(self,num_iter):
        # var = self.init_noise * (1 + self.gamma * num_iter) ** (-self.decay_rate) 
        # noise = var * self.rng.normal(0,var)

        bound = self.init_noise * (1 + self.gamma * num_iter) ** (-self.decay_rate) 
        noise = self.rng.uniform(0,bound)

        return noise

class DDPG(object):
    def __init__(self,memory_size=2000,dataset='cora',nb_states=None,gamma=100,model_path='cora_HyperU_RL',gpu=0):

        self.replay_memory_size = memory_size
        self.discount_factor = 0.95
        self.batch_size = 64
        self.dataset = dataset
        self.model_path = model_path
        self.nb_states = nb_states

        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

        self.memory = Memory(self.replay_memory_size, self.batch_size)

        self.actor = Actor(self.nb_states)
        self.actor_target = Actor(self.nb_states)
        self.actor = self.actor.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.actor_optim  = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(self.nb_states)
        self.critic_target = Critic(self.nb_states)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


        self.tau = 0.001
        self.noise = OUNoise(gamma=gamma)

        # for i in range(1000):
        #     logging.info(f'noise {self.noise.step(i)}')


    def feed_memory(self, state, action, reward, next_state, done):
        self.memory.save(state, action, reward, next_state, done)


    def feed(self, ts):
        for i in (zip(*ts)):
            (state, action, reward, next_state, done) = tuple(i)
            self.feed_memory(state, action, reward, next_state, done)
       

    def select_action(self,state,is_training=False,iter_cnt=0):
        if len(state.shape) <= 1:
            state = state.unsqueeze(dim=0)
        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state)

        if is_training:
            # logging.info(f'training {is_training} before action {action}')
            noise = self.noise.step(iter_cnt)
            # noise = torch.from_numpy(noise).to(self.device)
            action += noise
            action = torch.clamp(action, 1e-2,1.0)
            # logging.info(f'training {is_training} after action {action}')
        self.actor.train()

        action = torch.clamp(action, 1e-2,1.0)
        # action = torch.clamp(action+0.1, 0.3,1.0)
        # self.critic.eval()
        # a = self.critic(state,action)
        # logging.info(f'critic {a}')

        return action

    
    def learn(self, env, node_index, iter_cnt):
        state = env.reset(node_index)

        for j in range(env.candidate_adj_num):
            action = self.select_action(state,True,iter_cnt)
            action = action.squeeze()
            next_state, reward, done = env.step(state, action,(node_index,j))      
            self.feed((state, action, reward, next_state, done))    
            self.train()
            state = next_state


    def train(self):
        if len(self.memory) < 5 * self.batch_size:
            return False

        state, action, reward, next_state, done = self.memory.sample()

        reward = reward.to(self.device)
        reward = reward.unsqueeze(dim=1).detach()
        next_action = self.actor_target(next_state)
        next_state_action_values = self.critic_target(next_state, next_action)
        done = done.to(torch.uint8)
        done = done.to(self.device)
        done = done.unsqueeze(dim=1).detach()
        expected_values = reward + (1.0 - done) * self.discount_factor * next_state_action_values
        expected_values = expected_values.detach()

        self.critic_optim.zero_grad()
        state = state.detach()
        action = action.unsqueeze(dim=1).detach()
        state_action = self.critic(state, action)

        value_loss = F.mse_loss(state_action, expected_values)
        value_loss.backward()
        self.critic_optim.step()


        self.actor_optim.zero_grad()
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -1.0 * policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return True

    def save_model(self):
        path =Path(self.model_path)
        if not path.exists():
            path.mkdir()
        logging.info(f'save the ddpg network')
        torch.save(self.actor.state_dict(),str(path / 'actor_best_model.pth'))
        torch.save(self.critic.state_dict(),str(path / 'critic_best_model.pth'))

    def load_model(self):
        path =Path(self.model_path)
        if path.exists():
            self.actor.load_state_dict(torch.load(str(path / 'actor_best_model.pth')),strict=False)
            self.critic.load_state_dict(torch.load(str(path / 'critic_best_model.pth')),strict=False)


