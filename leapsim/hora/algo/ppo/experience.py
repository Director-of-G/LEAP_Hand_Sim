# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import gym
import torch
from torch.utils.data import Dataset


def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(
            self,
            num_envs,
            horizon_length,
            batch_size,
            seq_length,
            minibatch_size,
            obs_dim,
            act_dim,
            # priv_dim,
            is_rnn,
            device
        ):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        # self.priv_info_dim = priv_dim

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # self.priv_dim = priv_dim
        self.is_rnn = is_rnn
        self.storage_dict = {
            'obses': torch.zeros((self.transitions_per_env, self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device),
            # 'priv_info': torch.zeros((self.transitions_per_env, self.num_envs, self.priv_dim), dtype=torch.float32, device=self.device),
            'rewards': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            'values': torch.zeros((self.transitions_per_env, self.num_envs,  1), dtype=torch.float32, device=self.device),
            'neglogpacs': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.float32, device=self.device),
            'dones': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.uint8, device=self.device),
            'actions': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device),
            'mus': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device),
            'sigmas': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device),
            'returns': torch.zeros((self.transitions_per_env, self.num_envs,  1), dtype=torch.float32, device=self.device),
        }

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size

        self.num_games_batch = self.minibatch_size // self.seq_length

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)
        return sample

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)

        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]

        return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], input_dict['mus'], \
            input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
            input_dict['obses'], \
            input_dict['dones'], \
            None
        # , input_dict['priv_info']
    
    def _get_item_rnn(self, idx):
        gstart = idx * self.num_games_batch
        gend = (idx + 1) * self.num_games_batch
        start = gstart * self.seq_length
        end = gend * self.seq_length
        self.last_range = (start, end)

        input_dict = {}
        for k, v in self.data_dict.items():
            if isinstance(v, dict):
                v_dict = {kd:vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                if v is not None:
                    input_dict[k] = v[start:end]
                else:
                    input_dict[k] = None
        
        rnn_states = self.data_dict['rnn_states']
        input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in rnn_states]

        return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], input_dict['mus'], \
            input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
            input_dict['obses'], \
            input_dict['dones'], \
            input_dict['rnn_states']

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu
        self.data_dict['sigmas'][start:end] = sigma

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index,:] = v
        else:
            self.storage_dict[name][index,:] = val

    def computer_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict['rewards'])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict['values'][t + 1]
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = self.storage_dict['rewards'][t] + gamma * next_values * next_nonterminal - self.storage_dict['values'][t]
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam
            self.storage_dict['returns'][t, :] = mb_advs[t] + self.storage_dict['values'][t]

    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        advantages = self.data_dict['returns'] - self.data_dict['values']
        self.data_dict['advantages'] = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)
        return self.data_dict
