# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.common.layers.recurrent import GRUWithDones


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    def __init__(self):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        rnn_config = kwargs.pop('rnn_config')
        self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
        general_info = kwargs.pop('general_info')
        # self.priv_mlp = kwargs.pop('priv_mlp_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        # self.priv_info = kwargs['priv_info']
        # self.priv_info_stage2 = kwargs['proprio_adapt']
        # if self.priv_info:
        #     mlp_input_shape += self.priv_mlp[-1]
        #     self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

        #     if self.priv_info_stage2:
        #         self.adapt_tconv = ProprioAdaptTConv()

        # rnn
        self.separate = general_info['separate']
        self.has_rnn = False
        if rnn_config is not None:
            self.has_rnn = True
            self.rnn_name = rnn_config['name']
            self.rnn_units = rnn_config['units']
            self.rnn_layers = rnn_config['layers']

            rnn_in_size = input_shape[0]
            mlp_input_shape = self.rnn_units

            self.actor_rnn = GRUWithDones(input_size=rnn_in_size, hidden_size=self.rnn_units, num_layers=self.rnn_layers)
            self.rnn_ln = rnn_config.get('layer_norm', False)
            self.layer_norm = nn.LayerNorm(self.rnn_units)

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        # mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        mu, logstd, value, states = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            "rnn_states": states
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu
    
    def get_default_rnn_state(self):
        if not self.has_rnn:
            return None
        num_layers = self.rnn_layers
        if self.rnn_name == 'identity':
            rnn_units = 1
        else:
            rnn_units = self.rnn_units
        if self.rnn_name == 'lstm':
            if self.separate:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                        torch.zeros((num_layers, self.num_seqs, rnn_units)),
                        torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                        torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                        torch.zeros((num_layers, self.num_seqs, rnn_units)))
        else:
            if self.separate:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                        torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        states = obs_dict.get('rnn_states', None)
        dones = obs_dict.get('dones', None)
        bptt_len = obs_dict.get('bptt_len', 0)

        # extrin, extrin_gt = None, None
        
        # if self.priv_info:
        #     if self.priv_info_stage2:
        #         extrin = self.adapt_tconv(obs_dict['proprio_hist'])
        #         # during supervised training, extrin has gt label
        #         extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
        #         extrin_gt = torch.tanh(extrin_gt)
        #         extrin = torch.tanh(extrin)
        #         obs = torch.cat([obs, extrin], dim=-1)
        #     else:
        #         extrin = self.env_mlp(obs_dict['priv_info'])
        #         extrin = torch.tanh(extrin)
        #         obs = torch.cat([obs, extrin], dim=-1)

        out = obs
        out = out.flatten(1)

        if self.has_rnn:
            # rnn
            seq_length = obs_dict.get('seq_length', 1)

            batch_size = out.size()[0]
            num_seqs = batch_size // seq_length
            out = out.reshape(num_seqs, seq_length, -1)

            if len(states) == 1:
                states = states[0]

            out = out.transpose(0, 1)
            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1)
                dones = dones.transpose(0, 1)

            out, states = self.actor_rnn(out, states, dones, bptt_len)
            out = out.transpose(0, 1)
            out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

            if self.rnn_ln:
                out = self.layer_norm(out)
            out = self.actor_mlp(out)
            if type(states) is not tuple:
                states = (states,)
        else:
            out = self.actor_mlp(out)

        # x = self.actor_mlp(obs)
        # value = self.value(x)
        # mu = self.mu(x)

        value = self.value(out)
        mu = self.mu(out)

        sigma = self.sigma

        return mu, mu * 0 + sigma, value, states
        # , extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)

        # mu, logstd, value, extrin, extrin_gt = rst
        mu, logstd, value, states = rst

        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'rnn_states': states,
            # 'extrin': extrin,
            # 'extrin_gt': extrin_gt,
        }
        return result
