import h5py
import numpy as np

from a2c_ppo_acktr.model import ForwardInverseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as nrm
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, hidden_dim, lr, r_coef, favor_zero_expert_reward, device):
        """
        icml2017 - Intrinsic Curiosity Module(ICM)
        :param state_dim, action_dim, feature_dim
        :param lr: learning rate for forward and inverse model
        :param device: nvidia-gpu | cpu
        """
        super(IntrinsicCuriosityModule, self).__init__()

        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.lr = lr
        self.r_coef = r_coef

        # TODO: ADD CONDITION ON ACTION TYPE
        # self.loss = nn.CrossEntropyLoss()  # DISCRETE ACTION
        self.loss = nn.MSELoss()  # CONTINUOUS ACTION

        # output means state difference: next_state = state + output

        self.model = ForwardInverseModel(state_dim, action_dim, feature_dim, hidden_dim)

        self.model.to(device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def forward(self, input):
        raise NotImplementedError("forward func() in task transition model is not available yet")


    def predict_reward(self, cur_states, next_states, cur_actions):
        next_features = self.model.get_features(next_states)
        pred_next_features = self.model.predict_next_features(cur_states, cur_actions)

        rewards = self.r_coef * 0.5 * (next_features - pred_next_features).pow(2)
        rewards = torch.mean(torch.sum(rewards, dim=1))

        # rewards = -1/rewards
        # rewards = rewards / 10000
        # rewards = rewards.log()

        rewards = torch.sigmoid(rewards)

        if self.favor_zero_expert_reward:
            rewards = rewards.log()
        else:
            rewards = - (1 - rewards).log()

        rewards = rewards / 10

        return rewards

    # TODO: check if the loss functions do work
    def forward_loss(self, next_features, pred_next_features):
        fwd_loss = 0.5 * (next_features - pred_next_features).pow(2)
        fwd_loss = torch.mean(torch.sum(fwd_loss, dim=1))

        return fwd_loss

    # TODO: check if the loss functions do work
    def inverse_loss(self, cur_actions, pred_cur_actions, discrete=True):
        if discrete:
            # cur_actions = cur_actions.view(-1, self.action_dim)
            # one_hot_cur_actions = torch.zeros((cur_actions.shape[0], self.action_dim))
            # one_hot_cur_actions[:, cur_actions] = 1
            inv_loss = self.loss(pred_cur_actions, cur_actions)
        else:
            inv_loss = self.loss(pred_cur_actions, cur_actions)

        return inv_loss

    # TODO: change loader to replayBuffer
    def update(self, rollouts, batch_size, obsfilt=None):
        self.train()

        icm_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=batch_size)

        fwd_loss_epoch = 0
        inv_loss_epoch = 0
        n = 0

        for icm_batch in icm_data_generator:
            cur_states, cur_actions, next_states = icm_batch[0], icm_batch[2], icm_batch[-1]

            next_features = self.model.get_features(next_states)
            pred_next_features = self.model.predict_next_features(cur_states, cur_actions)
            pred_cur_actions = self.model.predict_cur_actions(cur_states, next_states)

            # print("expert_task: ", expert_task.view(1,-1))
            # print("predicted_task: ", _tasks.view(1,-1))

            fwd_loss = self.forward_loss(next_features, pred_next_features)
            inv_loss = self.inverse_loss(cur_actions, pred_cur_actions, discrete=False)

            self.optimizer.zero_grad()
            (fwd_loss + inv_loss).backward()
            self.optimizer.step()

            fwd_loss_epoch += fwd_loss.item()
            inv_loss_epoch += inv_loss.item()
            n = n + 1

        fwd_loss_epoch /= n
        inv_loss_epoch /= n
        icm_loss_epoch = fwd_loss_epoch + inv_loss_epoch

        # print(" task transition model step: {}  task_loss: {:.4f} ib_loss: {:.4f}".format(n-1, task_loss_epoch, ib_loss_epoch))

        return fwd_loss_epoch, inv_loss_epoch, icm_loss_epoch


