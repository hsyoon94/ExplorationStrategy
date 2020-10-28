import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import sys
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, favor_zero_expert_reward):
        super(Discriminator, self).__init__()

        self.device = device
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)

        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0

        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)

            # For acktr
            # expert_state = torch.squeeze(expert_state, 1)
            # expert_action = torch.squeeze(expert_action, 1)

            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def update_with_parse(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0

        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))

            index_start = 2
            index_end = 4

            expert_state, expert_action = expert_batch

            expert_state_tmp = np.array([])
            policy_state_tmp = np.array([])

            for i in range(expert_state.shape[0]):
                expert_state_tmp = np.append(expert_state_tmp, expert_state[i][index_start:index_end].cpu())
                policy_state_tmp = np.append(policy_state_tmp, policy_state[i][index_start:index_end].cpu())

            expert_state_tmp = np.reshape(expert_state_tmp, [int(expert_state_tmp.shape[0]/(index_end - index_start)), int(index_end - index_start)])
            policy_state_tmp = np.reshape(policy_state_tmp, [int(policy_state_tmp.shape[0]/(index_end - index_start)), int(index_end - index_start)])

            expert_state = torch.tensor(expert_state_tmp)
            policy_state = torch.tensor(policy_state_tmp)

            # print("expert_state shape", expert_state.shape)
            # print("policy_state_shape", policy_state.shape)

            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)

            # For acktr
            # expert_state = torch.squeeze(expert_state, 1)
            # expert_action = torch.squeeze(expert_action, 1)

            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)

            # reward = s.log() - (1 - s).log()

            if self.favor_zero_expert_reward:
                reward = s.log()
            else:
                reward = - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # return reward


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, env_name, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)

        # print("all_trajectories", all_trajectories['states'])

        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1

        # For MountainCarContinuous-v0
        if 'Mountain' in env_name:
            start_idx = torch.randint(0, all_trajectories['states'].size(1) % subsample_frequency, size=(num_trajectories,)).long()
        else:
            start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]

