import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ForwardInverseModel(nn.Module):
    def __init__(self, obs_shape, action_space, feature_size, hidden_size, base=None, base_kwargs=None):
        super(ForwardInverseModel, self).__init__()

        self.obs_shape = obs_shape
        self.action_space = action_space
        self.feature_space = feature_size

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNEncoding
            elif len(obs_shape) == 1:
                base = MLPEncoding
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            action_size = action_space.n
        elif action_space.__class__.__name__ == "Box":
            action_size = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            action_size = action_space.shape[0]
        else:
            raise NotImplementedError

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.forward_model = nn.Sequential(
            init_(nn.Linear(feature_size + action_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, feature_size)))

        self.inverse_model = nn.Sequential(
            init_(nn.Linear(feature_size + feature_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, action_size)))

        self.train()


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs):
        raise NotImplementedError

    def get_features(self, states):
        features = self.base(states)

        return features

    def predict_next_features(self, states, actions):
        cur_features = self.base(states)

        inputs = torch.cat([cur_features, actions], dim=1)
        pred_next_features = self.forward_model(inputs)

        return pred_next_features

    def predict_cur_actions(self, states, next_states):
        cur_features = self.base(states)
        next_features = self.base(next_states)

        inputs = torch.cat([cur_features, next_features], dim=1)
        pred_actions = self.inverse_model(inputs)

        return pred_actions


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError



    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def actForController(self, env_name, inputs, rnn_hxs, masks, brake_threshold, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        action_raw = torch.clone(action)

        if 'Carla' in env_name:

            action = action.cpu()
            action[0][0] = np.tanh(action[0][0])
            action[0][1] = sigmoid(action[0][1])
            action[0][2] = sigmoid(action[0][2])

            if action[0][2] < brake_threshold:
                action[0][2] = 0
            else:
                action[0][2] = 1
            action = action.cuda()

        dist_entropy = dist.entropy().mean()

        return value, action, action_raw, action_log_probs, rnn_hxs, dist.mean, dist.stddev

    def get_dist(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        return dist

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        # print("x", x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        # print("hidden_actor", hidden_actor)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPEncoding(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, feature_size=4):

        super(MLPEncoding, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.encoder = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, feature_size)))
        
        self.train()

    def forward(self, inputs):
        x = inputs

        if self.is_recurrent:
            raise NotImplementedError

        code = self.encoder(x)

        return code


class CNNEncoding(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, feature_size=32):
        super(CNNEncoding, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.encoder = init_(nn.Linear(hidden_size, feature_size))

        self.train()

    def forward(self, inputs):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            raise NotImplementedError

        code = self.encoder(x)

        return code
