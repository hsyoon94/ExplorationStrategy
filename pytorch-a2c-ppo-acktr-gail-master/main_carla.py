import copy
import glob
import os
import time
from collections import deque
import json
from collections import OrderedDict
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import custom_tasks
import sys

import pandas as pd
import seaborn as sns
from scipy import stats
import scipy as sp

from datetime import datetime

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.algo import icm
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from random import *
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import matplotlib.pyplot as plot

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def controller(env_name, pltActor, plrActor, c_coef, obs_plt, obs_plr, plt_rec_hidden, plr_rec_hidden, mask_plt, mask_plr, is_greedy, decay, deterministic=False):

    brake_threshold = 1.1

    plt_value, plt_action, plt_action_raw, plt_log_prob, plt_rec_hidden, plt_mean, plt_stddev = pltActor.actForController(env_name, obs_plt, plt_rec_hidden, mask_plt, brake_threshold, deterministic)
    plr_value, plr_action, plr_action_raw, plr_log_prob, plr_rec_hidden, plr_mean, plr_stddev = plrActor.actForController(env_name, obs_plr, plr_rec_hidden, mask_plr, brake_threshold, deterministic)

    kld = torch.log(plr_stddev / plt_stddev) + ((plt_stddev ** 2) + ((plt_mean - plr_mean) ** 2)) / (2 * (plr_stddev ** 2))

    if is_greedy is False:
        action = [torch.squeeze(plt_action, 0)[i] if torch.squeeze(kld, 0)[i] >= c_coef else torch.squeeze(plr_action, 0)[i] for i in range(torch.squeeze(kld, 0).shape[0])]
        action_raw = [torch.squeeze(plt_action_raw, 0)[i] if torch.squeeze(kld, 0)[i] >= c_coef else torch.squeeze(plr_action_raw, 0)[i] for i in range(torch.squeeze(kld, 0).shape[0])]
        agent = ['plt' if torch.squeeze(kld, 0)[i] >= c_coef else 'plr' for i in range(torch.squeeze(kld, 0).shape[0])]

    elif is_greedy is True:
        epsilon = 0.8 / math.exp(decay / 50000)

        if decay % 999 is 0:
            print("epsilon :", epsilon)

        value = random()
        # epsilon => exploration
        if value <= epsilon:
            # plr
            action = [uniform(-1, 1) for i in range(torch.squeeze(kld, 0).shape[0])]

        # 1 - epsilon => exploitation
        elif value > epsilon:
            action = [torch.squeeze(plt_action, 0)[i] for i in range(torch.squeeze(kld, 0).shape[0])]

    action = torch.FloatTensor(action).reshape(1, -1)
    action_raw = torch.FloatTensor(action_raw).reshape(1, -1)

    return plt_value, plr_value, action, action_raw, plt_log_prob, plr_log_prob, plt_rec_hidden, plr_rec_hidden, agent, plt_action, plr_action, plt_action_raw, plr_action_raw


def main():

    args = get_args()

    now = datetime.now()
    now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

    SAVE_DIR = args.save_dir + now_date + '/' + args.env_name + '/plt' + str(args.plt_entropy_coef) + '_plr' + str(args.plr_entropy_coef) + '_ctrlcoef' + str(args.controller_coef) + '_extr_weight' + str(args.extr_reward_weight) + '_expert_weight' + str(args.expert_reward_weight) + '/' + now_time + '/'

    try:
        if not (os.path.isdir(SAVE_DIR)):
            os.makedirs(os.path.join(SAVE_DIR + 'figure/'))
            os.makedirs(os.path.join(SAVE_DIR + 'log/'))
    except OSError:
        pass

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

    exploitation_actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs={'recurrent': args.recurrent_policy})
    exploration_actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs={'recurrent': args.recurrent_policy})

    exploitation_actor_critic.to(device)
    exploration_actor_critic.to(device)

    if args.algo == 'a2c':
        exploration_agent = algo.A2C_ACKTR(exploration_actor_critic, args.plr_value_loss_coef, args.plr_entropy_coef, lr=args.lr, eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm)
        exploitation_agent = algo.A2C_ACKTR(exploitation_actor_critic, args.plt_value_loss_coef, args.plt_entropy_coef, lr=args.lr, eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        exploration_agent = algo.PPO(exploration_actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.plr_value_loss_coef, args.plr_entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
        exploitation_agent = algo.PPO(exploitation_actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.plt_value_loss_coef, args.plt_entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        exploration_agent = algo.A2C_ACKTR(exploration_actor_critic, args.plr_value_loss_coef, args.plr_entropy_coef, acktr=True)
        exploitation_agent = algo.A2C_ACKTR(exploitation_actor_critic, args.plt_value_loss_coef, args.plt_entropy_coef, acktr=True)

    exploration_rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, envs.action_space, exploration_actor_critic.recurrent_hidden_state_size)
    exploitation_rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, envs.action_space, exploitation_actor_critic.recurrent_hidden_state_size)

    if args.icm:
        intr = icm.IntrinsicCuriosityModule(envs.observation_space.shape, envs.action_space, args.latent_dim, 100, lr=args.lr, r_coef=args.intr_coef, favor_zero_expert_reward=args.favor_zero_expert_reward, device=device)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(envs.observation_space.shape[0] + envs.action_space.shape[0], 100, device, args.favor_zero_expert_reward)
        if args.env_name == 'CarlaUrbanIntersection-v0':
            file_name = os.path.join('/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-20200101-50.pt')

        elif args.env_name == 'CarlaUrbanIntersection8423-v0':
            file_name = os.path.join('/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-20200217-50.pt')

        elif args.env_name == 'CarlaUrbanIntersection8424-v0':
            file_name = os.path.join('/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-20200217-50.pt')

        elif args.env_name == 'CarlaUrbanIntersection8425-v0':
            file_name = os.path.join('/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-20200301_last_8-100.pt')

        elif args.env_name == 'CarlaUrbanIntersectionMini-v0':
            file_name = os.path.join('/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-data-20200220_light_15-100.pt')

        elif args.env_name == 'Hopper-v2':
            file_name = os.path.join('/home/hsyoon/job/pds_sh_model/pytorch-a2c-ppo-acktr-gail-master/gail_experts/trajs_hopper_ikostrikov.pt')

        elif args.env_name == 'HalfCheetah-v2':
            file_name = os.path.join('/home/hsyoon/job/pds_sh_model/pytorch-a2c-ppo-acktr-gail-master/gail_experts/trajs_halfcheetah_ikostrikov.pt')

        elif args.env_name == 'Walker2d-v2':
            file_name = os.path.join('/home/hsyoon/job/pds_sh_model/pytorch-a2c-ppo-acktr-gail-master/gail_experts/trajs_walker2d_ikostrikov.pt')

        elif args.env_name == 'MountainCarContinuous-v0':
            file_name = os.path.join('/home/hsyoon/job/pds_sh_model/pytorch-a2c-ppo-acktr-gail-master/gail_experts/trajs_mountainoldcarcontinuous_ppo.pt')

        gail_train_loader = torch.utils.data.DataLoader(gail.ExpertDataset(args.env_name, file_name, num_trajectories=32, subsample_frequency=4), batch_size=args.gail_batch_size, shuffle=True, drop_last=True)

    obs = envs.reset()

    exploitation_rollouts.obs[0].copy_(obs)
    exploration_rollouts.obs[0].copy_(obs)
    exploitation_rollouts.to(device)
    exploration_rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    JSON_OVERALL = OrderedDict()
    TRAINING_DICT = {
        "training_reward": None,
        "count_plt": None,
        "count_plr": None,
        "policy_dist": None
    }
    TRAINING_DICT_LIST = []
    EVALUATION_REWARDS_LIST = []
    EPISODE_DICT = {
        "entropy_plt": None,
        "entropy_plr": None,
        "value_loss_plt": None,
        "value_loss_plr": None,
        "action_loss_plt": None,
        "action_loss_plr": None
    }
    EPISODE_DICT_LIST = []
    PRINT_INDEX = 0
    PRINT_INDEX_1 = 1
    PRINT_INDEX_2 = 2

    for j in range(num_updates):

        if j > 0 and j % 2 == 0:
            JSON_OVERALL["training"] = TRAINING_DICT_LIST
            JSON_OVERALL["evaluation"] = EVALUATION_REWARDS_LIST
            JSON_OVERALL["episode"] = EPISODE_DICT_LIST

            with open(SAVE_DIR + 'log/log{}.json'.format(j), 'w', encoding="utf-8") as make_file:
                json.dump(JSON_OVERALL, make_file, ensure_ascii=False, indent="\t")

            TRAINING_DICT_LIST = []
            EVALUATION_REWARDS_LIST = []
            EPISODE_DICT_LIST = []

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(exploitation_agent.optimizer, j, num_updates, exploitation_agent.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(exploration_agent.optimizer, j, num_updates, exploration_agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # Initialize the saving variables.
        count_plt = 0
        count_plr = 0
        policy_history = np.array([])

        history_plt_mean = np.array([])
        history_plr_mean = np.array([])
        history_plt_stddev = np.array([])
        history_plr_stddev = np.array([])


        print("envs:", args.env_name, "ctrlcoef:" + str(args.controller_coef), "env_weight:", str(args.extr_reward_weight), "expert_weight:", str(args.expert_reward_weight))
        for step in range(args.num_steps):

            with torch.no_grad():
                if 'Carla' in args.env_name:
                    if j % 10 == 0 and j != 0:
                        plt_value, plr_value, action, action_raw, plt_action_log_prob, plr_action_log_prob, plt_recurrent_hidden_states, plr_recurrent_hidden_states, agent_type, plt_action, plr_action, plt_action_raw, plr_action_raw = \
                            controller(args.env_name, exploitation_actor_critic, exploration_actor_critic, args.controller_coef, exploitation_rollouts.obs[step], exploration_rollouts.obs[step],
                                       exploitation_rollouts.recurrent_hidden_states[step],
                                       exploration_rollouts.recurrent_hidden_states[step], exploitation_rollouts.masks[step], exploration_rollouts.masks[step], args.egreedy, j * args.num_steps + step, deterministic=True)
                    else:
                        plt_value, plr_value, action, action_raw, plt_action_log_prob, plr_action_log_prob, plt_recurrent_hidden_states, plr_recurrent_hidden_states, agent_type, plt_action, plr_action, plt_action_raw, plr_action_raw = \
                            controller(args.env_name, exploitation_actor_critic, exploration_actor_critic, args.controller_coef, exploitation_rollouts.obs[step], exploration_rollouts.obs[step],
                                       exploitation_rollouts.recurrent_hidden_states[step],
                                       exploration_rollouts.recurrent_hidden_states[step], exploitation_rollouts.masks[step], exploration_rollouts.masks[step], args.egreedy, j * args.num_steps + step, deterministic=False)

                else:
                    plt_value, plr_value, action, action_raw, plt_action_log_prob, plr_action_log_prob, plt_recurrent_hidden_states, plr_recurrent_hidden_states, agent_type, plt_action, plr_action, plt_action_raw, plr_action_raw = \
                        controller(args.env_name, exploitation_actor_critic, exploration_actor_critic, args.controller_coef, exploitation_rollouts.obs[step], exploration_rollouts.obs[step],
                                   exploitation_rollouts.recurrent_hidden_states[step],
                                   exploration_rollouts.recurrent_hidden_states[step], exploitation_rollouts.masks[step], exploration_rollouts.masks[step], args.egreedy, j * args.num_steps + step, deterministic=False)

            if agent_type[PRINT_INDEX] == 'plt':
                count_plt = count_plt + 1
                policy_history = np.append(policy_history, [1])
            elif agent_type[PRINT_INDEX] == 'plr':
                count_plr = count_plr + 1
                policy_history = np.append(policy_history, [0])

            plt_dist = exploitation_actor_critic.get_dist(exploitation_rollouts.obs[step], exploitation_rollouts.recurrent_hidden_states[step], exploitation_rollouts.masks[step])
            plr_dist = exploration_actor_critic.get_dist(exploration_rollouts.obs[step], exploration_rollouts.recurrent_hidden_states[step], exploration_rollouts.masks[step])


            history_plt_mean = np.append(history_plt_mean, plt_dist.mean.cpu().detach().numpy()[0][PRINT_INDEX])
            history_plr_mean = np.append(history_plr_mean, plr_dist.mean.cpu().detach().numpy()[0][PRINT_INDEX])

            history_plt_stddev = np.append(history_plt_stddev, plt_dist.stddev.cpu().detach().numpy()[0][PRINT_INDEX])
            history_plr_stddev = np.append(history_plr_stddev, plr_dist.stddev.cpu().detach().numpy()[0][PRINT_INDEX])

            if step % 100 is 0:
                print("Step", step, ":", action)
            # actual_action = torch.clone(action)
            # # 0 : steer, 1 : throttle, 2 : brake

            obs, reward, done, infos = envs.step(action)

            # print("reward_norm", reward)

            for info in infos:
                if 'episode' in info.keys():
                    print("Reward of this episode :", info['episode']['r'])
                    episode_rewards.append(info['episode']['r'])
                    if info['episode']['r'] > 80:
                        print("[EPISODE {} POLICY TYPE] EXPLOITATION : {}  EXPLORATION : {} REWARD : {}\n".format(j, count_plt, count_plr, info['episode']['r']))

                    # For figures
                    points_plt = np.array([[]])
                    points_plr = np.array([[]])

                    fig = plot.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    X_SCOPE = 10
                    xx = np.linspace(-X_SCOPE, X_SCOPE, 100)

                    # plr
                    history_length = history_plt_mean.shape[0]
                    history_frequency = int(history_length / 5)

                    try:
                        for y in range(history_length):
                            if y % history_frequency == 0:

                                # print("PLT MEAN", history_plt_mean[y])
                                # print("PLT STEDDEV", history_plt_stddev[y])
                                #
                                # print("PLR MEAN", history_plr_mean[y])
                                # print("PLR_STDDEV", history_plr_stddev[y])

                                rv_plt = sp.stats.norm(history_plt_mean[y], history_plt_stddev[y])
                                rv_plr = sp.stats.norm(history_plr_mean[y], history_plr_stddev[y])

                                pdf_plt = rv_plt.pdf(xx)
                                pdf_plr = rv_plr.pdf(xx)

                                for xs in range(100):
                                    points_plt = np.append(points_plt, [-X_SCOPE + xs * (X_SCOPE * 2 / 100), y, pdf_plt[xs]])
                                    points_plr = np.append(points_plr, [-X_SCOPE + xs * (X_SCOPE * 2 / 100), y, pdf_plr[xs]])

                                points_plt = np.reshape(points_plt, [int(points_plt.shape[0] / 3), 3])
                                points_plr = np.reshape(points_plr, [int(points_plr.shape[0] / 3), 3])

                                pointx_plt = np.array([])
                                pointy_plt = np.array([])
                                pointz_plt = np.array([])

                                pointx_plr = np.array([])
                                pointy_plr = np.array([])
                                pointz_plr = np.array([])

                                for point in points_plt:
                                    pointx_plt = np.append(pointx_plt, [point[0]])
                                    pointy_plt = np.append(pointy_plt, [point[1]])
                                    pointz_plt = np.append(pointz_plt, [point[2]])

                                for point in points_plr:
                                    pointx_plr = np.append(pointx_plr, [point[0]])
                                    pointy_plr = np.append(pointy_plr, [point[1]])
                                    pointz_plr = np.append(pointz_plr, [point[2]])

                                ax.plot3D(pointx_plt, pointy_plt, pointz_plt, 'b')
                                ax.plot3D(pointx_plr, pointy_plr, pointz_plr, 'r--')

                                points_plt = np.array([[]])
                                points_plr = np.array([[]])

                        plot.savefig(SAVE_DIR + "figure/Episode{}_plt{}_plr{}.png".format(int(j), count_plt, count_plr), dpi=300)
                        plot.clf()

                    except ZeroDivisionError:
                        pass

                    TRAINING_DICT["training_reward"] = info['episode']['r']
                    TRAINING_DICT["count_plt"] = count_plt
                    TRAINING_DICT["count_plr"] = count_plr
                    TRAINING_DICT["policy_dist"] = str(policy_history.tolist()).splitlines()

                    tmp_training_dict = TRAINING_DICT.copy()

                    TRAINING_DICT_LIST.append(tmp_training_dict)

                    # Re-Initialize
                    policy_history = np.array([])
                    count_plt = 0
                    count_plr = 0

                    history_plt_mean = np.array([])
                    history_plr_mean = np.array([])
                    history_plt_stddev = np.array([])
                    history_plr_stddev = np.array([])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            with torch.no_grad():
                if agent_type[0] == 'plr':
                    _, plt_action_log_prob, _, _ = exploitation_actor_critic.evaluate_actions(exploitation_rollouts.obs[step], exploitation_rollouts.recurrent_hidden_states[step], exploitation_rollouts.masks[step], plr_action_raw)

                elif agent_type[0] == 'plt':
                    _, plr_action_log_prob, _, _ = exploration_actor_critic.evaluate_actions(exploration_rollouts.obs[step], exploration_rollouts.recurrent_hidden_states[step], exploration_rollouts.masks[step], plt_action_raw)

            exploitation_rollouts.insert(obs, plt_recurrent_hidden_states, action_raw, plt_action_log_prob, plt_value, reward, masks, bad_masks)
            exploration_rollouts.insert(obs, plr_recurrent_hidden_states, action_raw, plr_action_log_prob, plr_value, reward, masks, bad_masks)

        print("STEP STAGE OUT!")
        if 'Carla' in args.env_name:
            envs.pause()

        with torch.no_grad():
            print("GETTING VALUES FROM ACTOR-CRITICS")
            plt_next_value = exploitation_actor_critic.get_value(exploitation_rollouts.obs[-1], exploitation_rollouts.recurrent_hidden_states[-1], exploitation_rollouts.masks[-1]).detach()
            plr_next_value = exploration_actor_critic.get_value(exploration_rollouts.obs[-1], exploration_rollouts.recurrent_hidden_states[-1], exploration_rollouts.masks[-1]).detach()

        print("START COMPUTING DISCR REWARD...")
        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, exploitation_rollouts, utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                discr_reward = discr.predict_reward(exploitation_rollouts.obs[step], exploitation_rollouts.actions[step], args.gamma, exploitation_rollouts.masks[step])
                if step % 500 == 0:
                    print("env reward :", exploitation_rollouts.rewards[step])
                    print("exp reward :", discr_reward)

                w1 = args.extr_reward_weight
                w2 = args.expert_reward_weight
                exploitation_rollouts.rewards[step] = w1 * exploitation_rollouts.rewards[step] + w2 * discr_reward

        print("FINISH COMPUTING DISCR REWARD...")
        print("START COMPUTING DISCR REWARD...")
        if args.icm:
            for _ in range(args.icm_epoch):
                intr.update(exploration_rollouts, args.icm_batch_size)

            for step in range(args.num_steps):
                intr_reward = intr.predict_reward(exploration_rollouts.obs[step], exploration_rollouts.next_obs[step], exploration_rollouts.actions[step]).detach()
                exploration_rollouts.rewards[step] = intr_reward
                if step % 500 == 0:
                    print("int reward :", intr_reward)

                # if step % 300 == 0:
                #     print("plr reward :", exploration_rollouts.rewards[step])

        print("FINISH COMPUTING DISCR REWRAD...")
        print("START COMPUTING ROLLOUTS RETURNS")
        exploitation_rollouts.compute_returns(plt_next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        exploration_rollouts.compute_returns(plr_next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        print("START UPDATING ROLLOUTS")
        exploitation_value_loss, exploitation_action_loss, exploitation_dist_entropy = exploitation_agent.update(exploitation_rollouts)
        exploration_value_loss, exploration_action_loss, exploration_dist_entropy = exploration_agent.update(exploration_rollouts)

        exploitation_rollouts.after_update()
        exploration_rollouts.after_update()

        print("ROLLOUTS UPDATE COMPLETE")

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and SAVE_DIR != "":
            save_path = os.path.join(SAVE_DIR, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([exploitation_actor_critic, exploration_actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \nLast {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}\n"
                .format(j, total_num_steps, int(total_num_steps / (end - start)), len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
            print(
                "[EXPLOITATION] dist_entropy {:.5f}, value_loss {:.5f}, action_loss {:.5f}"
                .format(exploitation_dist_entropy, exploitation_value_loss, exploitation_action_loss))
            print(
                "[EXPLORATION] dist_entropy {:.5f}, value_loss {:.5f}, action_loss {:.5f}\n"
                .format(exploration_dist_entropy, exploration_value_loss, exploration_action_loss))

            EPISODE_DICT["entropy_plt"] = exploitation_dist_entropy
            EPISODE_DICT["entropy_plr"] = exploration_dist_entropy
            EPISODE_DICT["value_loss_plt"] = exploitation_value_loss
            EPISODE_DICT["value_loss_plr"] = exploration_value_loss
            EPISODE_DICT["action_loss_plt"] = exploitation_action_loss
            EPISODE_DICT["action_loss_plr"] = exploration_action_loss

            tmp_episode_dict = EPISODE_DICT.copy()
            EPISODE_DICT_LIST.append(tmp_episode_dict)

        # if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     if args.controller_coef < 5000:
        #         evaluation_reward = evaluate(exploitation_actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
        #     else:
        #         evaluation_reward = evaluate(exploration_actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
        #
        #     EVALUATION_REWARDS_LIST.append(evaluation_reward)

        if 'Carla' in args.env_name:
            envs.resume()


if __name__ == "__main__":
    main()
