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
import carla

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

def main():
    args = get_args()

    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)


    episode = 1
    success_episode = 0
    total_obs = None
    total_action = None
    total_reward = None
    total_len = None

    # Just get carla client and make world
    client = envs.carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = envs.world
    world.reset()
    world.mycar.set_autopilot()

    if 'mini' in args.filename:
        world.observation_space_dim = 8

    while True:
        obs = np.array([])
        action = np.array([])
        reward = np.array([])
        len = np.array([])
        termination = False
        save = False
        count = 0
        episode_step = 0
        # print("save", save)
        # print("termination", termination)
        # world.tick_without_clock()

        while termination is False:
            episode_step = episode_step + 1

            if episode_step > 600:
                save = False
                termination = True
                episode = episode + 1
                print("TOO LONG STEP!")
                world.restart()
                break


            """
            # Intersection 4
            destination_location = carla.Location(x=1.869288, y=111.384148, z=-0.004142)
            """

            # Roundabout
            destination_location = carla.Location(4.1, -46.9, 0)
            location_intersection = carla.Location(x=26.206285, y=-4.993992, z=-0.002554)

            """
            termination = True if 
            1. Car gets to the destination point (save = True) 
            2. Go left at the intersection (save = False)

            3. Episode step becomes larger than 600 
            """

            world_snapshot = world.get_snapshot()

            # Find my car and check if i am
            # 1. going right left direction
            # 2. reaching to my destination point
            actor_snapshot_mycar = None

            for actor_snapshot in world_snapshot:
                actual_actor = world.get_actor(actor_snapshot.id)

                # ===================================================================================================================
                # Temporary Observation

                # 0~1 : x, y of the location of the intersection
                # 2~3 : x, y of the location of mycar
                # 4~9 : properties of car1 (x,y value of location, velocity, acceleration)
                # 10~15 : properties of car2 (x,y value of location, velocity, acceleration)
                # 16~21 : properties of car3 (x,y value of location, velocity, acceleration)
                # 22~27 : properties of car4 (x,y value of location, velocity, acceleration)
                # ===================================================================================================================

                tmp_obs = np.zeros(world.observation_space_dim)

                tmp_obs[0] = location_intersection.x
                tmp_obs[1] = location_intersection.y

                if 'vehicle.audi.tt' in actual_actor.type_id:
                    actor_snapshot_mycar = actor_snapshot

                    tmp_obs[2] = actor_snapshot.get_transform().location.x
                    tmp_obs[3] = actor_snapshot.get_transform().location.y
                    tmp_obs[4] = actor_snapshot.get_velocity().x
                    tmp_obs[5] = actor_snapshot.get_velocity().y
                    tmp_obs[6] = actor_snapshot.get_acceleration().x
                    tmp_obs[7] = actor_snapshot.get_acceleration().y

                    # print("x", tmp_obs[2], "y", tmp_obs[3])

                    destination_distance = math.sqrt(math.pow(actor_snapshot.get_transform().location.x - destination_location.x, 2) + math.pow(actor_snapshot.get_transform().location.y - destination_location.y, 2) + math.pow(
                        actor_snapshot.get_transform().location.z - destination_location.z, 2))

                    if count % 500 is 0:
                        print("=================================================================================")
                        print("                               EXPERT DEMO LOG")
                        print("episode :", episode)
                        print("success episode :", success_episode)
                        print("step :", count)
                        print("destination distance :", destination_distance)
                        print("current steer :", actual_actor.get_control().steer)
                        print("current throttle :", actual_actor.get_control().throttle)
                        print("current brake :", actual_actor.get_control().brake)
                        print("=================================================================================")
                    """
                    # Intersection 4
                    if -18.0 < actor_snapshot.get_transform().location.x < -14.0 and actual_actor.get_control().steer >= 0:
                    """
                    tmp_brake = actual_actor.get_control().brake

                    if actual_actor.get_control().brake == 0:
                        tmp_brake = 0
                    else:
                        tmp_brake = 0.5

                    # action = np.append(action, [actual_actor.get_control().steer, actual_actor.get_control().throttle, actual_actor.get_control().brake])
                    action = np.append(action, [actual_actor.get_control().steer, actual_actor.get_control().throttle, tmp_brake])
                    reward = np.append(reward, [0.5])

                    # Roundabout
                    if actor_snapshot.get_transform().location.x < 10 and actual_actor.get_control().steer < 0 or actor_snapshot.get_transform().location.x < -0.5:
                        save = False
                        termination = True
                        episode = episode + 1

                        print("\n\nWrong way with x value {} and steering angle {}\n\n".format(actor_snapshot.get_transform().location.x, actual_actor.get_control().steer))
                        break

                    # if destination_distance <= 5.0:
                    #     print("episode_step", episode_step)

                    if destination_distance <= 5.0 and episode_step > 200:
                        save = True
                        termination = True
                        episode = episode + 1
                        success_episode = success_episode + 1

                        print("\n\nSuccess!!")
                        break

            # ===================================================================================================================
            # Record Observation Space

            # Assign the indexes of the observation values for own actor car
            # ===================================================================================================================
            if 'mini' not in args.filename:
                for actor_snapshot in world_snapshot:
                    actual_actor = world.get_actor(actor_snapshot.id)
                    if 'vehicle.tesla.model3' in actual_actor.type_id:
                        # print("actor_snapshot.get_transform().location.x", actor_snapshot.get_transform().location.x)
                        # print("actor_snapshot_mycar.get_transform().location.x", actor_snapshot_mycar.get_transform().location.x)

                        tmp_obs[4] = actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location.x
                        tmp_obs[5] = actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location.y
                        tmp_obs[6] = actor_snapshot.get_velocity().x - actor_snapshot_mycar.get_velocity().x
                        tmp_obs[7] = actor_snapshot.get_velocity().y - actor_snapshot_mycar.get_velocity().y
                        tmp_obs[8] = actor_snapshot.get_acceleration().x - actor_snapshot_mycar.get_acceleration().x
                        tmp_obs[9] = actor_snapshot.get_acceleration().y - actor_snapshot_mycar.get_acceleration().y

                    if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
                        tmp_obs[10] = actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location.x
                        tmp_obs[11] = actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location.y
                        tmp_obs[12] = actor_snapshot.get_velocity().x - actor_snapshot_mycar.get_velocity().x
                        tmp_obs[13] = actor_snapshot.get_velocity().y - actor_snapshot_mycar.get_velocity().y
                        tmp_obs[14] = actor_snapshot.get_acceleration().x - actor_snapshot_mycar.get_acceleration().x
                        tmp_obs[15] = actor_snapshot.get_acceleration().y - actor_snapshot_mycar.get_acceleration().y

                    if 'vehicle.chevrolet.impala' in actual_actor.type_id:
                        tmp_obs[16] = actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location.x
                        tmp_obs[17] = actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location.y
                        tmp_obs[18] = actor_snapshot.get_velocity().x - actor_snapshot_mycar.get_velocity().x
                        tmp_obs[19] = actor_snapshot.get_velocity().y - actor_snapshot_mycar.get_velocity().y
                        tmp_obs[20] = actor_snapshot.get_acceleration().x - actor_snapshot_mycar.get_acceleration().x
                        tmp_obs[21] = actor_snapshot.get_acceleration().y - actor_snapshot_mycar.get_acceleration().y

                    if 'vehicle.ford.mustang' in actual_actor.type_id:
                        tmp_obs[22] = actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location.x
                        tmp_obs[23] = actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location.y
                        tmp_obs[24] = actor_snapshot.get_velocity().x - actor_snapshot_mycar.get_velocity().x
                        tmp_obs[25] = actor_snapshot.get_velocity().y - actor_snapshot_mycar.get_velocity().y
                        tmp_obs[26] = actor_snapshot.get_acceleration().x - actor_snapshot_mycar.get_acceleration().x
                        tmp_obs[27] = actor_snapshot.get_acceleration().y - actor_snapshot_mycar.get_acceleration().y

            obs = np.append(obs, tmp_obs)

            if save is True:
                break

            # ===================================================================================================================
            # Actor Boundary Check

            # Check if there are the cars which are out of boundary.
            # For actor which is out of boundary, respawn the actor to initial point for its own point.
            # ===================================================================================================================

            for actor_snapshot in world_snapshot:
                actual_actor = world.get_actor(actor_snapshot.id)
                if 'vehicle.tesla.model3' in actual_actor.type_id:
                    if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                        actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_transform(world.transformation_car1)

                if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
                    if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                        actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_transform(world.transformation_car2)

                if 'vehicle.chevrolet.impala' in actual_actor.type_id:
                    if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                        actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_transform(world.transformation_car3)

                if 'vehicle.ford.mustang' in actual_actor.type_id:
                    if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                        actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_transform(world.transformation_car4)

            count = count + 1

        # ===================================================================================================================
        # Save Expert Demonstrations.

        # Save the recorded expert demonstrations.
        # 100 demos are recorded in one .pt file.
        # ===================================================================================================================

        if save is True:
            max_time_step = 600
            max_episode = 50

            if obs.shape[0] < max_time_step * world.observation_space_dim:
                # Get current time step (0~99). Fill the empty spaces of expert demo.
                # Example below
                # If episode ends with 520 stesp, other 80 steps are filled with the end values.
                for i in range(max_time_step - int(obs.shape[0] / world.observation_space_dim)):
                    obs = np.append(obs, obs[-(world.observation_space_dim + 1):-1])
                    action = np.append(action, action[-(world.action_space_dim + 1):-1])
                    reward = np.append(reward, reward[-1])

            obs = np.reshape(obs, (obs.shape[0], 1))
            action = np.reshape(action, (action.shape[0], 1))
            reward = np.reshape(reward, (reward.shape[0], 1))

            obs = np.array([np.reshape(obs, (int(obs.shape[0] / world.observation_space_dim), world.observation_space_dim))])
            action = np.array([np.reshape(action, (int(action.shape[0] / world.action_space_dim), world.action_space_dim))])
            reward = np.array([np.reshape(reward, (int(reward.shape[0]), 1))])

            print("obs", obs.shape)
            print("action", action.shape)
            print("reward", reward.shape)

            if total_obs is None:
                total_obs = obs
                total_action = action
                total_reward = reward

            else:
                total_obs = np.append(total_obs, obs, axis=0)
                total_action = np.append(total_action, action, axis=0)
                total_reward = np.append(total_reward, reward, axis=0)

            if total_obs.shape[0] is max_episode:

                tmp_len = np.array([])
                for i in range(max_episode):
                    tmp_len = np.append(tmp_len, np.array([max_time_step]))

                states = torch.from_numpy(total_obs).float()
                actions = torch.from_numpy(total_action).float()
                rewards = torch.from_numpy(total_reward).float()
                lens = torch.from_numpy(tmp_len).long()

                data = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'lengths': lens
                }

                # now = datetime.now()
                # now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
                # now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

                torch.save(data, '/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-%s-%s.pt' % (args.filename, success_episode))
                print("=================================================================================")
                print(" DATA SAVED COMPLETED ")
                print("states:", total_obs.shape)
                print("actions:", total_action.shape)
                print("rewards:", total_reward.shape)
                print("tmp_len", tmp_len)
                print("=================================================================================")

                total_obs = None
                total_action = None
                total_reward = None
                total_len = None

        # Respawn the mycar actor.
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle.audi.tt' in actual_actor.type_id:
                actual_actor.set_transform(world.transformation_mycar)
                actual_actor.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False))

                print("MYCAR RESPAWNED!!")

if __name__ == "__main__":
    main()
