from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import random
import math
import numpy as np
import copy
from collections import deque

from gym.spaces import Box, Discrete, Tuple, Dict

from carla2gym.core.maps.nodeid_coord_map import TOWN01, TOWN02, TOWN03
from carla2gym.carla.reward import Reward
from carla2gym.carla.scenarios import update_scenarios_parameter
from carla2gym.core.sensors.derived_sensors import CollisionSensor
from carla2gym.core.sensors.derived_sensors import LaneInvasionSensor

import sys
import os
import glob

try:
    sys.path.append('/home/hsyoon/software/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
except IndexError:
    print("Error!!!")
    pass

import carla


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def mycar_grid(x, y):
    if 25.0 <= x <= 40.0 and -25 <= y <= 0:
        return np.array([1, 0, 0, 0, 0])

    elif 0 <= x <= 25.0 and -25 <= y <= 0:
        return np.array([0, 1, 0, 0, 0])

    elif 0 <= x <= 25.0 and -42 <= y <= -25:
        return np.array([0, 0, 1, 0, 0])

    elif 0 <= x <= 25.0 and y <= -42:
        return np.array([0, 0, 0, 1, 0])

    else:
        return np.array([0, 0, 0, 0, 1])


def othercar_grid(x, y):
    if 0 <= x <= 25 and -25 <= y <= 0:
        return np.array([1, 0, 0, 0, 0])

    elif -25 <= x <= 0 and -25 <= y <= 0:
        return np.array([0, 1, 0, 0, 0])

    elif -25 <= x <= 0 and 0 <= y <= 25:
        return np.array([0, 0, 1, 0, 0])

    elif 0 <= x <= 25 and 0 <= y <= 25:
        return np.array([0, 0, 0, 1, 0])

    else:
        return np.array([0, 0, 0, 0, 1])



class CarlaUrbanIntersectionEnvMini(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        self.scenario = "UrbanIntersection"
        self.client = None
        self.world = None
        # action_space : [Steering angle, throttle, brake]
        self.action_space = Box(np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = np.array([0.0 for i in range(1 * 15)])
        self.reward = None
        self.reward_cdrange = (-float('inf'), float('inf'))
        self.map = "Town03"

        self.mycar = None
        self.car1 = None
        self.car2 = None
        self.car3 = None
        self.car4 = None
        self.car_tmp = None

        self.collisions_sensor = None
        self.lane_invasions_sensor = None
        # self._collisions = None
        # self._lane_invasions = None
        # Save the list of actor ids and remove it when done = True
        self.actor_list = None

        self.blueprint_mycar = None
        self.blueprint_car1 = None
        self.blueprint_car2 = None
        self.blueprint_car3 = None
        self.blueprint_car4 = None

        self.transformation_mycar = None
        self.transformation_car1 = None
        self.transformation_car2 = None
        self.transformation_car3 = None
        self.transformation_car4 = None

        self.location_mycar = None
        self.location_car1 = None
        self.location_car2 = None
        self.location_car3 = None
        self.location_car4 = None

        # Carla Setting
        self.server_host = "localhost"
        self.server_port = 8427
        self.server_port_eval = 8423
        self.location_end = carla.Location(4.1, -46.9, 0)
        self.location_middle = carla.Location(15, -15, 0)
        self.Rotation_start = carla.Rotation(0, 0, 0)
        self.Rotation_end = carla.Rotation(pitch=-0.131256, yaw=-90.461967, roll=-0.063721)

        self.goal_distance_prev = 0
        self.goal_distance_curr = 0

        self.collision_prev_prev = 0
        self.collision_prev = 0
        self.collision_curr = 0
        self.reach_middle_point = False
        self.collision_hist = deque([0, 0, 0])


        self.invasion_prev_prev = 0
        self.invasion_prev = 0
        self.invasion_curr = 0
        self.invasion_hist = deque([0, 0, 0])

        self.step_count = 0
        self.new_collision = 0
        self.new_invasion = 0

        weather = carla.WeatherParameters(cloudyness=1.0, precipitation=10.0, sun_altitude_angle=70.0)

        self.client = None
        print("@@@ Initializing carla server ...")
        while self.client is None:
            try:
                print("Connecting to carla client with", self.server_host, ":", self.server_port)
                self.client = carla.Client(self.server_host, self.server_port)
                self.client.set_timeout(5.0)
                print(self.client.get_server_version())
            except RuntimeError as re:
                self.client = None
        print("Connected to carla client with", self.server_host, ":", self.server_port)
        self.world = None
        while self.world is None:
            try:
                print("Connecting to the carla world ...")
                self.world = self.client.get_world()
                self.world.set_weather(weather)
            except RuntimeError as re:
                self.world = None
        print("Carla world made")

    # ============================================
    # == reset ===================================
    # ============================================
    def reset(self):

        # Blueprints, Transformations, Locations of vehicles
        self.blueprint_mycar = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.tt'))
        self.blueprint_car1 = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
        self.blueprint_car2 = random.choice(self.world.get_blueprint_library().filter('vehicle.bmw.grandtourer'))
        self.blueprint_car3 = random.choice(self.world.get_blueprint_library().filter('vehicle.chevrolet.impala'))
        self.blueprint_car4 = random.choice(self.world.get_blueprint_library().filter('vehicle.ford.mustang'))

        self.transformation_mycar = carla.Transform(carla.Location(38.9, -4.4, 2), carla.Rotation(0, 180, 0))
        self.transformation_car1 = carla.Transform(carla.Location(40.7, -7.9, 2), carla.Rotation(0, 180, 0))
        self.transformation_car2 = carla.Transform(carla.Location(22.471565, 4.197405, 2), carla.Rotation(0.089967, -79.148270, -0.010895))
        self.transformation_car3 = carla.Transform(carla.Location(13.526010, 14.151862, 1), carla.Rotation(0.206647, -49.756748, 0.130238))
        self.transformation_car4 = carla.Transform(carla.Location(-1.023689, 20.312210, 1), carla.Rotation(0.134186, 2.310838, 0.005359))

        self.location_mycar = carla.Location(38.9, -4.4, 2)
        self.location_car1 = carla.Location(40.7, -7.9, 2)
        self.location_car2 = carla.Location(22.471565, 4.197405, 2)
        self.location_car3 = carla.Location(13.526010, 14.151862, 1)
        self.location_car4 = carla.Location(-1.023689, 20.312210, 1)

        self.collision_prev_prev = 0
        self.collision_prev = 0
        self.collision_curr = 0
        self.reach_middle_point = False
        self.collision_hist = deque([0, 0, 0])

        self.invasion_prev_prev = 0
        self.invasion_prev = 0
        self.invasion_curr = 0
        self.invasion_hist = deque([0, 0, 0])

        self.new_collision = 0
        self.new_invasion = 0

        blueprint_tmp = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.etron'))
        tmp_transform = carla.Transform(carla.Location(69.489059, -204.821091, 0.1), carla.Rotation(0.061110, 2.803965, -0.020020))

        if self.mycar is not None:
            self.mycar.set_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.mycar.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.mycar.set_transform(self.transformation_mycar)

        else:
            while self.mycar is None:
                try:
                    self.mycar = self.world.try_spawn_actor(self.blueprint_mycar, self.transformation_mycar)
                    self.collisions_sensor = CollisionSensor(self.mycar)
                    self.lane_invasions_sensor = LaneInvasionSensor(self.mycar)
                    # self.mycar.set_autopilot()
                except AttributeError as ie:
                    self.mycar = None

        if self.car1 is not None:
            self.car1.set_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.car1.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.car1.set_transform(self.transformation_car1)

        else:
            while self.car1 is None:
                try:
                    self.car1 = self.world.try_spawn_actor(self.blueprint_car1, self.transformation_car1)
                    self.car1.set_autopilot(True)
                except AttributeError as ae:
                    self.car1 = None

        # if self.car2 is not None:
        #     self.car2.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        #     self.car2.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
        #     self.car2.set_transform(self.transformation_car2)
        #
        # else:
        #     while self.car2 is None:
        #         try:
        #             self.car2 = self.world.try_spawn_actor(self.blueprint_car2, self.transformation_car2)
        #             self.car2.set_autopilot(True)
        #         except AttributeError as ie:
        #             self.car2 = None

        #
        # if self.car3 is not None:
        #     self.car3.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        #     self.car3.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
        #     self.car3.set_transform(self.transformation_car3)
        #
        # else:
        #     while self.car3 is None:
        #         try:
        #             self.car3 = self.world.try_spawn_actor(self.blueprint_car3, self.transformation_car3)
        #             self.car3.set_autopilot(True)
        #         except AttributeError as ie:
        #             self.car3 = None
        #
        # if self.car4 is not None:
        #     self.car4.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        #     self.car4.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
        #     self.car4.set_transform(self.transformation_car4)
        #
        # else:
        #     while self.car4 is None:
        #         try:
        #             self.car4 = self.world.try_spawn_actor(self.blueprint_car4, self.transformation_car4)
        #             self.car4.set_autopilot(True)
        #         except AttributeError as ie:
        #             self.car4 = None
        #
        # if self.car_tmp is None:
        #     while self.car_tmp is None:
        #         try:
        #             self.car_tmp = self.world.try_spawn_actor(blueprint_tmp, tmp_transform)
        #
        #         except AttributeError as ie:
        #             self.car_tmp = None

        print("Cars reset finished")

        if self.mycar is not None:
            self.collisions_sensor.reset()
            self.lane_invasions_sensor.reset()
            # self.collisions_sensor = CollisionSensor(self.mycar)
            # self.lane_invasions_sensor = LaneInvasionSensor(self.mycar)

        print("Initializing carla completed!\n\n")

        world_snapshot_next = self.world.get_snapshot()

        # Roundabout
        location_intersection = carla.Location(x=26.206285, y=-4.993992, z=-0.002554)

        next_obs = np.zeros(self.observation_space.shape[0])

        for actor_snapshot in world_snapshot_next:

            actual_actor = self.world.get_actor(actor_snapshot.id)

            if 'vehicle.audi.tt' in actual_actor.type_id:
                actor_snapshot_mycar = actor_snapshot

                next_obs[0:5] = mycar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)
                next_obs[5] = actor_snapshot.get_transform().rotation.yaw  # orientation
                next_obs[6] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
                next_obs[7] = 1.0 - float(abs(actor_snapshot.get_transform().location.x - self.location_end.x) + abs(actor_snapshot.get_transform().location.y - self.location_end.y)) / float(
                    abs(self.location_mycar.x - self.location_end.x) + abs(self.location_mycar.y - self.location_end.y))

        for actor_snapshot in world_snapshot_next:
            actual_actor = self.world.get_actor(actor_snapshot.id)
            if 'vehicle.tesla.model3' in actual_actor.type_id:
                next_obs[8:13] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                next_obs[13] = actor_snapshot.get_transform().rotation.yaw  # orientation
                next_obs[14] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

            # if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
            #     next_obs[15:20] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
            #     next_obs[20] = actor_snapshot.get_transform().rotation.yaw  # orientation
            #     next_obs[21] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

            # if 'vehicle.chevrolet.impala' in actual_actor.type_id:
            #     next_obs[22:27] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
            #     next_obs[27] = actor_snapshot.get_transform().rotation.yaw  # orientation
            #     next_obs[28] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
            #
            # if 'vehicle.ford.mustang' in actual_actor.type_id:
            #     next_obs[29:34] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
            #     next_obs[34] = actor_snapshot.get_transform().rotation.yaw  # orientation
            #     next_obs[35] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

        # settings = self.world.get_settings()
        # settings.synchronous_mode = False
        # self.world.apply_settings(settings)

        return next_obs

    # ============================================
    # == step ====================================
    # ============================================
    def step(self, action):
        action = action.tolist()
        done = False
        info = {}

        steering_wheel = action[0]
        throttle = action[1]
        brake = action[2]
        # if action[2] < 0.5:
        #     brake = 0
        # else:
        #     brake = 1

        # steering_wheel = np.tanh(action[0])
        # throttle = sigmoid(action[1])
        # brake = sigmoid(action[2])

        #
        # if brake < 0.5:
        #     brake = 0
        # else:
        #     brake = 1

        self.mycar.apply_control(carla.VehicleControl(throttle=throttle, steer=steering_wheel, brake=brake, hand_brake=False, reverse=False))

        world_snapshot_next = self.world.get_snapshot()

        actor_snapshot_mycar = None
        location_intersection = carla.Location(x=26.206285, y=-4.993992, z=-0.002554)

        next_obs = np.zeros(self.observation_space.shape[0])

        for actor_snapshot in world_snapshot_next:

            actual_actor = self.world.get_actor(actor_snapshot.id)

            if 'vehicle.audi.tt' in actual_actor.type_id:
                actor_snapshot_mycar = actor_snapshot

                next_obs[0:5] = mycar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)
                next_obs[5] = actor_snapshot.get_transform().rotation.yaw  # orientation
                next_obs[6] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
                next_obs[7] = 1.0 - float(abs(actor_snapshot.get_transform().location.x - self.location_end.x) + abs(actor_snapshot.get_transform().location.y - self.location_end.y)) / float(
                    abs(self.location_mycar.x - self.location_end.x) + abs(self.location_mycar.y - self.location_end.y))

        for actor_snapshot in world_snapshot_next:
            actual_actor = self.world.get_actor(actor_snapshot.id)
            if 'vehicle.tesla.model3' in actual_actor.type_id:
                next_obs[8:13] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                next_obs[13] = actor_snapshot.get_transform().rotation.yaw  # orientation
                next_obs[14] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

            # if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
            #     next_obs[15:20] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
            #     next_obs[20] = actor_snapshot.get_transform().rotation.yaw  # orientation
            #     next_obs[21] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

            # if 'vehicle.chevrolet.impala' in actual_actor.type_id:
            #     next_obs[22:27] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
            #     next_obs[27] = actor_snapshot.get_transform().rotation.yaw  # orientation
            #     next_obs[28] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
            #
            # if 'vehicle.ford.mustang' in actual_actor.type_id:
            #     next_obs[29:34] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
            #     next_obs[34] = actor_snapshot.get_transform().rotation.yaw  # orientation
            #     next_obs[35] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

        collisions_hist = self.collisions_sensor.get_collision_history()
        invasions_hist = self.lane_invasions_sensor.get_invasion_history()

        # print("collisions_hist", collisions_hist)

        diff_x = self.mycar.get_location().x - self.location_end.x
        diff_y = self.mycar.get_location().y - self.location_end.y

        diff_x = math.pow(diff_x, 2)
        diff_y = math.pow(diff_y, 2)

        middle_dist = math.sqrt(math.pow(self.mycar.get_location().x - self.location_middle.x, 2) + math.pow(self.mycar.get_location().y - self.location_middle.y, 2))

        final_destination_distance = math.sqrt(diff_x + diff_y)

        if self.mycar.get_location().x >= self.location_middle.x:
            destination_distance = abs(self.location_end.y - self.location_middle.y) + abs(self.mycar.get_location().x - self.location_middle.x)

        else:
            destination_distance = abs(self.location_middle.x - self.location_end.x) + abs(self.mycar.get_location().y - self.location_end.y)

        done_distance = 5.0

        """
        self.reward = 0.0
        self.reward += np.clip(self.prev["distance_to_goal"] - cur_dist = self.curr["distance_to_goal"], -1.0, 1.0)
        self.reward += 0.05 * (self.curr["forward_speed"] - self.prev["forward_speed"])
        self.reward -= .00002 * (self.curr["collision_vehicles"] + self.curr["collision_pedestrians"] + self.curr["collision_other"] - self.prev["collision_vehicles"] - self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 2 * (self.curr["intersection_offroad"] - self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 2 * (self.curr["intersection_otherlane"] - self.prev["intersection_otherlane"])
        """

        reward = 0.0

        # For the first step of the episode
        if self.goal_distance_prev == 0:
            self.goal_distance_prev = destination_distance
            self.goal_distance_curr = destination_distance
        else:
            self.goal_distance_prev = self.goal_distance_curr
            self.goal_distance_curr = destination_distance

        reward = reward + np.clip(self.goal_distance_prev - self.goal_distance_curr, -10.0, 10.0)
        # reward = reward - 0.1 * np.square(action).sum()

        if len(collisions_hist) != 0:
            reward = reward - 3
            info['bad_transition'] = True
            print("COLLISION!!")
            done = True

        if len(invasions_hist) != 0:
            if next_obs[1] != 1:
                reward = reward - 3
                print("INVASION TYPE 1!!")
                info['bad_transition'] = True
                done = True

        if next_obs[4] == 1 or actor_snapshot_mycar.get_transform().location.y >= -2.0 or (actor_snapshot_mycar.get_transform().location.x >= 25.0 and actor_snapshot_mycar.get_transform().location.y <= -5.0):
            reward = reward - 3
            print("INVASION TYPE 2!!")
            info['bad_transition'] = True
            done = True

        # check if car1 and car2's position are outer than criteria, then destroy it and respawn the car
        for actor_snapshot in world_snapshot_next:
            actual_actor = self.world.get_actor(actor_snapshot.id)
            if 'vehicle.tesla.model3' in actual_actor.type_id:
                if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                    actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                    actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                    actual_actor.set_transform(self.transformation_car1)

            # if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
            #     if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
            #         actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
            #         actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
            #         actual_actor.set_transform(self.transformation_car2)

            # if 'vehicle.chevrolet.impala' in actual_actor.type_id:
            #     if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
            #         actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
            #         actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
            #         actual_actor.set_transform(self.transformation_car3)
            #
            # if 'vehicle.ford.mustang' in actual_actor.type_id:
            #     if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
            #         actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
            #         actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
            #         actual_actor.set_transform(self.transformation_car4)

        if final_destination_distance <= done_distance:
            reward = reward + 100
            print("GET DESTINATION POINT!!")
            done = True

        if middle_dist <= 5 and self.reach_middle_point is False:
            self.reach_middle_point = True
            print("GET MIDDLE POINT!!")
            reward = reward + 50

        self.step_count = self.step_count + 1

        if self.step_count == 600:
            self.step_count = 0
            done = True

        reward = np.exp(reward / 10)
        return next_obs, reward, done, {}

    def pause(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)

    def resume(self):
        self.world.tick()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

