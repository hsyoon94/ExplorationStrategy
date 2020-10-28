# import gym
# import custom_tasks
# import numpy as np
# from random import *
#
# env = gym.make("CarlaUrbanIntersection-v0")
#
# env.reset()
# next_obs = None
# reward = None
# done = False
# info = None
#
# while done is not True:
#   # random() : function that randomly makes float between 0 and 1
#   rnd_steering = (random() - 0.5) * 2
#   rnd_throttle = random()
#   rnd_brake = random()
#   rnd_action = np.array([rnd_steering, rnd_throttle, rnd_brake])
#
#   next_obs, reward, done, info = env.step(rnd_action)
#   print("reward", reward)

from datetime import datetime

now = datetime.now()

date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

print(date)
print(time)