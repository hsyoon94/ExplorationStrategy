carla2gym manual

[config arguments]

--controller-coef : KL Divergence threshold value for Policy Selector
--expert-reward-weight : weight of discriminator reward from expert demonstration
--extr-reward-weight : weight of environment reward from environment

(refer to commands.txt for more information)

[carla2gym requirements]
- python 3.5
- carla 0.9.6
- pytorch 1.3.1, tensorflow-gpu 1.14.0
- git clone openai baselines & install custom-tasks with command "$ pip install -e ."
- add import custom_tasks in main.py
- add PYTHONPATH in ~/.bashrc

- carla environment in custom-tasks/custom_tasks/envs/
- default environment is CarlaUrbanIntersection8425.py with "envs = gym.make("CarlaUrbanIntersection8425-v0")"


- carla environment functions 
def reset : reset agent location.
def step(action) : conduct given action then return next_obs, reward, done, info.
def pause, def resume : pause carla env then resume with synchronizing.

