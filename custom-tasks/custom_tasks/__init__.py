from gym.envs.registration import register

register(
    id='MountainGolfCar-v0',
    entry_point='custom_tasks.envs:MountainGolfCarEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainGolfCar-v1',
    entry_point='custom_tasks.envs:MountainGolfCarWithoutVelEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainToyCar-v0',
    entry_point='custom_tasks.envs:MountainToyCarEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainToyCar-v1',
    entry_point='custom_tasks.envs:MountainToyCarWithoutVelEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainOldCar-v1',
    entry_point='custom_tasks.envs:MountainOldCarWithoutVelEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainOldCarContinuous-v0',
    entry_point='custom_tasks.envs:MountainOldCarContinuousEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainOldCarContinuous-v1',
    entry_point='custom_tasks.envs:MountainOldCarContinuousWithoutVelEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='CarlaUrbanIntersection-v0',
    entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv'
)
register(
    id='CarlaUrbanIntersectionMini-v0',
    entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnvMini'
)
register(
    id='CarlaUrbanIntersection8423-v0',
    entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv8423'
)
register(
    id='CarlaUrbanIntersection8424-v0',
    entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv8424'
)
register(
    id='CarlaUrbanIntersection8425-v0',
    entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv8425'
)
# register(
#     id='CarlaUrbanIntersection8423_eval-v0',
#     entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv8423_eval'
# )
# register(
#     id='CarlaUrbanIntersection8424_eval-v0',
#     entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv8424_eval'
# )
# register(
#     id='CarlaUrbanIntersection8425_eval-v0',
#     entry_point='custom_tasks.envs:CarlaUrbanIntersectionEnv8425_eval'
# )
# register(
    # id='foo-extrahard-v0',
    # entry_point='gym_foo.envs:FooExtraHardEnv',
# )
