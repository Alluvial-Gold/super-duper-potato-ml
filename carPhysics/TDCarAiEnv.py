
import numpy as np

import gym




class TDCarEnv(gym.Env):
    def __init__(self):
        pass

        speed_bounds = [-10, 100]
        angle_bounds = [-90, 90]  # deg
        ray_bounds = [0, np.inf]
        num_rays = 11

        # Create the action space
        action_bounds = [speed_bounds, angle_bounds]
        action_low_bounds =  np.array([x[0] for x in action_bounds], dtype=np.float32)
        action_high_bounds = np.array([x[1] for x in action_bounds], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            action_low_bounds,
            action_high_bounds
        )

        self.action_space = gym.spaces.Discrete(2)


        # Create the observation space
        observation_bounds = [speed_bounds, angle_bounds] + num_rays*[ray_bounds]
        observation_low_bounds =  np.array([x[0] for x in observation_bounds], dtype=np.float32)
        observation_high_bounds = np.array([x[1] for x in observation_bounds], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            observation_low_bounds,
            observation_high_bounds
        )

    def step(self, action):
        pass

    def reset(self):
        pass


