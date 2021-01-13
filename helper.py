
from collections import deque
import numpy as np
import cv2
import torch
from gym.spaces import Box
from gym import Wrapper



class FrameStack(Wrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    def __init__(self, env, args):
        super(FrameStack, self).__init__(env)
        self.state_buffer = deque([], maxlen=args.history_length)
        self.env = env
        self.size = args.size
        self.device = args.device
        self.history_length = args.history_length

    def step(self, action):
        """

        map the actions 
        2 up  -> 0
        5 down -> 1
        3 right ->  2
        4 left -> 3
        11 jump right -> 4
        12 jump left -> 5

        """
        if action == 0:
            action = 2
        elif action == 1:
            action = 5
        elif action == 2:
            action = 3
        elif action == 3:
            action = 4
        elif action == 4:
            action = 11
        elif action == 5:
            action = 12
        elif action == 6:
            action = 0

        observation, reward, done, info = self.env.step(action)
        state = self._create_next_obs(observation)
        return state, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        state = self._stacked_frames(observation)
        return state
        
    def _create_next_obs(self, state):
        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.size, self.size))
        state = torch.tensor(state, dtype=torch.uint8, device=self.device)
        self.state_buffer.append(state)
        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs


    def _stacked_frames(self, state):
        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.size, self.size))
        state = torch.tensor(state, dtype=torch.uint8, device=self.device)
        zeros = torch.zeros_like(state)
        for idx in range(self.history_length - 1):
            self.state_buffer.append(zeros)
        self.state_buffer.append(state)

        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs

