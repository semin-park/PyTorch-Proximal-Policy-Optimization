from collections import deque

import gym
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv
import numpy as np
import cv2

# Below are copied or adapted from OpenAI's baselines.common.atari_wrappers.py
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)

    def observation(self, obs):
        return obs / 255.0


class ChannelFirstFrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Identical to OpenAI's FrameStack, except that this is channel first.
        """
        super(ChannelFirstFrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[-1] * k,) + shp[:-1]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        ob = ob.transpose(2,0,1)
        for _ in range(self.k):
            self.frames.append(ob)
        state = np.concatenate(self.frames, axis=0).astype(np.float32)
        return state

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = ob.transpose(2,0,1)
        self.frames.append(ob)
        state = np.concatenate(self.frames, axis=0).astype(np.float32)
        return state, reward, done, info

def make_env(env_name='PongNoFrameskip-v4', size=84, skip=4, scale=True, is_train=True):
    env = gym.make(env_name)
    env = NoopResetEnv(env, noop_max=30)
    if is_train:
        env = MaxAndSkipEnv(env, skip=skip)
    if env.unwrapped.ale.lives() > 0:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=size, height=size, grayscale=True) # obs_space is now (84,84,1)
    if scale:
        env = ScaledFloatFrame(env)
    env = ChannelFirstFrameStack(env, 4)
    return env
