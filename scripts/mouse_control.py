"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys
import gym

import numpy as np


from metaworld.envs.mujoco import *
from robosuite.devices import SpaceMouse
from metaworld.envs.mujoco.utils import rotation
from robosuite.utils.transform_utils import mat2quat, mat2euler
from robosuite.wrappers import DataCollectionWrapper
from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat

import gym
import metaworld
import time
import os
from shutil import copyfile

class MetaWorldDataCollectionWrapper(DataCollectionWrapper):

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        copyfile(self.env.model_name,xml_path)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            env=env_name,
        )
        self.states = []
        self.action_infos = []

    def reset(self):
        ret = super().reset()
        self._start_new_episode()
        return ret

    def step(self, action):
        ret = self.env.step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)
            self.action_infos.append({})

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret

space_mouse = SpaceMouse()
space_mouse.start_control()
env = MetaWorldDataCollectionWrapper(gym.make('SawyerPickupEnv-v0'), 'datas')
NDIM = env.action_space.low.size
lock_action = False
obs = env.reset()
action = np.zeros(10)
closed = False
last_rotation = None
grip = 0.0
while True:
    done = False
    env.render()

    state = space_mouse.get_controller_state()
    dpos, rotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["grasp"],
        state["reset"],
    )
    if last_rotation is None:
        last_rotation = np.copy(rotation)

    rotation = 0
    if grasp:
        dpos[:] = 0
        grip+=space_mouse.control[3]/5000.0
        if grip > 0.5:
            grip = 0.5
        elif grip < -0.5:
            grip = -0.5
        rotation = space_mouse.control[5]

    obs, reward, done, _ = env.step(np.hstack([dpos/0.1, rotation, grip]))
    space_mouse._reset_internal_state()

    # if done:
    #     obs = env.reset()
