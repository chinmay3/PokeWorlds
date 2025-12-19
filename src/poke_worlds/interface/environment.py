from abc import abstractmethod, ABC
from typing import Optional, Type, Dict, Any, List, Tuple

from poke_worlds.utils import load_parameters, log_error, log_info, log_warn


from poke_worlds.emulation.emulator import Emulator
from poke_worlds.emulation.tracker import StateTracker
from poke_worlds.interface.controller import Controller

import numpy as np
import gymnasium as gym

class Environment(gym.Env, ABC):
    """ Base class for environments interfacing with the emulator. """

    REQUIRED_EMULATOR = Emulator
    """ The highest level emulator that the environment can interface with. """

    REQUIRED_TRACKER = StateTracker
    """ The state tracker that tracks the minimal state information required for the environment to function. """

    REQUIRED_CONTROLLER = Controller
    """ The highest level controller that provides actions to the emulator. """

    def __init__(self):
        """
        Ensures that the environment has the required attributes.
        All subclasses must call this __init__ method AFTER setting up the required attributes.

        """
        if not hasattr(self, "_parameters"):
            raise ValueError("Environment must have a '_parameters' attribute.")
        required_attributes = ["_emulator", "_controller", "observation_space", "action_space"]
        for attr in required_attributes:
            if not hasattr(self, attr):
                log_error(f"Environment requires attribute '{attr}' to be set. Implement this in the subclass __init__", self._parameters)
        if not issubclass(type(self._emulator), self.REQUIRED_EMULATOR):
            log_error(f"Environment requires an Emulator of type {self.REQUIRED_EMULATOR.NAME}, but got {type(self._emulator).NAME}", self._parameters)
        if not issubclass(type(self._emulator.state_tracker), self.REQUIRED_TRACKER):
            log_error(f"Environment requires a StateTracker of type {self.REQUIRED_TRACKER.NAME}, but got {type(self._emulator.state_tracker).NAME}", self._parameters)
        if not issubclass(type(self._controller), self.REQUIRED_CONTROLLER):
            log_error(f"Environment requires a Controller of type {self.REQUIRED_CONTROLLER.NAME}, but got {type(self._controller).NAME}", self._parameters)
        self._controller.assign_emulator(self._emulator)

    
    @abstractmethod
    def get_observation(self) -> gym.spaces.Space:
        """
        Returns the current observation from the emulator.
        Returns:
            observation (gym.spaces.Space): The current observation.
        """
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the full state information as defined by the emulator's state tracker.
        Returns:
            info (dict): The full state information from the state tracker.
        """
        return self._emulator.state_tracker.report()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment and emulator to the initial state.
        Args:
            seed (int, optional): Seed for random number generators.
            options (dict, optional): Additional options for resetting the environment.
        Returns:
            observation (object): The initial observation of the environment.

            info (dict): Additional information about the reset.
        """
        super().reset(seed=seed, options=options)

        self._emulator.reset()
        return self.get_observation(), self.get_info()
    
    @abstractmethod
    def determine_reward(self, start_state: Dict[str, Dict[str, Any]], action: gym.spaces.Space, transition_states: List[Dict[str, Dict[str, Any]]], action_success: bool) -> float:
        """
        Determines the reward based on the transition from start_state through transition_states.
        Args:
            start_state (Dict[str, Dict[str, Any]]): The state before the action was taken.
            action (gym.spaces.Space): The action taken.
            transition_states (List[Dict[str, Dict[str, Any]]]): A list of states observed during the action execution.
            action_success (bool): Whether the action was successful.
        Returns:
            float: The computed reward.
        """
        raise NotImplementedError
    
    @abstractmethod
    def determine_terminated(self, state: Dict[str, Dict[str, Any]]) -> bool:
        """
        Determines whether the episode reaches the goal / terminal state based on the transition from start_state through transition_states.
        This method is NOT meant to be used to determine if the step count has exceeded the maximum. 

        Args:
            state (Dict[str, Dict[str, Any]]): The current state after the action was taken.
        Returns:
            bool: Whether the episode is terminated.
        """
        pass

    
    def step(self, action: gym.spaces.Space) -> Tuple[gym.spaces.Space, float, bool, bool, Dict[str, Dict[str, Any]]]:
        """
        Executes the given action in the environment via the controller.

        Args:
            action (gym.spaces.Space): The action to execute.

        Returns:
            observation (gym.spaces.Space): The observation after executing the action.
            reward (float): The reward obtained from executing the action.
            terminated (bool): Whether the episode has ended (reached the terminal state of the MDP).
            truncated (bool): Whether the episode was truncated (exceeded the maximum allowed steps).
            info (Dict[str, Dict[str, Any]]): Full state information.
        """
        if self._emulator.check_if_done():
            log_error("Cannot step environment because emulator indicates done. Please reset the environment.", self._parameters)
        start_state = self.get_info()
        transition_states, action_success = self._controller.execute_action(action)
        truncated = self._emulator.check_if_done()
        observation = self.get_observation()
        current_state = self.get_info()
        terminated = self.determine_terminated(current_state)
        reward = self.determine_reward(start_state, action, transition_states, action_success)
        return observation, reward, terminated, truncated, current_state