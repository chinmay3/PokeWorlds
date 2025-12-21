from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional, Type
from enum import Enum
from poke_worlds.utils import verify_parameters, log_info, log_warn, log_error, load_parameters, get_lowest_level_subclass
from poke_worlds.emulation.emulator import Emulator, LowLevelActions
from poke_worlds.interface.action import HighLevelAction, LowLevelAction, RandomPlayAction, LowLevelPlayAction

import numpy as np
from gymnasium.spaces import OneOf, Space


class Controller(ABC):
    """ 
    Abstract base class for controllers interfacing with the emulator. 
    Handles conversion between high level actions and Gym action spaces.
    
    """
    ACTIONS: List[Type[HighLevelAction]] = [HighLevelAction]
    """ A list of HighLevelAction classes that define the possible high level actions. 
    This is (almost) always, the only part that must be customized in subclasses.
    """
    def __init__(self, parameters: Optional[dict] = None, seed: Optional[int] = None):
        self._parameters = load_parameters(parameters)
        self.actions: List[HighLevelAction] = [action(self._parameters) for action in self.ACTIONS]
        """ A list of instantiated high level actions. """
        self.REQUIRED_STATE_TRACKER = get_lowest_level_subclass(
            [action.REQUIRED_STATE_TRACKER for action in self.actions]
        )
        """ The required state tracker class inferred from the high level actions. """
        self.action_space = OneOf([action.get_action_space() for action in self.actions])
        """ The Gym action Space consisting of a choice over all high level action spaces. """
        self.unassign_emulator()
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed: Optional[int]= None):
        """
        Sets the random seed for the controller and its actions.
        Args:
            seed (int): The random seed to set.
        """
        self._rng = np.random.default_rng(seed)
        self.action_space.seed(seed)
        seed_value = seed
        for action in self.actions:
            if isinstance(seed, int):
                seed_value = seed + 1 # Simple way to get different seeds for each action
            else:
                seed_value = None
            action.seed(seed_value)

    def unassign_emulator(self):
        """
        Clears the reference to the emulator instance.
        """
        self._emulator = None
        for action in self.actions:
            action.unassign_emulator()

    def assign_emulator(self, emulator: Emulator):
        """
        Sets a reference to the emulator instance.
        Args:
            emulator (Emulator): The emulator instance to be tracked.
        """
        for action in self.actions:
            action.assign_emulator(emulator)
        self._emulator = emulator

    def get_action_space(self) -> OneOf:
        """
        Getter for the controller's Gym action space.
        Returns:
            OneOf: The Gym action Space consisting of a choice over all high level action spaces.
        """
        return self.action_space
    
    def sample(self) -> OneOf:
        """
        Samples a random action from the controller's action space.
        Returns:
            OneOf: A random action from the controller's action space.
        """
        return self.action_space.sample()
    
    def _space_action_to_high_level_action(self, space_action: OneOf) -> Tuple[HighLevelAction, Dict[str, Any]]:
        """
        Interprets a Gym space action into a high level action and its parameters.

        Args:
            space_action (OneOf): The action in the controller's action space.

        Returns: 
            Tuple[HighLevelAction, Dict[str, Any]]: The high level action and its parameters.
        """
        action_index, space_action = space_action
        action = self.actions[action_index]
        parameters = action.space_to_parameters(space_action)
        return action, parameters

    def _high_level_action_to_space_action(self, action: HighLevelAction, **kwargs) -> Space:
        """
        Converts a high level action and its parameters into a Gym Space action.

        Args:
            action (HighLevelAction): The high level action to convert.
            **kwargs: Additional arguments required for the specific high level action.
        Returns:
            Space: The action in the controller's action space.
        """
        space_action = action.parameters_to_space(**kwargs)
        action_index = self.actions.index(action)
        return (action_index, space_action)
    
    def _emulator_running(self) -> bool:
        """
        Checks if the emulator is currently running.

        Returns:
            bool: True if the emulator is running, False otherwise.
        """
        if self._emulator is None:
            log_error("Emulator reference not assigned to controller.", self._parameters)
        return not self._emulator.check_if_done()    
    
    def get_valid_high_level_actions(self) -> Dict[HighLevelAction, List[Dict[str, Any]]]:
        """
        Returns a list of all valid high level actions (including valid parameter inputs) that can be performed in the current state.

        Will fail if there are high level actions with infinite valid parameterizations.
        Use get_possibly_valid_high_level_actions() instead if that is the case.
        """
        valid_actions = {}
        if not self._emulator_running():
            return valid_actions
        for action in self.actions:
            valid_parameters = action.get_all_valid_parameters()
            if len(valid_parameters) > 0:
                valid_actions[action] = valid_parameters
        return valid_actions
        
    def get_valid_space_actions(self) -> Dict[HighLevelAction, Space]:
        """
        Returns a list of valid actions in the controller's action space that can be performed in the current state.

        Returns:

            Dict[HighLevelAction, Space]: A dictionary mapping high level actions to their corresponding valid space actions.
        """
        valid_space_actions = {}
        if not self._emulator_running():
            return valid_space_actions
        valid_high_level_actions = self.get_valid_high_level_actions()
        for action, parameter_list in valid_high_level_actions.items():
            for parameters in parameter_list:
                space_action = self._high_level_action_to_space_action(action, **parameters)
                valid_space_actions[action] = space_action
        return valid_space_actions

    def get_possibly_valid_high_level_actions(self) -> List[HighLevelAction]:
        """
        Returns a list of valid high level actions that can be performed (with some parameterized input) in the current state.

        Returns:
            List[HighLevelAction]: A list of valid high level actions.
        """
        if not self._emulator_running():
            return []
        actions = []
        for action in self.actions:
            if action.is_valid():
                actions.append(action)
        return actions
    
    def execute_space_action(self, action: OneOf) -> Tuple[Optional[List[Dict[str, Dict[str, Any]]]], Optional[bool]]:
        """
        Executes the specified high level action on the emulator after checking for validity.

        Args:
            action (OneOf): The action in the controller's action space.

        Returns:

            List[Dict[str, Dict[str, Any]]]: A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.

            bool: Whether the action was successful or not. (Is often an estimate.)
        """
        action_index, space_action = action
        executing_action = self.actions[action_index]
        return executing_action.execute_space_action(space_action)
    
    def execute(self, action: Type[HighLevelAction], **kwargs) -> Tuple[Optional[List[Dict[str, Dict[str, Any]]]], Optional[bool]]:
        """
        Executes the specified high level action on the emulator after checking for validity.

        Args:
            action (HighLevelAction): The high level action class to execute.
            **kwargs: Additional arguments required for the specific high level action.

        Returns:

            List[Dict[str, Dict[str, Any]]]: A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.

            bool: Whether the action was successful or not. (Is often an estimate.)
        """
        if action not in self.ACTIONS:
            log_error("Action not recognized by controller. Are you passing in an instance of the action class?", self._parameters)
        # Find the action instance
        action_index = self.ACTIONS.index(action)
        executing_action = self.actions[action_index]
        return executing_action.execute(**kwargs)


class LowLevelController(Controller):
    """ A controller that executes low level actions directly on the emulator. """
    ACTIONS = [LowLevelAction]
    """ A HighLevelAction subclass that directly maps to low level actions. """


class LowLevelPlayController(Controller):
    """ A controller that executes low level actions directly, but no menu button presses. """
    ACTIONS = [LowLevelPlayAction]
    """ A HighLevelAction subclass that directly maps to low level actions, but no menu button presses. """


class RandomPlayController(Controller):
    """ A controller that performs random play on the emulator using low level actions. """
    ACTIONS = [RandomPlayAction]
    """ A HighLevelAction subclass that performs random low level actions. """