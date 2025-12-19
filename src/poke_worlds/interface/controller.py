from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional
from enum import Enum
from poke_worlds.utils import verify_parameters, log_info, log_warn, log_error, load_parameters
from poke_worlds.emulation.emulator import LowLevelActions, Emulator
from poke_worlds.emulation.tracker import StateTracker
import numpy as np


class Controller(ABC):
    """ Abstract base class for controllers interfacing with the emulator. """

    REQUIRED_TRACKER = StateTracker
    """ The state tracker that tracks the minimal state information required for the controller to function. """

    class HighLevelActions(Enum):
        """ Enum defining high level actions that can be performed by the controller. """
        WAIT = 0
        """ No operation. """
        RANDOM_ACTION = 1
        """ Perform a random LowLevelAction. """

    def __init__(self, parameters: Optional[dict] = None):
        self._parameters = load_parameters(parameters)
        self.unassign_emulator()

    def unassign_emulator(self):
        """
        Clears the reference to the emulator instance.
        """
        self._emulator = None
        self._state_tracker = None

    def assign_emulator(self, emulator: Emulator):
        """
        Sets a reference to the emulator instance.
        Args:
            emulator (Emulator): The emulator instance to be tracked.
        """
        self._emulator = emulator
        self._state_tracker = emulator.state_tracker
        if not issubclass(type(self._state_tracker), self.REQUIRED_TRACKER):
            log_error(f"Controller requires a StateTracker of type {self.REQUIRED_TRACKER.NAME}, but got {type(self._state_tracker).NAME}", self._parameters)

    def get_actions(self) -> Enum:
        """
        Getter for the HighLevelActions enum.
        """
        return self.HighLevelActions
    
    @abstractmethod
    def _get_valid_actions(self) -> List[HighLevelActions]:
        """
        Returns a list of valid high level actions that can be performed in the current state.

        Returns:
            List[HighLevelActions]: A list of valid high level actions.
        """
        raise NotImplementedError
    
    def get_valid_actions(self) -> List[HighLevelActions]:
        """
        Returns a list of valid high level actions that can be performed in the current state.

        Returns:
            List[HighLevelActions]: A list of valid high level actions.
        """
        if self._emulator is None:
            log_error("Emulator reference not assigned to controller.", self._parameters)
        if self._emulator.check_if_done():
            return []
        return self._get_valid_actions()
    
    @abstractmethod
    def _execute_action(self, action: HighLevelActions) -> Tuple[List[Dict[str, Dict[str, Any]]], bool]:
        """
        Executes the specified high level action on the emulator. 
        Does not check for validity or assignment of the emulator reference.


        Args:
            action (HighLevelActions): The high level action to execute.
        Returns:
            List[Dict[str, Dict[str, Any]]]: A list of state tracker reports after each low level action executed.
            bool: Whether the action was successful or not. (Is often an estimate.)

        """
        raise NotImplementedError
    
    def execute_action(self, action: HighLevelActions) -> Tuple[Optional[List[Dict[str, Dict[str, Any]]]], Optional[bool]]:
        """
        Executes the specified high level action on the emulator after checking for validity.

        Args:
            action (HighLevelActions): The high level action to execute.

        Returns:
            List[Dict[str, Dict[str, Any]]]: A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.

            bool: Whether the action was successful or not. (Is often an estimate.)
        """
        if action not in self.get_valid_actions():
            #log_warn(f"Attempted to execute invalid action {action}. Valid actions are: {self.get_valid_actions()}", self._parameters)
            return None, None
        return self._execute_action(action)

class RandomController(Controller):
    """ An example controller that performs random Low Level actions. """
    class HighLevelActions(Enum):
        """  Splits high level actions into two categories: movement and buttons """
        RANDOM_MOVEMENT = 0
        """ Perform a random movement action. """
        RANDOM_BUTTON_PRESS = 1
        """ Perform a random button press action. """
        RANDOM_START_PRESS = 2
        """ Perform a random start/select button press action. """

    def _get_valid_actions(self) -> List[Controller.HighLevelActions]:
        return [self.HighLevelActions.RANDOM_MOVEMENT, self.HighLevelActions.RANDOM_BUTTON_PRESS] # Start is never valid
    
    def _execute_action(self, action: Controller.HighLevelActions) -> Tuple[List[Dict[str, Dict[str, Any]]], bool]:
        all_reports = []
        success = False
        low_level_choices = []
        if action == self.HighLevelActions.RANDOM_MOVEMENT:
            low_level_choices = [
                LowLevelActions.PRESS_ARROW_UP,
                LowLevelActions.PRESS_ARROW_DOWN,
                LowLevelActions.PRESS_ARROW_LEFT,
                LowLevelActions.PRESS_ARROW_RIGHT
            ]
        else:
            low_level_choices = [LowLevelActions.PRESS_BUTTON_A]
        chosen_action = np.random.choice(low_level_choices)
        _, done = self._emulator.step(chosen_action)
        report = self._state_tracker.report()
        all_reports.append(report)
        success = True
        return all_reports, success
