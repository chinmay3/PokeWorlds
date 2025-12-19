from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from enum import Enum
from poke_worlds.utils import verify_parameters, log_info, log_warn, log_error
from poke_worlds.emulation.emulator import LowLevelActions, Emulator
from poke_worlds.emulation.tracker import StateTracker


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

    def __init__(self, parameters: dict):
        verify_parameters(parameters)
        self._parameters = parameters
        self._emulator = None

    def clear_emulator(self):
        """
        Clears the reference to the emulator instance.
        """
        self._emulator = None

    def set_emulator(self, emulator: Emulator):
        """
        Sets a reference to the emulator instance.
        Args:
            emulator (Emulator): The emulator instance to be tracked.
        """
        self._emulator = emulator

    def get_actions(self) -> Enum:
        """
        Getter for the HighLevelActions enum.
        """
        return self.HighLevelActions
    
    @abstractmethod
    def get_valid_actions(self) -> List[HighLevelActions]:
        """
        Returns a list of valid high level actions that can be performed in the current state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def execute_action(self, action: HighLevelActions):
        """
        Executes the specified high level action on the emulator.
        """
        raise NotImplementedError 