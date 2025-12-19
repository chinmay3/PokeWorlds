from poke_worlds.emulation.parser import StateParser
from poke_worlds.utils import nested_dict_to_str, verify_parameters


import numpy as np
from typing import Optional

from abc import ABC, abstractmethod


class StateTracker(ABC):
    def __init__(self, name: str, session_name: str, instance_id: str, state_parser: StateParser, parameters: dict):
        """
        Initializes the StateTracker.
        Args:
            name (str): Name of the game.
            session_name (str): Name of the session.
            instance_id (str): Unique identifier for this environment instance.
            state_parser (StateParser): An instance of the StateParser to parse game state variables.
            parameters (dict): A dictionary of parameters for configuration.
        """
        verify_parameters(parameters)
        self.name = name
        """ Name of the game. """
        self.session_name = session_name
        """ Name of the session. """
        self.instance_id = instance_id
        """ Unique identifier for this environment instance. """
        self.state_parser = state_parser
        """ An instance of the StateParser to parse game state variables. """
        self._parameters = parameters
        self.metrics = {}
        self.reset()
        """ Dictionary to store tracked metrics. TODO: Review design. """

    def reset(self):
        """
        Is called once per environment reset.

        At this level, only manages the step counter
        """
        self.metrics["steps"] = 0
        self.metrics["previous_frame"] = None

    def step(self, recent_frames: Optional[np.ndarray], epsilon: float=0.01):
        """
        Is called once per environment step to update any tracked metrics.
        At this level, only manages the step counter and some other basics. 

        Args:
            recent_frames (Optional[np.ndarray]): The stack of frames that were rendered during the last action. Shape is [n_frames, height, width, channels]. Can be None if rendering is disabled.
            epsilon (float, optional): The threshold for considering a frame change.
        """
        self.metrics["steps"] =  self.metrics.get("steps", 0) + 1
        current_frame = None
        if recent_frames is not None:
            current_frame = recent_frames[-1]
        else:
            current_frame = self.state_parser.get_current_frame()
            self.metrics["previous_frame"] = current_frame
        self.metrics["frame_changed"] = np.abs(current_frame - self.metrics["previous_frame"]).mean() > epsilon
        self.metrics["previous_frame"] = current_frame


    @abstractmethod
    def close(self):
        """
        Is called once when the environment is closed to finalize any tracked metrics.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<StateTracker(name={self.name}, session_name={self.session_name}, instance_id={self.instance_id})>"

    def __str__(self) -> str:
        start = f"***\t{self.__repr__()}\t***"
        body = nested_dict_to_str(self.metrics, indent=1)
        return f"{start}\n{body}"