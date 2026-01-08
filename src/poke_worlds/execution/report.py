from poke_worlds.utils import verify_parameters, log_error
from poke_worlds.emulation import StateTracker
from poke_worlds.interface import HighLevelAction, Environment, History


import numpy as np
from copy import deepcopy


from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class ExecutionReport(ABC):
    """ Holds the report of an execution run. 
    """
    REQUIRED_STATE_TRACKER = StateTracker
    """ The required state tracker class for this execution report (needed to guarantee safety of state_info_to_str). """

    def __init__(self, *, environment: Environment, high_level_goal: str, immediate_task: str, initial_plan: str, visual_context: str, exit_conditions: List[str], parameters: dict):
        verify_parameters(parameters)
        self._parameters = parameters
        if environment is not None and not issubclass(type(environment._emulator.state_tracker), self.REQUIRED_STATE_TRACKER):
            log_error(f"Environment's state tracker {type(environment._emulator.state_tracker)} is not compatible with required {self.REQUIRED_STATE_TRACKER} for this ExecutionReport.", parameters)
        self._environment = environment
        if environment is None:
            self._history_starting_index = None
        else:
            self._history_starting_index = len(environment._history) - 1
        self._history: History = None
        """ The history object from the environment at the start of the execution. Is only set when close() is called. Use get_history to access safely. """
        self.high_level_goal = high_level_goal
        """ The overall high level goal of the execution. """
        self.immediate_task = immediate_task
        """ The immediate task the execution was trying to accomplish. """
        self.exit_conditions = exit_conditions
        """ The exit conditions provided for the execution. """
        self.steps_taken = 0
        """ Number of steps taken in the execution. """
        self.step_contexts: List[Tuple[str, str]] = [(None, visual_context)]
        """ List of tuples containing (difference from previous frame, visual context) at each step. """
        self.plans: List[str] = [initial_plan]
        """ List of plans at each step of the execution. """
        self.exit_reasoning: str = None
        self._action_strings: List[str] = []
        """ List of action strings used during the execution. """
        self._action_messages: List[str] = []
        """ List of action messages received during the execution. """
        self.environment_done: bool = None
        """ Whether or not the environment is terminated / truncated. """

    def __deepcopy__(self, memo):
        if self in memo:
            return memo[self]
        freshReport = type(self)(
            environment=None,
            high_level_goal=self.high_level_goal,
            immediate_task=self.immediate_task,
            initial_plan=self.plans[0],
            visual_context=self.step_contexts[0][1],
            exit_conditions=self.exit_conditions,
            parameters=self._parameters
        )
        memo[self] = freshReport
        freshReport.steps_taken = deepcopy(self.steps_taken, memo)
        freshReport.step_contexts = deepcopy(self.step_contexts, memo)
        freshReport.plans = deepcopy(self.plans, memo)
        freshReport.exit_reasoning = deepcopy(self.exit_reasoning, memo)
        freshReport._action_strings = deepcopy(self._action_strings, memo)
        freshReport._action_messages = deepcopy(self._action_messages, memo)
        freshReport._history = deepcopy(self.get_history(), memo)
        freshReport._history_starting_index = self._history_starting_index
        return freshReport


    def _add_step(self, *, action_string: str, action_messages: str, frame_difference: str, visual_context: str, plan: str):
        """ Adds a step to the execution report. """
        self.steps_taken += 1
        self._action_strings.append(action_string)
        self._action_messages.append(action_messages)
        self.step_contexts.append((frame_difference, visual_context))
        self.plans.append(plan)

    def get_history(self) -> History:
        if self._history is None:
            history = self._environment._history[self._history_starting_index: ]
        else:
            history = deepcopy(self._history)
        return history
 
    def get_observations(self) -> List[Any]:
        """ Returns the list of observation dicts received during the execution. """
        history = self.get_history()
        return history.observations
    
    def get_state_infos(self) -> List[Dict[str, Dict[str, Any]]]:
        """ Returns the list of state info dicts received during the execution. """
        history = self.get_history()
        return history.infos
    
    def get_step_frames(self) -> List[np.ndarray]:
        """ Returns the list of screen frames captured at each step of the execution. """
        history = self.get_history()
        return history.get_step_frames()
    
    def get_transition_frames(self) -> List[np.ndarray]:
        """ Returns the list of transition frames captured between each action execution. """
        history = self.get_history()
        return history.get_transition_frames()
    
    def get_actions_taken(self) -> List[Tuple[str, HighLevelAction, Dict[str, Any], Dict[str, Dict[str, Any]], int, Dict[str, Any], str]]:
        """
        Docstring for get_actions_taken
        
        :param self: Description
        :return: List of actions details taken during the execution. Each entry is a tuple of (action_string, action_class, action_kwargs, transition_states, success_code, action_return_info, action_message).
        :rtype: List[Tuple[str, Any, Dict[str, Any], int, Any, str]]
        """
        history = self.get_history()
        action_details = history.get_action_details()
        use_action_details = []
        for i, action_detail in enumerate(action_details):
            action_class, action_kwargs, transition_states, success_code, action_return_info = action_detail
            action_string = self._action_strings[i]
            action_message = self._action_messages[i]
            use_action_details.append((action_string, action_class, action_kwargs, transition_states, success_code, action_return_info, action_message))
        return use_action_details
    
    def _close(self, exit_reasoning: str, environment_done: bool):
        """ Closes the execution report with the given exit reasoning. """
        self.exit_reasoning = exit_reasoning
        self.environment_done = environment_done
        if self._history is not None:
            log_error("ExecutionReport is already closed.", self._parameters)
        self._history = self.get_history()

    def get_state_info_strings(self) -> List[str]:
        """ Returns the list of state info strings for all state infos in the report. """
        return [self.state_info_to_str(state_info) for state_info in self.get_state_infos()]
        
    @abstractmethod
    def state_info_to_str(self, state_info: dict) -> str:
        """ Converts a state info to a string representation. Useful for VLM Prompting """
        pass

    def get_execution_summary(self) -> List[str]:
        """
        Returns a list describing each step taken during the execution and a final line describing the exit reasoning.
        
        :return: List of strings summarizing each step of the execution.
        :rtype: List[str]
        """
        summary_lines = []
        actions_taken = self.get_actions_taken()
        state_infos = self.get_state_infos()
        for i in range(self.steps_taken):
            action_string, action_class, action_kwargs, transition_states, success_code, action_return_info, action_message = actions_taken[i]
            frame_difference, visual_context = self.step_contexts[i + 1]
            state_info = state_infos[i + 1]
            plan = self.plans[i + 1]
            summary_line = f"Step {i + 1}:\n"
            summary_line += f"Action Taken: {action_string}\n"
            summary_line += f"Action Message: {action_message}\n"
            summary_line += f"Change in Game Frame: {frame_difference}\n"
            summary_line += f"State Info: {self.state_info_to_str(state_info)}\n"
            summary_line += f"Visual Context: {visual_context}\n"
            summary_line += f"Plan at this step: {plan}\n"
            summary_lines.append(summary_line)
        summary_lines.append(f"Execution ended because: {self.exit_reasoning}")
        return summary_lines