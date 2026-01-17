# Meant to store vlm utilities for the lowest level VLM agent. 
from copy import deepcopy

from poke_worlds.interface import Controller
from poke_worlds.execution.report import ExecutionReport
from poke_worlds.execution.executor_action import ExecutorAction
from poke_worlds.execution.vlm import ExecutorVLM, ocr
from poke_worlds.utils import load_parameters, log_error, log_warn, log_info
from poke_worlds.interface import Environment, HighLevelAction
from typing import List, Tuple, Type, Dict, Any
from abc import ABC, abstractmethod

from tqdm import tqdm

import numpy as np

class Executor(ABC):
    """
    A VLM Agent that can execute high level actions in an environment
    Subclasses of should create prompt engineered workflows that follow the general structure outlined below.   

    Each step of execution involves a couple of stages. 
    1. Action Prediction: Some process by which the executor decides on the next high level action to take and executes it. 
    2. Exit Condition Check: Some process by which the executor decides whether to stop execution.

    The Executor loops through these stages until either an exit condition is met, the environment signals done, or the step limit is reached.
    """

    REQUIRED_CONTROLLER = Controller
    """ The required controller class for this executor (needed to guarantee safety of get_action_message). """

    REQUIRED_ENVIRONMENT = Environment
    """ The required environment class for this executor. """

    EXECUTOR_ACTIONS: ExecutorAction = []
    """ The list of available ExecutorAction types for this executor. """

    def __init__(self, *, game: str, environment: Environment, execution_report_class: Type[ExecutionReport], report_init_kwargs: dict = None, action_buffer_size: int = None, seed: int = None, parameters: dict=None):
        self._parameters = load_parameters(parameters)
        if not issubclass(type(environment), self.REQUIRED_ENVIRONMENT):
            log_error(f"Provided environment {type(environment)} is not compatible with required {self.REQUIRED_ENVIRONMENT} for this Executor.", self._parameters)
        if not issubclass(type(environment._controller), self.REQUIRED_CONTROLLER):
            log_error(f"Environment's controller {type(environment._controller)} is not compatible with required {self.REQUIRED_CONTROLLER} for this Executor.", self._parameters)
        self._action_buffer_size = action_buffer_size if action_buffer_size is not None else self._parameters["executor_action_buffer_size"]
        self._max_retries_per_action = self._parameters["executor_retries_per_action"]
        self._game = game
        self._environment = environment
        if not issubclass(execution_report_class, ExecutionReport):
            log_error(f"Provided execution_report_class is not a subclass of ExecutionReport", self._parameters)
        report_init_kwargs["parameters"] = self._parameters
        self._execution_report = execution_report_class(environment=environment, **report_init_kwargs)
        self._vlm = ExecutorVLM # Should not be an instance, but the class itself.
        self._vlm.start()
        self.actions: Dict[Type[ExecutorAction], ExecutorAction] = {action_class: action_class(parameters=self._parameters, seed=seed) for action_class in self.EXECUTOR_ACTIONS}
        """ Instances of the available ExecutorAction types for this executor. Can be access with the class as the key."""
        for action_class, action in self.actions.items():
            action.assign_emulator(environment._emulator)
            
    def get_execution_report(self) -> ExecutionReport:
        """ 
        Returns the execution report for this executor. 
        
        :return: The execution report.
        :rtype: ExecutionReport
        """
        return deepcopy(self._execution_report)
    
    def _execute_next_action(self) -> Tuple[str, str, Tuple[str, Type[HighLevelAction], dict, int, dict, str], dict, bool, bool]:
        n_tries = 0
        previous_invalid_action_strings = []
        while n_tries < self._max_retries_per_action:
            action_str = self._decide_next_action_str(prev_action_strings=previous_invalid_action_strings)
            n_tries += 1
            action, action_kwargs = self._environment.string_to_high_level_action(action_str)
            if action is None: # then invalid action
                previous_invalid_action_strings.append(action_str)
                continue
            possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info = self._environment.step_str(action_str)
            if possible_obs is not None:
                observation, reward, terminated, truncated, info = possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info
                action, action_kwargs, transition_states, action_success, action_return = info["core"]["previous_action_details"]
                environment_done = terminated or truncated
                return (action_str, action, action_kwargs, transition_states, action_success, action_return), info, environment_done, False
            else: # action was a valid option, but invalid in the current state. 
                previous_invalid_action_strings.append(action_str)
                continue
        # If we reach here, then we have exceeded max retries
        return None, None, False, True

    def run_executor_action(self, action_str) -> Tuple[dict, int, str]:
        """
        Infers and runs and executor action.

        :param action_str: The string representation of the executor action to run.
        :type action_str: str
        :return: A tuple containing the return information from the action, its success code and the action message.
        :rtype: Tuple[dict, int, str]
        """
        action_class, action_kwargs = self._string_to_executor_action(action_str)
        if action_class is None:
            return None, None, None
        action_instance = self.actions[action_class]
        action_return_info, success_code = action_instance.execute(**action_kwargs)
        action_message = None
        if action_return_info is not None:
            action_message = self.get_action_message(action=action_class, action_kwargs=action_kwargs, action_success=success_code, action_return=action_return_info, last_action_hint=True)
            self._execution_report._add_executor_action(executor_action_str=action_str, executor_action=action_class, action_kwargs=action_kwargs, action_return=action_return_info, action_success_code=success_code, action_message=action_message)
        return action_return_info, success_code, action_message

    @abstractmethod
    def _string_to_executor_action(self, action_str: str) -> Tuple[Type[ExecutorAction], dict]:
        """
        Converts an action string to the corresponding ExecutorAction class and its execution arguments.
        Essentially does Controller.string_to_high_level_action but for ExecutorActions.

        :param action_str: The string representation of the action.
        :type action_str: str
        :return: A tuple containing the ExecutorAction class and its execution arguments.
        :rtype: Tuple[Type[ExecutorAction], dict]
        """
        pass

    @abstractmethod
    def _decide_next_action_str(self, prev_action_strings: List[str] = []) -> str:
        """
        Decides the next high level action to take as a string.

        :param prev_action_strings: A list of invalid action strings were attempted this turn. 
        :return: The string representation of the next high level action.
        :rtype: str
        """
        pass

    @abstractmethod
    def _decide_exit(self) -> bool:
        """
        Decides whether to exit execution.

        :return: True if execution should exit, False otherwise.
        :rtype: bool
        """
        pass

    @abstractmethod
    def _after_action(self, action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str]):
        """
        Hook to perform any necessary updates or processing after an action has been executed.

        :param action_details: Details of the action that was executed. Is in the form:

            -  `action_string` (`str`): The string representation of the action taken.
            -  `action_class` (`Type[HighLevelAction]`): The class of the high level action taken.
            -  `action_kwargs` (`dict`): The execution arguments used for the action.
            -  `transition_states` (`dict`): The state information before and after the action.
            -  `success_code` (`int`): Success code of the action.
            -  `action_return_info` (`dict`): Return information from the action.
            -  `action_message` (`str`): The message describing the action taken and its status.

        :type action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str]        
        """
        pass

    @abstractmethod
    def _get_update_kwargs(self, action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str]) -> dict:
        """
        Returns additional keyword arguments for updating internal state after taking an action and receiving an observation.
        Should match the arguments expected by `_update_additional`.

        :param action_details: Details of the action that was executed. Is in the form:

            -  `action_string` (`str`): The string representation of the action taken.
            -  `action_class` (`Type[HighLevelAction]`): The class of the high level action taken.
            -  `action_kwargs` (`dict`): The execution arguments used for the action.
            -  `transition_states` (`dict`): The state information before and after the action.
            -  `success_code` (`int`): Success code of the action.
            -  `action_return_info` (`dict`): Return information from the action.
            -  `action_message` (`str`): The message describing the action taken and its status.

        :type action_details: Tuple[str, Type[HighLevelAction], dict, dict, int, dict, str]        
        :return: Additional keyword arguments for updating report. Should match those expected by `self._execution_report._add_step_additional`.
        :rtype: dict
        """
        pass

    @abstractmethod
    def _get_exit_kwargs(self) -> dict:
        """
        Returns additional keyword arguments for closing out the execution report. 

        :return: Additional keyword arguments for closing out the execution report. Should match those expected by `self._execution_report._close_additional`.
        :rtype: dict
        """
        pass

    
    def execute(self, step_limit: int, show_progress: bool = True) -> ExecutionReport:
        """
        Executes the immediate task within the environment up to the step limit.
        
        :param step_limit: The maximum number of steps to execute.
        :type step_limit: int
        :param show_progress: Whether to display a progress bar during execution.
        :type show_progress: bool
        :return: The execution report.
        :rtype: ExecutionReport
        """
        n_steps = -1
        environment_done = False
        error_out = False
        if show_progress:
            pbar = tqdm(total=step_limit, desc=f"Executing")
        while True:
            n_steps += 1
            if show_progress:
                pbar.update(1)
            if error_out:
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(exit_code=-1, **exit_kwargs)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            if environment_done:
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(exit_code=1, **exit_kwargs)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            if n_steps >= step_limit:
                # need to close out
                self._execution_report._close("Reached step limit.", environment_done=environment_done)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            action_details, _ , environment_done, error_out = self._execute_next_action()
            if error_out:
                continue
            next_action_str, next_high_level_action, next_high_level_action_kwargs, transition_states, action_success, action_return = action_details
            action_message = self.get_action_message(action=next_high_level_action, action_kwargs=next_high_level_action_kwargs, action_success=action_success, action_return=action_return, last_action_hint=True)
            self._after_action(action_details + (action_message,))
            update_kwargs = self._get_update_kwargs(action_details + (action_message,))
            self._execution_report._add_step(action_string=next_action_str, action_messages=action_message, **update_kwargs)
            exit_decision = self._decide_exit()
            if exit_decision:
                exit_kwargs = self._get_exit_kwargs()
                self._execution_report._close(exit_code=0, **exit_kwargs)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            else:
                continue
            
    @abstractmethod
    def get_action_message(self, *, action: Type[HighLevelAction], action_kwargs: dict, action_success: int, action_return: dict, last_action_hint: bool=False) -> str:
        """
        Returns a string message describing the action taken and its status. 

        Should work for both HighLevelAction and the available ExecutorAction types. 
        
        :param action: The high level action taken.
        :type action: Type[HighLevelAction]
        :param action_kwargs: The execution arguments used for the action.
        :type action_kwargs: dict
        :param action_success: Success code of the action
        :type action_success: int
        :param action_return: Return information from the action
        :type action_return: dict
        :param last_action_hint: Whether to format the message as a hint for the next action. If True, includes guidance for the next action. This can include a hint to guide the next action.
        :type last_action_hint: bool
        :return: The action message string.
        :rtype: str
        """
        pass


class SimpleExecutor(Executor, ABC):
    """
    A VLM Agent that can execute high level actions in an environment to accomplish an immediate task. 
    This is essentially an attempt to create a generic prompt engineered workflow that gives reasonable results across a variety of games/environments.     

    At the start, the executor is given:
    - `high_level_goal`: The overall high level goal that the one calling the executor wants to achieve, beyond just the immediate task the Executor will be asked to tackle. This is (hopefully) to help guide the Executor's decisions towards actions that not only accomplish the immediate task, but also contribute to the overall high level goal.
    - `immediate_task`: The specific task that the Executor is to meant to accomplish within the environment.
    - `initial_plan`: A string representing the initial plan to guide the Executor in accomplishing the immediate task.
    - `visual_context`: A description of the initial visual context of the environment.
    - `exit_conditions`: Conditions under which the Executor should exit early. The system prompt always includes achieving the immediate task, exceeding the immediate task towards the high level goal, or encountering unexpected situations that make the immediate task irrelevant or unachievable.

    Each step involves a couple of stages. 
    Action Prediction:

        Given: the current visual context, plan, past actions and observed changes, current state and allowed actions.
        Produces: A revised plan, reasoning for next action, and the next high level action to take.
    
    Change Description:
        Given: The previous visual context and screen, the action taken and its status message, and the new screen after taking the action.
        Produces: A brief description of the changes observed between the two screens, and a concise description of the new visual context.

    Exit Condition Check:
        Given: The current visual context, high level goal, immediate task, and past actions and observed changes.
        Produces: A reasoning and decision on whether any exit conditions have been met.

    The Executor loops through these stages until either an exit condition is met, the environment signals done, or the step limit is reached.
    """

    REQUIRED_CONTROLLER = Controller
    """ The required controller class for this executor (needed to guarantee safety of get_action_message). """

    _screen_description_prompt = """
You are playing [GAME], trying to achieve the immediate task of [IMMEDIATE_TASK] and are given two snapshots of the screen from the game. Picture 1 is the earlier snapshot with the context [PICTURE_1_VISUAL_CONTEXT], after which the following action was taken, with a status message for the action [ACTION_AND_STATUS]. 

Picture 2 is the later snapshot, taken after the action. This picture comes with the following additional context: [ADDITIONAL_CONTEXT].

Describe, very briefly, what has changed between Picture 1 and Picture 2. Focus on any changes in location, player position, text, menus, characters, or significant visual elements that indicate a change in the game state. Provide only an extremely short and concise summary of the differences. 
Then, create a concise description of the visual context of Picture 2, focusing on the most important elements that define the current game state.
Structure your response as follows:
Changes: A very brief summary of the most salient changes between Picture 1 and Picture 2. 
Visual Context: A concise description of the visual context of Picture 2.
    """

    _exit_condition_prompt = """
You are playing [GAME] and are given the current screen snapshot from the game with context [VISUAL_CONTEXT]. 
The overall goal in mind is [HIGH_LEVEL_GOAL], and the immediate task you were trying to accomplish with the previous action is [IMMEDIATE_TASK]. The previous actions you took, led to the following changes [ACTIONS_AND_CHANGES].

You have been instructed to stop playing if any of the following conditions are met: 
1. If you have achieved your immediate task. 
2. If you have exceeded the immediate task and made even further progress towards the high level goal. This includes completing sub-goals that contribute to the high level goal, that go beyond just the immediate task.
3. If something unexpected has happened that leads you to believe that your immediate task is no longer relevant towards the high level goal or is not achievable.
[EXIT_CONDITIONS]
Determine whether any of these exit conditions have been satisfied. Respond with the following format: 
Reasoning: An extremely brief explanation of why or why not any of the exit conditions have been met. If you are not able to judge it at all, then assume they have not been met.
Decision: YES (if any exit conditions have been met), NO (otherwise). [STOP]
Response:
    """
    
    _first_execution_prompt = """
You are playing [GAME]. The overall goal in mind is [HIGH_LEVEL_GOAL], and the immediate task you are trying to accomplish is [IMMEDIATE_TASK].
You have the following plan to accomplish the immediate task: [PLAN].
You are given the Picture and context: [VISUAL_CONTEXT].
The allowed actions you can take are: [HIGH_LEVEL_ACTIONS]. Under no circumstances should you attempt to take any action that is not in this list and you must always follow the specified format. 
Based on all of this information, revise your plan and then decide on the best next high level action to take to accomplish the immediate task. Provide a very brief reasoning for why this action is the best choice given the current situation and then state the action to take in the correct syntax.
Structure your response as follows:
Revised Plan: An extremely short and brief revised plan to accomplish the immediate task. The plan may need significant revision, or could stay the same. Use the information provided to judge this. 
Next Action Reasoning: A very brief explanation of why the chosen action is the best next step.
Action: The high level action to take, in the correct format as given in the allowed actions list.
[SYSTEM]
Response:
    """
    
    _execution_prompt = """
You are playing [GAME]. The overall goal in mind is [HIGH_LEVEL_GOAL], and the immediate task you are trying to accomplish is [IMMEDIATE_TASK].
You have the following plan to accomplish the immediate task: [PLAN].
You took the following actions and observed the following changes: [ACTIONS_AND_CHANGES].
Finally, your thoughts about the next action to take were: [NEXT_ACTION_THOUGHTS].
You then took the action (with status message) : [LAST_ACTION].
This left you with the following screen capture and context: [VISUAL_CONTEXT].
The allowed actions you can take are: [HIGH_LEVEL_ACTIONS]. Under no circumstances should you attempt to take any action that is not in this list and you must always follow the specified format. 
Based on all of this information, revise your plan and then decide on the best next high level action to take to accomplish the immediate task. Provide a very brief reasoning for why this action is the best choice given the current situation and then state the action to take in the correct syntax.
Structure your response as follows:
Revised Plan: An extremely short and brief revised plan to accomplish the immediate task. The plan may need significant revision, or could stay the same. Use the information provided to judge this. 
Next Action Reasoning: A very brief explanation of why the chosen action is the best next step.
Action: The high level action to take, in the correct format as given in the allowed actions list.
[SYSTEM]
Response:
    """

    _default_str = "Unable to parse. You'll have to figure it out yourself."

    def __init__(self, *, game: str, environment: Environment, execution_report_class: Type[ExecutionReport], high_level_goal: str, task: str, initial_plan: str, visual_context: str, exit_conditions: List[str] = [], action_buffer_size: int = None, seed: int = None, parameters: dict=None):
        self._parameters = load_parameters(parameters)
        if not issubclass(execution_report_class, ExecutionReport):
            log_error(f"Provided execution_report_class is not a subclass of ExecutionReport", self._parameters)
        self._game = game
        self._high_level_goal = high_level_goal
        self._task = task
        self._initial_plan = initial_plan
        self._plan = initial_plan
        self._next_action_thoughts = None
        self._visual_context = visual_context
        self._exit_conditions = exit_conditions
        state_info = environment.get_info()
        self._previous_screen = state_info["core"]["current_frame"]
        self._screen_description_prompt = self._screen_description_prompt.replace("[GAME]", self._game).replace("[IMMEDIATE_TASK]", self._task)
        self._exit_condition_prompt = self._exit_condition_prompt.replace("[GAME]", self._game).replace("[HIGH_LEVEL_GOAL]", self._high_level_goal).replace("[IMMEDIATE_TASK]", self._task)
        exit_condition_str = ""
        starting_index = 4
        for i, condition in enumerate(self._exit_conditions):
            exit_condition_str += f"{starting_index + i}. {condition}\n"
        self._exit_condition_prompt = self._exit_condition_prompt.replace("[EXIT_CONDITIONS]", exit_condition_str)
        self._execution_prompt = self._execution_prompt.replace("[GAME]", self._game).replace("[HIGH_LEVEL_GOAL]", self._high_level_goal).replace("[IMMEDIATE_TASK]", self._task)
        self._first_execution_prompt = self._first_execution_prompt.replace("[GAME]", self._game).replace("[HIGH_LEVEL_GOAL]", self._high_level_goal).replace("[IMMEDIATE_TASK]", self._task).replace("[VISUAL_CONTEXT]", self._visual_context)
        report_kwargs = {
            "high_level_goal": self._high_level_goal,
            "task": self._task,
            "initial_plan": self._initial_plan,
            "visual_context": self._visual_context,
            "exit_conditions": self._exit_conditions
        }
        self._most_recent_decision_reasoning = None
        super().__init__(game=game, environment=environment, execution_report_class=execution_report_class, report_init_kwargs=report_kwargs, action_buffer_size=action_buffer_size, seed=seed, parameters=parameters)
        
    
    def get_additional_context(self, state_info: dict) -> str:
        """ 
        Returns any additional context to provide when describing screen changes. 
        
        Args:
            state_info (dict): The state information from the environment after taking an action.
        Returns:
            str: The additional context string.
        """
        return ""
    
    def get_actions_and_changes(self, new_action_details: Tuple[str, Type[HighLevelAction], Dict[str, Any], Dict[str, Dict[str, Any]], int, dict, str] = None, last_action_hint: bool = False) -> List[str]:
        """
        Returns a string summarizing the past self.action_buffer_size actions taken and their resulting changes.

        Args:
            new_action_details: If provided, includes this action and its resulting change as the latest entry in the summary.
            last_action_hint: If True, formats the last action message as if it is for the previous action taken. This can include a hint to guide the next action.
        Returns:
            A list of strings summarizing the past actions and their resulting changes. Will be of length self.action_buffer_size
        """
        actions_and_changes = []
        max_actions = self._action_buffer_size - 1 if new_action_details is not None else self._action_buffer_size
        all_actions = self._execution_report.get_actions_taken()[-max_actions:]
        all_frame_differences = [item[0] for item in self._execution_report.step_contexts[1:][-max_actions:]] # Skip the first
        if len(all_actions) == 0:
            log_error(f"This shouldn't be happening: trying to get actions and changes but no actions taken yet.", self._parameters)
        if len(all_frame_differences) != len(all_actions):
            log_error(f"Mismatch in actions and frame differences lengths when getting actions and changes.", self._parameters)
        total_actions = len(all_actions) - 1
        for i, (action_str, action, action_kwargs, transition_states, action_success, action_return, action_message) in enumerate(all_actions):
            actions_and_changes.append(f"{total_actions - 1*(new_action_details is not None) - i} Actions ago: {action_str} | Status Message: {action_message} | Change in screen: {all_frame_differences[i]}\n")
        if new_action_details is not None:
            new_action_str, new_high_level_action, new_high_level_action_kwargs, new_actions_transition_states, new_action_success, new_action_return, _ = new_action_details
            new_action_message = self.get_action_message(new_high_level_action, new_high_level_action_kwargs, new_action_success, new_action_return, last_action_hint=last_action_hint)
            actions_and_changes.append(f"Previous Action: {new_action_str} | Status Message: {new_action_message}")
        return actions_and_changes
    
    def _describe_screen_change(self, *, next_frame: np.ndarray, action_details: Tuple[str, Type[HighLevelAction], dict, Dict[str, Dict[str, Any]], int, dict], additional_context: str) -> Tuple[str, str]:
        prompt = self._screen_description_prompt.replace("[PICTURE_1_VISUAL_CONTEXT]", self._visual_context).replace("[ADDITIONAL_CONTEXT]", additional_context)
        next_action_str, next_high_level_action, next_high_level_action_kwargs, transition_states, action_success, action_return = action_details
        action_message = self.get_action_message(action=next_high_level_action, action_kwargs=next_high_level_action_kwargs, action_success=action_success, action_return=action_return)
        action_str = f"Action Taken: {next_action_str} | Status Message: {action_message}"
        prompt = prompt.replace("[ACTION_AND_STATUS]", action_str)
        images = [self._previous_screen, next_frame]
        response = self._vlm.multi_infer(prompt, images, max_new_tokens=250)[0]
        if response.count("Visual Context:") != 1:
            log_warn(f"Unable to parse screen change description response: {response}", self._parameters)
            return self._default_str, self._default_str
        changes_part, visual_context_part = response.split("Visual Context:")
        changes_part = changes_part.replace("Changes:", "").strip()
        visual_context_part = visual_context_part.strip()
        return changes_part, visual_context_part
    
    def _after_action(self, action_details):
        self._plan = self._most_recent_plan
        self._next_action_thoughts = self._most_recent_next_action_thoughts
        state_info = self._environment.get_info()
        next_frame = state_info["core"]["current_frame"]
        additional_context = self.get_additional_context(state_info)
        changes_description, new_visual_context = self._describe_screen_change(next_frame=next_frame, action_details=action_details[:-1], additional_context=additional_context)
        self._previous_screen = next_frame
        action_str, action_class, action_kwargs, transition_states, success_code, action_return, action_message = action_details
        # get the ocr text from the last transition_state
        last_state = transition_states[-1]
        #state_info["ocr"] = {"transition_ocr_regions": all_ocr_regions}
        all_ocr_regions = last_state["ocr"]["transition_ocr_regions"]
        ocr_texts = {}
        for transition_state_ocr_regions in all_ocr_regions:
            for region_name, region_captures in transition_state_ocr_regions.items():
                if region_name not in ocr_texts:
                    ocr_texts[region_name] = []
                # region_captures is a stack of numpy arrays
                ocr_text = ocr(region_captures)
                ocr_text = "\n".join(ocr_text)
                ocr_texts[region_name].append(ocr_text)
        if len(ocr_texts) > 0:
            changes_description += "\nOCR Texts from relevant regions:\n"
            for key, texts in ocr_texts.items():
                ocr_texts[key] = "\n".join(texts)
                changes_description += f"{key}:\n{ocr_texts[key]}\n"
        self._changes_description = changes_description
        if "ocr_regions" in state_info["ocr"]:
            ocr_text = ""
            for region_name, region_captures in state_info["ocr"]["ocr_regions"].items():
                ocr_result = ocr(region_captures)
                ocr_result = "\n".join(ocr_result)
                ocr_text += f"{region_name}:\n{ocr_result}\n"
            if ocr_text != "":
                new_visual_context += f"\nOCR Texts from relevant regions:\n{ocr_text}"
        self._visual_context = new_visual_context


    def _get_update_kwargs(self, action_details):
        kwargs = {
            "frame_difference": self._changes_description,
            "visual_context": self._visual_context,
            "plan": self._most_recent_plan,
        }
        return kwargs
    
    def _decide_next_action_str(self, prev_action_strings = []):
        if self._execution_report.steps_taken == 0:
            prompt = self._first_execution_prompt
        else:
            prompt = self._execution_prompt
        prompt = prompt.replace("[PLAN]", self._plan)
        allowed_actions = self._environment.get_action_strings()
        allowed_actions_str = "Allowed Actions:\n"
        for action_class, action_desc in allowed_actions.items():
            allowed_actions_str += f"- {action_desc}\n"
        prompt = prompt.replace("[HIGH_LEVEL_ACTIONS]", allowed_actions_str)
        prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        if self._execution_report.steps_taken > 0:
            actions_and_changes = self.get_actions_and_changes(last_action_hint=True)
            actions_and_changes_str = "\n".join(actions_and_changes)
            prompt = prompt.replace("[ACTIONS_AND_CHANGES]", actions_and_changes_str)
            prompt = prompt.replace("[NEXT_ACTION_THOUGHTS]", self._next_action_thoughts) # should be set already
            last_action_str, last_high_level_action, last_high_level_action_kwargs, last_action_transition_states, last_action_success, last_action_return, _ = self._execution_report.get_actions_taken()[-1]
            last_action_message = self.get_action_message(action=last_high_level_action, action_kwargs=last_high_level_action_kwargs, action_success=last_action_success, action_return=last_action_return, last_action_hint=True)
            last_action_full_str = f"Action Taken: {last_action_str} | Status Message: {last_action_message}"
            prompt = prompt.replace("[LAST_ACTION]", last_action_full_str)
        # Now need to parse the response into plan, action, reasoning
        action_str = None
        max_internal_retries = self._max_retries_per_action
        n_retries = 0
        system_prompt = ""
        if len(prev_action_strings) > 0:
            system_prompt += f"\n[IMPORTANT SYSTEM MESSAGE] The following previous action strings were invalid or could not be executed in the current state: {prev_action_strings}. Remember to only choose from the allowed actions list and use input parameters that fit with the current context and format. \nAllowed Actions: {allowed_actions_str}"
        while n_retries < max_internal_retries:
            n_retries += 1
            final_prompt = prompt.replace("[SYSTEM]", system_prompt)
            response = self._vlm.infer(final_prompt, self._previous_screen, max_new_tokens=500)[0]
            if "Action:" not in response or "Next Action Reasoning:" not in response:
                system_prompt += "\n[IMPORTANT SYSTEM MESSAGE] Your previous response could not be parsed correctly, it did not contain Action: or Next Action Reasoning:. Remember to follow the specified format exactly. Try again. Make sure your output is not too long, so that it fits within the token limit."
                continue
            if response.count("Action:") != 1 or response.count("Next Action Reasoning:") != 1:
                system_prompt += "\n[IMPORTANT SYSTEM MESSAGE] Your previous response could not be parsed correctly, it contained multiple Action: or Next Action Reasoning: sections. Remember to follow the specified format exactly. Try again."
                continue
            plan_and_reasoning_part, action_part = response.split("Action:")
            plan_part, reasoning_part = plan_and_reasoning_part.split("Next Action Reasoning:")
            plan = plan_part.replace("Revised Plan:", "").strip()
            next_action_reasoning = reasoning_part.strip()
            action_str = action_part.strip()
            self._most_recent_plan = plan
            self._most_recent_next_action_thoughts = next_action_reasoning
            return action_str

    def _check_exit_conditions(self) -> bool:
        prompt = self._exit_condition_prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        actions_and_changes = self.get_actions_and_changes(last_action_hint=False)
        actions_and_changes = "\n".join(actions_and_changes)
        prompt = prompt.replace("[ACTIONS_AND_CHANGES]", actions_and_changes)
        response = self._vlm.infer(prompt, self._previous_screen, max_new_tokens=250)[0]
        if "Decision:" not in response:
            log_warn(f"Unable to parse exit condition response: {response}", self._parameters)
            return self._default_str, False
        reasoning_part, decision_part = response.split("Decision:")
        reasoning_part = reasoning_part.replace("Reasoning:", "").strip()
        decision_part = decision_part.strip().upper()
        self._most_recent_decision_reasoning = reasoning_part
        if "YES" in decision_part and "NO" not in decision_part:
            return True
        elif "NO" in decision_part and "YES" not in decision_part:
            return False
        else:
            log_warn(f"Unable to parse exit condition decision: {response}", self._parameters)
            return False
        
    def _get_exit_kwargs(self):
        return {"exit_reasoning": self._most_recent_decision_reasoning}
        
    def _decide_exit(self):
        return self._check_exit_conditions()
