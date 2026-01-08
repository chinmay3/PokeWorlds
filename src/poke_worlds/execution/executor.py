# Meant to store vlm utilities for the lowest level VLM agent. 
from copy import deepcopy

from poke_worlds.interface import Controller
from poke_worlds.execution.report import ExecutionReport
from poke_worlds.execution.vlm import ExecutorVLM
from poke_worlds.utils.vlm import HuggingFaceVLM
from poke_worlds.utils import load_parameters, log_error, log_warn, log_info
from poke_worlds.interface import Environment, HighLevelAction
from typing import List, Tuple, Type, Dict, Any
from abc import ABC, abstractmethod

from tqdm import tqdm

import numpy as np

class Executor(ABC):
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
Your current state in the game is: [CURRENT_STATE].
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
Your current state in the game is: [CURRENT_STATE].
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

    def __init__(self, *, game: str, environment: Environment, execution_report_class: Type[ExecutionReport], high_level_goal: str, immediate_task: str, initial_plan: str, visual_context: str, exit_conditions: List[str] = [], action_buffer_size: int = None, parameters: dict=None):
        self._parameters = load_parameters(parameters)
        if not issubclass(type(environment._controller), self.REQUIRED_CONTROLLER):
            log_error(f"Environment's controller {type(environment._controller)} is not compatible with required {self.REQUIRED_CONTROLLER} for this Executor.", self._parameters)
        self._action_buffer_size = action_buffer_size if action_buffer_size is not None else self._parameters["executor_action_buffer_size"]
        self._max_retries_per_action = self._parameters["executor_retries_per_action"]
        self._game = game
        self._environment = environment
        self._high_level_goal = high_level_goal
        self._immediate_task = immediate_task
        self._initial_plan = initial_plan
        self._plan = initial_plan
        self._next_action_thoughts = None
        self._visual_context = visual_context
        self._exit_conditions = exit_conditions
        state_info = self._environment.get_info()
        self._previous_screen = state_info["core"]["current_frame"]
        self._screen_description_prompt = self._screen_description_prompt.replace("[GAME]", self._game).replace("[IMMEDIATE_TASK]", self._immediate_task)
        self._exit_condition_prompt = self._exit_condition_prompt.replace("[GAME]", self._game).replace("[HIGH_LEVEL_GOAL]", self._high_level_goal).replace("[IMMEDIATE_TASK]", self._immediate_task)
        exit_condition_str = ""
        starting_index = 4
        for i, condition in enumerate(self._exit_conditions):
            exit_condition_str += f"{starting_index + i}. {condition}\n"
        self._exit_condition_prompt = self._exit_condition_prompt.replace("[EXIT_CONDITIONS]", exit_condition_str)
        self._execution_prompt = self._execution_prompt.replace("[GAME]", self._game).replace("[HIGH_LEVEL_GOAL]", self._high_level_goal).replace("[IMMEDIATE_TASK]", self._immediate_task)
        self._first_execution_prompt = self._first_execution_prompt.replace("[GAME]", self._game).replace("[HIGH_LEVEL_GOAL]", self._high_level_goal).replace("[IMMEDIATE_TASK]", self._immediate_task).replace("[VISUAL_CONTEXT]", self._visual_context)
        if not issubclass(execution_report_class, ExecutionReport):
            log_error(f"Provided execution_report_class is not a subclass of ExecutionReport", self._parameters)
        self._execution_report = execution_report_class(environment=environment, high_level_goal=self._high_level_goal, immediate_task=self._immediate_task, initial_plan=self._initial_plan, visual_context=self._visual_context, exit_conditions=self._exit_conditions, parameters=self._parameters)
        self._vlm = ExecutorVLM # Should not be an instance, but the class itself.
        self._vlm.start()
    
    def _update_all(self, *, action_str: str, action_message: str, frame: np.ndarray, frame_difference: str, visual_context: str, plan: str, next_action_thoughts: str):
        """ Updates all internal state after taking an action and receiving an observation. """
        self._previous_screen = frame
        self._visual_context = visual_context
        self._plan = plan
        self._next_action_thoughts = next_action_thoughts
        self._execution_report._add_step(action_string=action_str, action_messages=action_message,
                                        frame_difference=frame_difference, visual_context=visual_context, 
                                        plan=plan)
        
    def get_execution_report(self) -> ExecutionReport:
        """ Returns the execution report for this executor. """
        return deepcopy(self._execution_report)
    
    def get_additional_context(self, state_info: dict) -> str:
        """ Returns any additional context to provide when describing screen changes. """
        return ""
    
    def get_actions_and_changes(self, new_action_details: Tuple[str, HighLevelAction, Dict[str, Any], Dict[str, Dict[str, Any]], int, dict, str] = None, last_action_hint: bool = False) -> List[str]:
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
            breakpoint()
            new_action_str, new_high_level_action, new_high_level_action_kwargs, new_actions_transition_states, new_action_success, new_action_return, _ = new_action_details
            new_action_message = self.get_action_message(new_high_level_action, new_high_level_action_kwargs, new_action_success, new_action_return, last_action_hint=last_action_hint)
            actions_and_changes.append(f"Previous Action: {new_action_str} | Status Message: {new_action_message}")
        return actions_and_changes
    
    def _describe_screen_change(self, *, next_frame: np.ndarray, action_details: Tuple[str, HighLevelAction, dict, Dict[str, Dict[str, Any]], int, dict], additional_context: str) -> Tuple[str, str]:
        prompt = self._screen_description_prompt.replace("[PICTURE_1_VISUAL_CONTEXT]", self._visual_context).replace("[ADDITIONAL_CONTEXT]", additional_context)
        next_action_str, next_high_level_action, next_high_level_action_kwargs, transition_states, action_success, action_return = action_details
        action_message = self.get_action_message(action=next_high_level_action, action_kwargs=next_high_level_action_kwargs, action_success=action_success, action_return=action_return)
        action_str = f"Action Taken: {next_action_str} | Status Message: {action_message}"
        prompt = prompt.replace("[ACTION_AND_STATUS]", action_str)
        images = [self._previous_screen, next_frame]
        response = self._vlm.multi_infer(prompt, images, max_new_tokens=250)
        if response.count("Visual Context:") != 1:
            log_warn(f"Unable to parse screen change description response: {response}", self._parameters)
            return self._default_str, self._default_str
        changes_part, visual_context_part = response.split("Visual Context:")
        changes_part = changes_part.replace("Changes:", "").strip()
        visual_context_part = visual_context_part.strip()
        return changes_part, visual_context_part
    
    def _execute_next_action(self) -> Tuple[str, str, Tuple[str, HighLevelAction, dict, int, dict, str], dict]:
        if self._execution_report.steps_taken == 0:
            prompt = self._first_execution_prompt
        else:
            prompt = self._execution_prompt
        prompt = prompt.replace("[PLAN]", self._plan)
        current_state_str = self._environment.get_agent_state()
        prompt = prompt.replace("[CURRENT_STATE]", str(current_state_str))
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
        action = None
        action_kwargs = None
        n_tries = 0
        system_prompt = ""
        while n_tries < self._max_retries_per_action:
            n_tries += 1
            final_prompt = prompt.replace("[SYSTEM]", system_prompt)
            response = self._vlm.infer(final_prompt, self._previous_screen, max_new_tokens=500)
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
            action, action_kwargs = self._environment.string_to_high_level_action(action_str)
            if action is None: # then invalid action
                system_prompt += f"\n[IMPORTANT SYSTEM MESSAGE] Your previous response contained an invalid action: {action_str}. Remember to only choose from the allowed actions list and follow the specified format exactly. \nAllowed Actions: {allowed_actions_str}"
                continue
            possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info = self._environment.step_str(action_str)
            if possible_obs is not None:
                observation, reward, terminated, truncated, info = possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info
                action, action_kwargs, transition_states, action_success, action_return = info["core"]["previous_action_details"]
                environment_done = terminated or truncated
                return plan, next_action_reasoning, (action_str, action, action_kwargs, transition_states, action_success, action_return), info, environment_done, False
            else: # action was a valid option, but invalid in the current state. 
                system_prompt += f"\n[IMPORTANT SYSTEM MESSAGE] Your previous response contained an action and parameter combination that could not be executed in the current state: {action_str}. Remember to only choose from the allowed actions list and use input parameters that fit with the current context and format. \nAllowed Actions: {allowed_actions_str}"
                continue
        # If we reach here, then we have exceeded max retries
        return None, None, None, None, False, True

    def _check_exit_conditions(self) -> Tuple[str, bool]:
        prompt = self._exit_condition_prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
        actions_and_changes = self.get_actions_and_changes(last_action_hint=False)
        actions_and_changes = "\n".join(actions_and_changes)
        prompt = prompt.replace("[ACTIONS_AND_CHANGES]", actions_and_changes)
        response = self._vlm.infer(prompt, self._previous_screen, max_new_tokens=250)
        if "Decision:" not in response:
            log_warn(f"Unable to parse exit condition response: {response}", self._parameters)
            return self._default_str, False
        reasoning_part, decision_part = response.split("Decision:")
        reasoning_part = reasoning_part.replace("Reasoning:", "").strip()
        decision_part = decision_part.strip().upper()
        if "YES" in decision_part and "NO" not in decision_part:
            return reasoning_part, True
        elif "NO" in decision_part and "YES" not in decision_part:
            return reasoning_part, False
        else:
            log_warn(f"Unable to parse exit condition decision: {response}", self._parameters)
            return self._default_str, False

    def execute(self, step_limit: int, show_progress: bool = True) -> ExecutionReport:
        n_steps = -1
        environment_done = False
        error_out = False
        if show_progress:
            pbar = tqdm(total=step_limit, desc=f"\nExecuting Task: {self._immediate_task}")
        while True:
            n_steps += 1
            if show_progress:
                pbar.update(1)
            if error_out:
                self._execution_report._close("VLM could not produce a valid action output.", environment_done=environment_done)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            if environment_done:
                self._execution_report._close("Environment signaled done.", environment_done=environment_done)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            if n_steps >= step_limit:
                # need to close out
                self._execution_report._close("Reached step limit.", environment_done=environment_done)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            revised_plan, next_action_thoughts, action_details, \
                next_state_info, environment_done, error_out = self._execute_next_action()
            if error_out:
                continue
            next_action_str, next_high_level_action, next_high_level_action_kwargs, transition_states, action_success, action_return = action_details
            current_frame = next_state_info["core"]["current_frame"]
            additional_context = self.get_additional_context(next_state_info)
            if additional_context == "":
                additional_context = "None"
            frame_difference, visual_context = self._describe_screen_change(next_frame=current_frame,
                                                                           action_details=action_details,
                                                                           additional_context=additional_context)
            action_message = self.get_action_message(action=next_high_level_action, action_kwargs=next_high_level_action_kwargs, action_success=action_success, action_return=action_return, last_action_hint=True)
            self._update_all(action_str=next_action_str,
                            action_message=action_message,
                            frame=current_frame,
                            frame_difference=frame_difference,
                            visual_context=visual_context,
                            plan=revised_plan, 
                            next_action_thoughts=next_action_thoughts)
            exit_reasoning, exit_decision = self._check_exit_conditions()
            if exit_decision:
                self._execution_report._close(exit_reasoning, environment_done=environment_done)
                if show_progress:
                    pbar.close()
                return self.get_execution_report()
            else:
                continue
            
    @abstractmethod
    def get_action_message(self, *, action: HighLevelAction, action_kwargs: dict, action_success: int, action_return: dict, last_action_hint: bool=False) -> str:
        """ Returns a string message describing the action taken and its status. 
        last_action_hint: If True, formats the message as if it is for the previous action taken. This can include a hint to guide the next action. 
        """
        pass
