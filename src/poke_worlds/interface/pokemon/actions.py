from poke_worlds.utils import log_error, perform_vlm_inference
from poke_worlds.interface.action import HighLevelAction
from poke_worlds.emulation.pokemon.parsers import AgentState, PokemonStateParser
from poke_worlds.emulation.pokemon.trackers import CorePokemonTracker
from poke_worlds.emulation import LowLevelActions
from abc import ABC
from typing import List, Tuple, Dict
import numpy as np

from gymnasium.spaces import Box, Discrete, Text
import matplotlib.pyplot as plt

HARD_MAX_STEPS = 20
""" The hard maximum number of steps we'll let agents take in a sequence """

def frame_changed(past: np.ndarray, preset: np.ndarray, epsilon=0.01):
    return np.abs(past -preset).mean() > epsilon 

def _plot(past: np.ndarray, current: np.ndarray):
    # plot both side by side for debug
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(past)
    axs[1].imshow(current)
    plt.show()


class PassDialogueAction(HighLevelAction):
    """
    This only triggers in the gaps 
    Is just a skip through dialogue action.
    """
    REQUIRED_STATE_PARSER = PokemonStateParser
    REQUIRED_STATE_TRACKER = CorePokemonTracker

    def is_valid(self, **kwargs):
        """
        Just checks if the agent is in dialogue state.
        """
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.IN_DIALOGUE
    
    def _execute(self):
        frames, done = self._emulator.step(LowLevelActions.PRESS_BUTTON_B)
        # if we are still in dialogue, its a fail:
        action_success = 0 if self._emulator.state_parser.get_agent_state(frames[-1]) != AgentState.IN_DIALOGUE else -1
        return [self._state_tracker.report()], action_success
    
    def get_action_space(self):
        return Discrete(1)
    
    def parameters_to_space(self):
        return 0
    
    def space_to_parameters(self, space_action):
        return {}


class InteractAction(HighLevelAction):
    """
    Handles interaction actions in the Pokemon environment.
    Currently only supports the "interact" action, which presses the A button.
    """

    REQUIRED_STATE_PARSER = PokemonStateParser    
    REQUIRED_STATE_TRACKER = CorePokemonTracker

    def is_valid(self, **kwargs):
        """
        Just checks if the agent is in free roam state.
        """
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM

    def _execute(self):
        current_frame = self._emulator.get_current_frame()
        frames, done = self._emulator.step(LowLevelActions.PRESS_BUTTON_A)
        action_success = 0
        # Check if the frames have changed. Be strict and require all to not permit jittering screens.
        prev_frames = []
        for frame in frames:
            if self._emulator.state_parser.get_agent_state(frame) != AgentState.FREE_ROAM: # something happened lol
                action_success = 1
                break
            for past_frame in prev_frames:
                if not frame_changed(past_frame, frame):
                    action_success = -1
                    break
            if action_success != 0:
                break
            prev_frames.append(frame)        
        if action_success == 0:
            action_success = -1 # I guess? For some reason the previous thing doesn't catch same frames
        return [self._state_tracker.report()], action_success # 0 means something maybe happened. 1 means def happened.
    
    def get_action_space(self):
        return Discrete(1)
    
    def parameters_to_space(self):
        return 0
    
    def space_to_parameters(self, space_action):
        return {}
    



class BaseMovementAction(HighLevelAction, ABC):
    """
    Base class for movement actions in the Pokemon environment.
    Has utility methods for moving in directions.
    """
    REQUIRED_STATE_TRACKER = CorePokemonTracker
    REQUIRED_STATE_PARSER = PokemonStateParser

    def move(self, direction: str, steps: int) -> Tuple[np.ndarray, int]:
        """
        Move in a given direction for a number of steps.
        Args:
            direction (str): One of "up", "down", "left", "right"
            steps (int): Number of steps to move in that direction
        Returns:
            Tuple[List[Dict], int]: A tuple containing:
                - A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.
                - An integer action success status:
                    0 -> finished all steps
                    1 -> took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter)
                    2 -> took some steps, but agent state changed from free roam
                   -1 -> frame didn't change, even on the first step
        """
        action_dict = {"right": LowLevelActions.PRESS_ARROW_RIGHT, 
                       "down": LowLevelActions.PRESS_ARROW_DOWN, 
                       "up": LowLevelActions.PRESS_ARROW_UP,
                       "left": LowLevelActions.PRESS_ARROW_LEFT}
        if direction not in action_dict.keys():
            log_error(f"Got invalid direction to move {direction}", self._parameters)
        action = action_dict[direction]
        # keep trying the action.
        # exit status 0 -> finished steps
        # 1 -> took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter)
        # 2 -> took some steps, but agent state changed from free roam
        # -1 -> frame didn't change, even on the first step 
        action_success = -1
        transition_state_dicts = []
        transition_frames = []
        previous_frame = self._emulator.get_current_frame() # Do NOT get the state tracker frame, as it may have a grid on it. 
        n_step = 0
        agent_state = AgentState.FREE_ROAM
        while n_step < steps and agent_state == AgentState.FREE_ROAM:
            frames, done = self._emulator.step(action)
            transition_state_dicts.append(self._state_tracker.report())
            transition_frames.extend(frames)
            if done:
                break
            # check if frames changed. If not, break out. 
            # We check all frames in sequence to try and catch oscillations. But nothing will catch 1 step into wall in areas like this
            if not all([frame_changed(previous_frame, current_frame) for current_frame in frames]):
                break
            agent_state = self._emulator.state_parser.get_agent_state(self._emulator.get_current_frame())
            if agent_state != AgentState.FREE_ROAM:
                break
            n_step += 1
            previous_frame = frames[-1]

        if agent_state != AgentState.FREE_ROAM:
            action_success = 2
        else:
            if n_step <= 0:
                action_success = -1
            elif n_step == steps:
                action_success = 0
            else:
                action_success = 1
        return transition_state_dicts, action_success 
    
    def chain_move(self, direction_steps: List[Tuple[str, int]]):
        """
        Chain together several move operations. Exit at the first action_success != 0
        Args:
            direction_steps (List[Tuple[str, int]]): A list of tuples, each containing a direction and number of steps to move in that direction.
        Returns:
            Tuple[List[Dict], int]: A tuple containing:
                - A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.
                - An integer action success status:
                    0 -> finished all steps
                    1 -> took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter)
                    2 -> took some steps, but agent state changed from free roam
                   -1 -> frame didn't change, even on the first step
        """
        all_transition_states = []
        for direction, steps in direction_steps:
            transition_states, action_success = self.move(direction=direction, steps=steps)
            all_transition_states.extend(transition_states)
            if action_success != 0:
                return all_transition_states, action_success
        return all_transition_states, 0

    def move_until_stop(self, direction, ind_step=5):
        """
        Move in a direction until action_success is no longer 0. 
        Args:
            direction (str): One of "up", "down", "left", "right"
            ind_step (int): Number of steps to move in each individual move call.
        Returns:
            Tuple[List[Dict], int]: A tuple containing:
                - A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.
                - An integer action success status:
                    0 -> finished all steps
                    1 -> took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter)
                    2 -> took some steps, but agent state changed from free roam
                   -1 -> frame didn't change, even on the first step
        """
        all_transition_states = []
        n_total_steps = 0
        while True:
            transition_states, action_success = self.move(direction=direction, steps=ind_step)
            all_transition_states.extend(transition_states)
            n_total_steps += ind_step
            if action_success != 0 or n_total_steps >= HARD_MAX_STEPS:
                return all_transition_states, action_success


    def is_valid(self, **kwargs):
        """
        Just checks if the agent is in free roam state.
        """
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM


class MoveStepsAction(BaseMovementAction):

    def get_action_space(self):
        """
        Returns a Box space representing movement in 2D.
        The first dimension represents vertical movement (positive is up, negative is down).
        The second dimension represents horizontal movement (positive is right, negative is left).

        Returns:
            Box: A Box space with shape (2,) and values ranging from -HARD_MAX_STEPS//2 to HARD_MAX_STEPS//2.

        """
        return Box(low=-HARD_MAX_STEPS//2, high=HARD_MAX_STEPS//2, shape=(2,), dtype=np.int8)

    def space_to_parameters(self, space_action):
        direction = None
        if space_action[0] > 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)
            direction = "up"
        if space_action[0] < 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)            
            direction = "down"
        if space_action[1] > 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)
            direction = "right"
        if space_action[1] < 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)
            direction = "left"
        if direction is None:
            log_error(f"Weird space vector: {space_action}", self._parameters)
        steps = abs(int(space_action.sum()))
        return {"direction": direction, "steps": steps}
        

    def parameters_to_space(self, direction: str, steps: int):
        move_vec = np.zeros(2) # x, y
        if direction == "up":
            move_vec[1] = 1
        elif direction == "right":
            move_vec[0] = 1
        elif direction == "down":
            move_vec[1] = -1
        elif direction == "left":
            move_vec[0] = -1
        else:
            log_error(f"Unrecognized direction {direction}", self._parameters)
        if steps <= 0:
            log_error(f"Bro. What you trying here. Don't step weirdly: {steps}", self._parameters)
        move_vec *= steps
        return move_vec


    def _execute(self, direction, steps):
        transition_states, status = self.move(direction=direction, steps=steps)
        return transition_states, status
    
    def is_valid(self, **kwargs):
        direction = kwargs.get("direction")
        step = kwargs.get("step")
        if direction is not None and direction not in ["up", "down", "left", "right"]:
            return False
        if step is not None:
            if not isinstance(step, str):
                return False
        return super().is_valid(**kwargs)
        

class MenuAction(HighLevelAction):
    """
    Handles opening and navigating the in-game start menu. Will also handle PC screens and dialogue choices. 
    Unfortunately, since PokemonCrystal based games has a bag with submenus, I need to add left right actions as well.
    Perhaps I can add a bag actions class later and separate these out. 
    """

    REQUIRED_STATE_PARSER = PokemonStateParser    
    REQUIRED_STATE_TRACKER = CorePokemonTracker

    _MENU_ACTION_MAP = {
            "up": LowLevelActions.PRESS_ARROW_UP,
            "down": LowLevelActions.PRESS_ARROW_DOWN,
            "confirm": LowLevelActions.PRESS_BUTTON_A,
            "left": LowLevelActions.PRESS_ARROW_LEFT,
            "right": LowLevelActions.PRESS_ARROW_RIGHT,
            "exit": LowLevelActions.PRESS_BUTTON_B,
            "open": LowLevelActions.PRESS_BUTTON_START
    }

    def is_valid(self, **kwargs):
        """
        Checks if the menu action is valid in the current state.

        Args:
            menu_action (str, optional): The menu action to check. One of "up", "down", "confirm", "exit", "open".
        Returns:
            bool: True if the action is valid, False otherwise.
        """
        menu_action = kwargs.get("menu_action")
        if menu_action is not None:
            if menu_action not in self._MENU_ACTION_MAP.keys():
                return False
        state = self._state_tracker.get_episode_metric(("pokemon_core", "agent_state"))
        if menu_action is None:
            return state != AgentState.IN_DIALOGUE
            #return (state != AgentState.IN_BATTLE) and (state != AgentState.IN_DIALOGUE)
        if menu_action == "open":
            return state == AgentState.FREE_ROAM
        else:
            return state == AgentState.IN_MENU or state == AgentState.IN_BATTLE # TODO: For now we merge
            #return state == AgentState.IN_MENU
            
    def get_action_space(self):
        """
        Returns a Discrete space representing menu actions.
        Returns:
            Discrete: A Discrete space with size equal to the number of menu actions.
        """
        return Discrete(len(self._MENU_ACTION_MAP))

    def parameters_to_space(self, menu_action):
        if menu_action not in self._MENU_ACTION_MAP.keys():
            log_error(f"Invalid menu action {menu_action}", self._parameters)

    def space_to_parameters(self, space_action):
        menu_action = None
        if space_action == 0:
            menu_action = "up"
        elif space_action == 1:
            menu_action = "down"
        elif space_action == 2:
            menu_action = "confirm"
        elif space_action == 3:
            menu_action = "exit"
        elif space_action == 4:
            menu_action = "open"
        else:
            log_error(f"Invalid space action {space_action}")
        return {"menu_action": menu_action}
    
    def _execute(self, menu_action):
        action = self._MENU_ACTION_MAP[menu_action]
        current_frame = self._emulator.get_current_frame()
        frames, done = self._emulator.step(action)
        action_success = frame_changed(current_frame, frames[-1])
        return [self._state_tracker.report()], action_success
        

class BattleActions:
    pass

class PokemonCrystalBagActions:
    pass









class LocateAction(HighLevelAction):
    prompt = """
    You are playing Pokemon and are given a screen capture of the game. 
    Your job is to locate the target that best fits the description `[TARGET]`

    Do you see the target described? Answer with a single sentence and then [YES] or [NO]
    [STOP]
    Output:
    """
    REQUIRED_STATE_PARSER = PokemonStateParser
    REQUIRED_STATE_TRACKER = CorePokemonTracker
    _MAX_NEW_TOKENS = 60
    _CUTOFF_QUADRANT_LIMIT = 20

    def is_valid(self, target: str = None):
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM
    
    def get_action_space(self):
        return Text(max_length=50)
    
    def parameters_to_space(self, target: str):
        return target
    
    def space_to_parameters(self, space_action: str):
        return {"target": space_action}
    

    def check_for_target(self, prompt, screens):
        texts = [prompt] * len(screens)
        outputs = perform_vlm_inference(texts=texts, images=screens, max_new_tokens=self._MAX_NEW_TOKENS)
        founds = []
        for i, output in enumerate(outputs):
            if "yes" in output.lower():
                founds.append(True)
            else:
                founds.append(False)
        return founds
    
    def get_cells_found(self, prompt: str, grid_cells: Dict[Tuple[int, int], np.ndarray]) -> Tuple[bool, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Recursively divides the grid cells into quadrants and checks each quadrant for the target.
        Args:
            prompt: description of target
            grid_cells: the dict of the subset of grid cells to search over
        Returns:
            found (bool): whether the target was found in any of the grid cells at any point. 
            potential_cells (List[Tuple[int, int]]): list of grid cell coordinates that may contain the target. If found is true, this is almost always populated with something
                                                    The only exception is when the item was found at too high a scan and not found at lower levels (and so too many cells would have been potentials)
            definitive_cells (List[Tuple[int, int]]): list of grid cell coordinates that, with high confidence, contain the target.                        
        """
        quadrant_keys = ["tl", "tr", "bl", "br"]
        if len(grid_cells) == 1:
            screen = list(grid_cells.values())[0]
            keys = list(grid_cells.keys())[0]
            target_in_grid = self.check_for_target(prompt, [screen])[0]
            if target_in_grid:
                return False, list(grid_cells.keys()), list(grid_cells.keys())
            else:
                return False, [], []
        quadrants = self._emulator.state_parser.get_quadrant_frame(grid_cells=grid_cells) # TODO: Verify how this works on the edges
        screens = []
        for quadrant in quadrant_keys:
            screen = quadrants[quadrant]["screen"]
            screens.append(screen)
        quadrant_founds = self.check_for_target(prompt, screens)
        if not any(quadrant_founds):
            return False, [], []
        else:
            potential_cells = []
            all_cells_found = []
            quadrant_definites = []
            for i in range(len(quadrant_keys)):
                quadrant = quadrant_keys[i]
                if quadrant_founds[i]:
                    cells = quadrants[quadrant]["cells"]
                    if len(cells) < 4:
                        potential_cells.extend(cells.keys())
                        cell_keys = list(cells.keys())
                        cell_screens = [cells[key] for key in cell_keys]
                        cell_founds = self.check_for_target(prompt, cell_screens)
                        for i, found in enumerate(cell_founds):
                            if found:
                                quadrant_definites.append(cell_keys[i])
                            else:
                                pass                        
                    else:
                        found_in_quadrant, quadrant_potentials, quadrant_definites = self.get_cells_found(prompt, cells)
                        if len(quadrant_definites) > 0:
                            all_cells_found.extend(quadrant_definites)
                        else:
                            if found_in_quadrant: # then there is some potential
                                if len(quadrant_potentials) != 0:
                                    potential_cells.extend(quadrant_potentials)
                                else:
                                    if len(cells) <= self._CUTOFF_QUADRANT_LIMIT:
                                        potential_cells.extend(cells.keys())
            return True, potential_cells, all_cells_found
    
    def _execute(self, target: str):
        percieve_prompt = self.prompt.replace("[TARGET]", target)
        grid_cells = self._emulator.state_parser.capture_grid_cells(self._emulator.get_current_frame())
        found, potential_cells, definitive_cells = self.get_cells_found(percieve_prompt, grid_cells)
        self._emulator.step() # just to ensure state tracker is populated. #TODO: THIS FAILS IN DIALOGUE STATES. 
        ret_dict = self._state_tracker.report()
        ret_dict["vlm_perception"] = (found, potential_cells, definitive_cells)
        return [ret_dict], 0
    
class GridLocateAction(HighLevelAction):
    prompt = """
    You are playing Pokemon and are trying to identify whether you have found the target `[TARGET]` in the current screen. 
    Output YES if the image provided has the target, and NO otherwise.
    Give a one sentence reasoning for your decision before you do so.
    Output format:
    Reasoning: extremely brief reasoning here
    Final Answer: YES or NO
    [STOP]
    """

    REQUIRED_STATE_PARSER = PokemonStateParser
    REQUIRED_STATE_TRACKER = CorePokemonTracker

    def is_valid(self, target: str = None, quadrant: str = None):
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM and quadrant in ["tr", "tl", "br", "bl"]
    
    def get_action_space(self):
        return Text(max_length=50) # the expected action space is a concatenation with | quadrant at the end
    
    def parameters_to_space(self, target: str, quadrant: str):
        return f"{target}|{quadrant}"
    
    def space_to_parameters(self, space_action: str):
        target, quadrant = space_action.split("|")
        return {"target": target, "quadrant": quadrant}
    
    def _execute(self, target: str, quadrant: str):
        percieve_prompt = self.prompt.replace("[TARGET]", target)
        cells = self._emulator.state_parser.capture_grid_cells(self._emulator.get_current_frame(), quadrant=quadrant)
        keys = list(cells.keys())
        images = [cells[key] for key in keys]
        texts = [percieve_prompt] * len(images)
        output = perform_vlm_inference(texts=texts, images=images, max_new_tokens=256, batch_size=len(texts))
        hits = []
        for i, out in enumerate(output):
            if "final answer: yes" in out.lower():
                hits.append(keys[i])
        self._emulator.step() # just to ensure state tracker is populated. THIS FAILS IN DIALOGUE STATES. 
        ret_dict = self._state_tracker.report()
        ret_dict["grid_location_result"] = str(hits)
        return [ret_dict], 0

class TestAction(HighLevelAction):
    REQUIRED_STATE_PARSER = PokemonStateParser
    REQUIRED_STATE_TRACKER = CorePokemonTracker
    prompt = """
    You are playing Pokemon and are trying to identify whether you have found the target `[TARGET]` in the current screen. 
    Output YES if the image provided has the target, and NO otherwise. Only output YES if the full target occupies most of the image. If only a small part of it or a corner is visible, output NO.
    Give a one sentence reasoning for your decision before you do so.
    Output format:
    Reasoning: extremely brief reasoning here
    Final Answer: YES or NO
    [STOP]
    """

    def is_valid(self, **kwargs):
        return True
    
    def get_action_space(self):
        return Discrete(1) # Dummy
    
    def parameters_to_space(self):
        return 0
    
    def space_to_parameters(self, space_action):
        return {"context": "You playin pokemon"} # Dummy
    
    def _execute(self, context="A single Pokeball"):
        # Do the percieve action in the free roam state:
        percieve_prompt = self.prompt.replace("[TARGET]", context)
        quadrant = "tr"
        cells = self._emulator.state_parser.capture_grid_cells(self._emulator.get_current_frame(), quadrant=quadrant)
        keys = list(cells.keys())
        images = [cells[key] for key in keys]
        texts = [percieve_prompt] * len(images)
        output = perform_vlm_inference(texts=texts, images=images, max_new_tokens=256, batch_size=len(texts))
        hits = []
        for i, out in enumerate(output):
            if "final answer: yes" in out.lower():
                hits.append(keys[i])
        self._emulator.step() # just to ensure state tracker is populated. THIS FAILS IN DIALOGUE STATES. 
        ret_dict = self._state_tracker.report()
        ret_dict["vlm_perception"] = str(hits)
        return [ret_dict], 0
    
    # TODO: Add the action to break down the grid into pieces and check if the target is in each piece and return the grid coordinates where it is found. 
