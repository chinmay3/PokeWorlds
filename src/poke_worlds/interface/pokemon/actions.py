from poke_worlds.utils import log_error, perform_vlm_inference, perform_object_detection, log_warn
from poke_worlds.interface.action import HighLevelAction, SingleHighLevelAction
from poke_worlds.emulation.pokemon.parsers import AgentState, PokemonStateParser
from poke_worlds.emulation.pokemon.trackers import CorePokemonTracker
from poke_worlds.emulation import LowLevelActions
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from poke_worlds.utils import show_frames
from poke_worlds.utils import identify_matches
import numpy as np

from gymnasium.spaces import Box, Discrete, Text
import matplotlib.pyplot as plt
from PIL import Image

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


class PassDialogueAction(SingleHighLevelAction):
    """
    Skips dialogue by pressing the B button.

    Is Valid When: In Dialogue State

    Action Success Interpretation:
        - -1: Frame did not change
        - 0: Frame changed and no longer in dialogue state
        - 1: Frame changed but still in dialogue state
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
        report = self._state_tracker.report()
        if not report["core"]["frame_changed"]:
            action_success = -1
        else:
            action_success = 0 if self._emulator.state_parser.get_agent_state(frames[-1]) != AgentState.IN_DIALOGUE else 1
        return [report], action_success


class InteractAction(SingleHighLevelAction):
    """
    Presses the A button to interact with an object in front of the agent.

    Is Valid When: In Free Roam State

    Action Success Interpretation:
        - -1: Frame did not change or agent still in free roam state
        - 1: Agent not in free roam state
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
    

class BaseMovementAction(HighLevelAction, ABC):
    """
    Base class for movement actions in the Pokemon environment.
    Has utility methods for moving in directions.

    Is Valid When: In Free Roam State

    Action Success Interpretation:
        - -1: Frame did not change, even on the first step
        - 0: Finished all steps
        - 1: Took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter). This usually means we ran into an obstacle.
        - 2: Took some steps, but agent state changed from free roam. This often means we entered a cutscene or battle. 

    Action Returns:
        - n_steps_taken (int): Number of steps actually taken
        - rotated (bool or None): True if the player has not moved, but has rotated. If the player has moved, this will be None. If it is False, it means the player tried to walk straight into an obstacle. 
    """
    REQUIRED_STATE_TRACKER = CorePokemonTracker
    REQUIRED_STATE_PARSER = PokemonStateParser

    def judge_movement(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Judges whether movement has occurred between two frames.

        Args:
            previous_frame (np.ndarray): The previous frame.
            current_frame (np.ndarray): The current frame.
        Returns:
            Tuple[bool, bool]: A tuple containing:
            - bool: True if movement has occurred, False otherwise.
            - bool: True if the player has not moved, but has rotated.  
        """
        # if the full screen hasn't changed at all, player has neither moved nor rotated
        if not frame_changed(previous_frame, current_frame):
            return False, False
        # split the screen into quadrants and check which quadrants have changed. If any of them stayed the same, the player has not moved, but may have rotated. 
        flag = False
        for quadrant in ["screen_quadrant_1", "screen_quadrant_2", "screen_quadrant_3", "screen_quadrant_4"]:
            prev_quad = self._emulator.state_parser.capture_named_region(previous_frame, quadrant)
            curr_quad = self._emulator.state_parser.capture_named_region(current_frame, quadrant)
            prev_uniform = prev_quad.max() == prev_quad.min()
            curr_uniform = curr_quad.max() == curr_quad.min()
            if not frame_changed(prev_quad, curr_quad) and not prev_uniform and not curr_uniform: # then screen isn't just black, but also hasn't changed.
                flag = True
                break
        if flag: # then some frame stayed the same, so no movement, but maybe rotation.
            prev_player_cell = self._emulator.state_parser.capture_grid_cells(previous_frame)[(0, 0)]
            curr_player_cell = self._emulator.state_parser.capture_grid_cells(current_frame)[(0, 0)]
            if frame_changed(prev_player_cell, curr_player_cell):
                return False, True
            else:
                return False, False
        else:
            return True, None
    
    def move(self, direction: str, steps: int) -> Tuple[np.ndarray, int]:
        """
        Move in a given direction for a number of steps.
        
        :param direction: One of "up", "down", "left", "right"
        :type direction: str
        :param steps: Number of steps to move in that direction
        :type steps: int
        :return:  A tuple containing:
                
                - A list of state tracker reports after each low level action executed. Length is equal to the number of low level actions executed.
                
                - An integer action success status
        :rtype: Tuple[ndarray[_AnyShape, dtype[Any]], int]
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
        n_successful_steps = 0
        has_rotated = None
        agent_state = AgentState.FREE_ROAM
        while n_step < steps and agent_state == AgentState.FREE_ROAM:
            frames, done = self._emulator.step(action)
            transition_state_dicts.append(self._state_tracker.report())
            transition_frames.extend(frames)
            current_frame = self._emulator.get_current_frame() # Do NOT use the emulator frame, as it may have a grid on it.
            if done:
                break
            # check if frames changed. If not, break out. 
            # We check all frames in sequence to try and catch oscillations. But nothing will catch 1 step into wall in areas like this
            player_moved, player_rotated = self.judge_movement(previous_frame, current_frame)
            if player_rotated == True:
                has_rotated = True
            if not player_moved and not player_rotated:
                break
            if player_moved:
                n_successful_steps += 1 # don't count rotation as a step
            agent_state = self._emulator.state_parser.get_agent_state(self._emulator.get_current_frame())
            if agent_state != AgentState.FREE_ROAM:
                break
            n_step += 1
            previous_frame = current_frame
        if agent_state != AgentState.FREE_ROAM:
            action_success = 2
        else:
            if n_step <= 0:
                action_success = -1
            elif n_step == steps:
                action_success = 0
            else:
                action_success = 1
        transition_state_dicts[-1]["action_return"] = {"n_steps_taken": n_successful_steps, "rotated": has_rotated}
        return transition_state_dicts, action_success 

    def is_valid(self, **kwargs):
        """
        Just checks if the agent is in free roam state.
        """
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM


class MoveStepsAction(BaseMovementAction):
    """
    Moves the agent in a specified cardinal direction for a specified number of steps.

    Is Valid When: In Free Roam State
    Action Success Interpretation:
        - -1: Frame did not change, even on the first step
        - 0: Finished all steps
        - 1: Took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter). This usually means we ran into an obstacle.
        - 2: Took some steps, but agent state changed from free roam. This often means we entered a cutscene or battle.

    Action Returns:
        - n_steps_taken (int): Number of steps actually taken
        - rotated (bool or None): True if the player has not moved, but has rotated. If the player has moved, this will be None. If it is False, it means the player tried to walk straight into an obstacle.         
    """

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
            if step <= 0:
                return False
        return super().is_valid(**kwargs)
        

class MoveGridAction(BaseMovementAction):
    """
    Moves the agent on both axes. Will always try to move right first and then up. 

    Is Valid When: In Free Roam State
    Action Success Interpretation:
        - -1: Frame did not change, even on the first step
        - 0: Finished all steps
        - 1: Took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter). This usually means we ran into an obstacle.
        - 2: Took some steps, but agent state changed from free roam. This often means we entered a cutscene or battle.

    Action Returns:
        - n_steps_taken (int): Number of steps actually taken
        - rotated (bool or None): True if the player has not moved, but has rotated. If the player has moved, this will be None. If it is False, it means the player tried to walk straight into an obstacle. 
    """

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
        right_action = space_action[0]
        up_action = space_action[1]
        return {"x_steps": right_action, "y_steps": up_action}
        

    def parameters_to_space(self, x_steps, y_steps):
        move_vec = np.zeros(2) # x, y
        move_vec[0] = x_steps
        move_vec[1] = y_steps
        return move_vec


    def _execute(self, x_steps, y_steps):
        x_direction = "right" if x_steps >=0 else "left"
        y_direction = "up" if y_steps >= 0 else "down"
        if x_steps != 0:
            transition_states, status = self.move(direction=x_direction, steps=abs(x_steps))
            if status != 0:
                return transition_states, status
        else:
            transition_states = []
        if y_steps != 0:
            more_transition_states, status = self.move(direction=y_direction, steps=abs(y_steps))
            transition_states.extend(more_transition_states)
        return transition_states, status
    
    def is_valid(self, x_steps: int=None, y_steps: int=None):
        if x_steps is not None and y_steps is not None:
            if not isinstance(x_steps, int) or not isinstance(y_steps, int):
                return False
            if x_steps == 0 and y_steps == 0:
                return False
        return super().is_valid()

class MenuAction(HighLevelAction):
    """
    Allows simple navigation and option selection of menus. 

    Is Valid When: In Menu State

    Action Success Interpretation:
        - -1: Frame did not change.
        - 0: Frame changed. 

    TODO: Unfortunately, since PokemonCrystal based games has a bag with submenus, I need to add left right actions as well.
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
            "back": LowLevelActions.PRESS_BUTTON_B,
            "open": LowLevelActions.PRESS_BUTTON_START
    }

    def is_valid(self, **kwargs):
        """
        Checks if the menu action is valid in the current state.

        Args:
            menu_action (str, optional): The menu action to check. One of "up", "down", "confirm", "back", "open".
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
            menu_action = "back"
        elif space_action == 4:
            menu_action = "open"
        else:
            log_error(f"Invalid space action {space_action}")
        return {"menu_action": menu_action}
    
    def _execute(self, menu_action):
        action = self._MENU_ACTION_MAP[menu_action]
        current_frame = self._emulator.get_current_frame()
        frames, done = self._emulator.step(action)
        action_success = 0 if frame_changed(current_frame, frames[-1]) else -1
        return [self._state_tracker.report()], action_success
        
class LocateAction(HighLevelAction):
    """
    Locates a target in the current screen.
    1. Divides the screen into grid cells.
    2. Recursively divides the grid cells into quadrants and checks each quadrant for the target.
    3. If a quadrant contains the target, further divides it into smaller quadrants until the smallest grid cells are reached.
    4. Returns the grid cell coordinates that may contain the target.
    
    Uses VLM inference to check each grid cell for the target.

    Is Valid When: In Free Roam State
    
    Action Success Interpretation:
            - 0: Object found definitively
            - 1: Object found but only potentially
            - 2: Object not found
    """

    prompt = """
    You are playing Pokemon and are given a screen capture of the game. 
    Your job is to locate the target that best fits the description `[TARGET]`

    Do you see the target described? Answer with a single sentence and then [YES] or [NO]
    [STOP]
    Output:
    """
    REQUIRED_STATE_PARSER = PokemonStateParser
    REQUIRED_STATE_TRACKER = CorePokemonTracker

    pre_described_options = {
        "item": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "pokeball": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "npc": "a pixelated human-like character sprite",
        "grass": "a pixelated, greyscale patch of grass that resembles wavy dark lines.",
        "sign": "a pixelated, greyscale white signpost with dots on its face"
    }    

    image_references = {
        "item": "pokeball",
        "pokeball": "pokeball",
        "grass": "grass",
        "sign": "sign"
    }

    def coord_to_string(self, coord: Tuple[int, int]) -> str:
        start = "("
        c1 = coord[0]
        if c1 > 0:
            start += f"{c1} steps to right from you, "
        elif c1 < 0:
            start += f"{-c1} steps to left from you, "
        c2 = coord[1]
        if c2 > 0:
            start += f"{c2} steps up from you)"
        elif c2 < 0:
            start += f"{-c2} steps down from you)"
        return start
    
    def coords_to_string(self, coords: List[Tuple[int, int]]) -> str:
        coord_strings = [self.coord_to_string(coord) for coord in coords]
        return "[" + ", ".join(coord_strings) + "]"

    def is_valid(self, target: str=None):
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM
    
    def get_action_space(self):
        return Text(max_length=50)
    
    def parameters_to_space(self, target: str):
        return target
    
    def space_to_parameters(self, space_action: str):
        return {"target": space_action}

    def check_for_target(self, prompt, screens, image_reference: str = None):
        if image_reference is None:
            description = prompt
            texts = [prompt] * len(screens)
            return perform_object_detection(images=screens, texts=texts)
        else:
            description = prompt
            reference_image = self._emulator.state_parser.get_image_reference(image_reference)
            founds = identify_matches(description=description, screens=screens, reference=reference_image)
            return founds
    
    def get_centroid(self, cells: Dict[Tuple[int, int], np.ndarray]) -> Tuple[float, float]:
        min_x = min([coord[0] for coord in cells.keys()])
        min_y = min([coord[1] for coord in cells.keys()])
        max_x = max([coord[0] for coord in cells.keys()])
        max_y = max([coord[1] for coord in cells.keys()])
        centroid_x = (min_x + max_x) // 2
        centroid_y = (min_y + max_y) // 2
        return (centroid_x, centroid_y)
    
    def get_cells_found(self, grid_cells: Dict[Tuple[int, int], np.ndarray], prompt: str, image_reference: str=None) -> Tuple[bool, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Recursively divides the grid cells into quadrants and checks each quadrant for the target.
        Args:
            grid_cells: the dict of the subset of grid cells to search over
            prompt: description of target
            image_reference: reference image for the target

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
            target_in_grid = self.check_for_target(prompt, [screen], image_reference=image_reference)[0]
            if target_in_grid:
                return True, list(grid_cells.keys()), list(grid_cells.keys())
            else:
                return False, [], []
        quadrants = self._emulator.state_parser.get_quadrant_frame(grid_cells=grid_cells)
        screens = []
        for quadrant in quadrant_keys:
            screen = quadrants[quadrant]["screen"]
            screens.append(screen)
        quadrant_founds = self.check_for_target(prompt, screens, image_reference=image_reference)
        if not any(quadrant_founds):
            return False, [], []
        else:
            potential_cells = []
            quadrant_definites = []
            for i in range(len(quadrant_keys)):
                quadrant = quadrant_keys[i]
                if quadrant_founds[i]:
                    cells = quadrants[quadrant]["cells"]
                    if len(cells) < 4:
                        potential_cells.append(self.get_centroid(cells))
                        cell_keys = list(cells.keys())
                        cell_screens = [cells[key] for key in cell_keys]
                        cell_founds = self.check_for_target(prompt, cell_screens, image_reference=image_reference)
                        for i, found in enumerate(cell_founds):
                            if found:
                                quadrant_definites.append(cell_keys[i])
                            else:
                                pass                        
                    else:
                        found_in_quadrant, quadrant_potentials, recursive_quadrant_definites = self.get_cells_found(cells, prompt, image_reference=image_reference)
                        if len(recursive_quadrant_definites) > 0:
                            quadrant_definites.extend(recursive_quadrant_definites)
                        if found_in_quadrant: # then there is some potential, so add the quadrants potentials. 
                            if len(quadrant_potentials) != 0:
                                potential_cells.extend(quadrant_potentials)
                            else:
                                potential_cells.append(self.get_centroid(cells))
            return True, potential_cells, quadrant_definites
    
    
    def do_location(self, target: str, image_reference: str = None):
        """
        Execute location on a free-form target string.
        """
        if image_reference is None:
            percieve_prompt = self.prompt.replace("[TARGET]", target)
        else:
            percieve_prompt = f"a {target} from Pokemon"
        grid_cells = self._emulator.state_parser.capture_grid_cells(self._emulator.get_current_frame())
        found, potential_cells, definitive_cells = self.get_cells_found(grid_cells, percieve_prompt, image_reference=image_reference)
        self._emulator.step() # just to ensure state tracker is populated.
        ret_dict = self._state_tracker.report()
        ret_dict["action_return"] = {"found": found, "potential_cells": potential_cells, "definitive_cells": definitive_cells}
        action_success = None
        if len(definitive_cells) > 0:
            action_success = 0
        elif found:
            action_success = 1
        else:
            action_success = -1
        return [ret_dict], action_success
    
    def _execute(self, target: str):
        if target in self.image_references and False:
            return self.do_location(target=self.pre_described_options[target], image_reference=self.image_references[target])
        elif target in self.pre_described_options:
            return self.do_location(target=self.pre_described_options[target])
        else:
            return self.do_location(target=target)


class CheckInteractionAction(SingleHighLevelAction):
    """
    Checks whether a target object is in the interaction sphere of the agent.
    Uses VLM inference to check each grid cell for the target.
    1. Checks the orientation of the agent using VLM inference.
    2. Captures the grid cell in front of the agent and uses VLM inference to describe what is in the cell.

    Is Valid When: In Free Roam State

    Action Success Interpretation:
        - -1: VLM failure: Could not parse yes or no from response
        - 0: There is something to interact with in front of the agent.
        - 1: There is nothing to interact with in front of the agent.

    """
    orientation_prompt = """
    You are playing Pokemon and are given a screen capture of the player. Which direction is the player facing?
    Do not give any explanation, just your answer. 
    Answer with one of: UP, DOWN, LEFT, RIGHT and then [STOP]
    Output:
    """

    percieve_prompt = """
    You are playing Pokemon and are given a screen capture of the grid cell in front of the player. 
    Briefly describe what you see in the image, is it an item, or NPC that can be interacted with? Or is it a door or cave that can be entered? If the cell seems empty (or a background texture), say so.
    Give your answer in the following format:
    Description: <a single sentence description of the cell>
    Answer: <YES (if there is something to interact with) or NO (if there is nothing to interact with)>
    and then [STOP]
    Description:
    """
    def is_valid(self, **kwargs):
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM
    
    def _execute(self):
        current_frame = self._emulator.get_current_frame()
        grid_cells = self._emulator.state_parser.capture_grid_cells(current_frame=current_frame)

        orientation_output = perform_vlm_inference(texts=[self.orientation_prompt], images=[grid_cells[(0, 0)]], max_new_tokens=5)[0]
        cardinal = None
        if "up" in orientation_output.lower():
            cardinal = (0, 1)
        elif "down" in orientation_output.lower():
            cardinal = (0, -1)
        elif "left" in orientation_output.lower():
            cardinal = (-1, 0)
        elif "right" in orientation_output.lower():
            cardinal = (1, 0)
        if cardinal is None:
            return [self._state_tracker.report()], -1 # This should not happen. It is a VLM failure. 
        cardinal_screen = grid_cells[cardinal]
        percieve_output = perform_vlm_inference(texts=[self.percieve_prompt], images=[cardinal_screen], max_new_tokens=50)[0]
        action_success = 0
        if "answer: yes".lower() in percieve_output.lower():
            percieve_output = percieve_output + " you can use interact() now"
        elif "answer: no".lower() in percieve_output.lower():
            action_success = 1
            percieve_output += "You cannot use interact() now"
        else:
            action_success = -1
        ret_dict = self._state_tracker.report()
        ret_dict["action_return"] = {"orientation": cardinal, "percieve_output": percieve_output}
        return [ret_dict], action_success


class SeekAction(LocateAction):
    """
    Chains together location, movement and check interaction. 
    """
    resolve_prompt = """
You are playing Pokemon and are given a screen capture of the game. The user is looking for [TARGET] with the intent of [INTENT] and has narrowed it down to the following possible locations:
[POSSIBLE_CELLS]
Your job is to reason about the screens and the options and identify the single best candidate cell to move towards. 
Format your response as follows:
Picture Reasoning: Look at the screen and describe which area seems to best match the target description and why. State roughly where it is with respect to the player (straight ahead, to the left and forwward, etc.)
Cell Reasoning: Out of the given possible cells, which cell seems to align best with your picture reasoning and why? You must select one answer by the end, so if you cannot decide, just pick one of them. 
Final Answer: the single best cell to move towards in the format (<x: int> steps <right or left>, <y: int> steps <up or down>)
[STOP]
Output:
    """

    def is_valid(self, intent: str=None, target: str=None):
        return LocateAction.is_valid(self, target=target)

    def move(self, **kwargs):
        return BaseMovementAction.move(self, **kwargs)
    
    def get_action_space(self):
        return LocateAction.get_action_space(self)

    def parameters_to_space(self, intent: str, target: str):
        return f"{intent} | {target}"
    
    def space_to_parameters(self, space_action: str):
        if "|" not in space_action:
            return None
        intent = space_action.split("|")[0].strip()
        target = space_action.split("|")[1].strip()
        return {"intent": intent, "target": target}
    
    def _execute(self, intent: str, target: str):
        self.orientation_prompt = CheckInteractionAction.orientation_prompt
        self.percieve_prompt = CheckInteractionAction.percieve_prompt
        ret_state_list, locate_status = super()._execute(target=target)
        location_results = ret_state_list[-1]["action_return"]
        if not location_results["found"]: #
            return ret_state_list, -1 # Not found
        else:
            # don't discriminate between definitive and potential here. Just use both.
            possible_cells = list(set(location_results["potential_cells"] + location_results["definitive_cells"]))
            print(f"Possible cells for seek: {self.coords_to_string(possible_cells)}")
            if len(possible_cells) > 1:
                cell_string = ""
                for i, cell in enumerate(possible_cells):
                    cell_string += f"Cell {i+1}: {self.coord_to_string(cell)}. "
                resolve_prompt = self.resolve_prompt.replace("[TARGET]", target)
                resolve_prompt = resolve_prompt.replace("[INTENT]", intent)
                resolve_prompt = resolve_prompt.replace("[POSSIBLE_CELLS]", cell_string)
                resolve_output = perform_vlm_inference(texts=[resolve_prompt], images=[ret_state_list[-1]["core"]["current_frame"]], max_new_tokens=350)[0]
                error = f"VLM failed to produce final answer in seek action. \nOutput: {resolve_output}"
                if "final answer:" not in resolve_output.lower():
                    log_warn(error, self._parameters)
                    selected_cell = possible_cells[0]
                else:
                    final_answer_part = resolve_output.lower().split("final answer:")[-1]
                    #(x steps right/left, y steps up/down)
                    if "(" not in final_answer_part or ")" not in final_answer_part:
                        log_warn(error, self._parameters)
                        selected_cell = possible_cells[0]
                    else:
                        coord_part = final_answer_part.split("(")[-1].split(")")[0]
                        coord_parts = coord_part.split(",")
                        if len(coord_parts) != 2:
                            log_warn(error, self._parameters)
                            selected_cell = possible_cells[0]
                        else:
                            x_part = coord_parts[0].strip()
                            y_part = coord_parts[1].strip()
                            if "steps" not in x_part or "steps" not in y_part:
                                log_warn(error, self._parameters)
                                selected_cell = possible_cells[0]
                            else:
                                x_steps = x_part.split("steps")[0].strip()
                                y_steps = y_part.split("steps")[0].strip()
                                if not x_steps.isnumeric() or not y_steps.isnumeric():
                                    log_warn(error, self._parameters)
                                    selected_cell = possible_cells[0]
                                else:
                                    x_steps = int(x_steps)
                                    y_steps = int(y_steps)
                                    if "left" in x_part:
                                        x_steps = -x_steps
                                    if "down" in y_part:
                                        y_steps = -y_steps
                                    selected_cell = (x_steps, y_steps)
                                    print(f"Selected cell for seek: {self.coord_to_string(selected_cell)} from VLM output: {resolve_output}")
            else:
                selected_cell = possible_cells[0]
            # Now move to the selected cell
            x_move = selected_cell[0]
            y_move = selected_cell[1]
            print("Moving with args: ", x_move, y_move)
            transition_states, move_status = MoveGridAction._execute(self, x_steps=x_move, y_steps=y_move)
            if move_status != 0:
                # then we stop here, return the movement status shifted up by maximum locate status
                move_status = move_status + 3
                # so now, 2 -> no move, 3 -> SHOULD NOT HAPPEN, 4 -> partial move, 5 -> state change
                return transition_states, move_status
            check_states, check_status = CheckInteractionAction._execute(self)
            print("Got check results: ", check_states[-1]["action_return"])
            transition_states.extend(check_states)
            if check_status == 0: # we can interact. Hit interact
                interaction_states, interact_status = InteractAction._execute(self)
                transition_states.extend(interaction_states)
                if interact_status == 0:
                    return transition_states, check_status # 0 should be success here
                else:
                    return transition_states, 9 # 9 -> interaction failed after seek. This means checkInteract said we could interact but we couldn't.
            else:
                return transition_states, check_status + 7 # So now 6 -> VLM failure, 7 -> Should not happen, 8 -> Nothing to interact with
            
        
class BattleMenuAction(HighLevelAction):
    """
    Allows navigation of the battle menu.

    Is Valid When: In Battle State

    Action Success Interpretation:
        - -1: Navigation Failure. Screen did not end up in expected end state. 
        - 0: Navigation Success. Screen ended up in expected end state. For run, got away safely.
        - 1: Run Attempt Failed (cannot escape wild pokemon)
        - 2: Run Attempt Failed (cannot escape trainer battle)
    """
    _OPTIONS = ["fight", "bag", "pokemon", "run", "progress"]
    def is_valid(self, option: str = None):
        if option is not None and option not in self._OPTIONS:
            return False
        state = self._state_tracker.get_episode_metric(("pokemon_core", "agent_state"))
        # TODO: Must check we aren't in a 'learn new move' screen
        return state == AgentState.IN_BATTLE

    def get_action_space(self):
        return Discrete(len(self._OPTIONS))
    
    def get_all_valid_parameters(self):
        state = self._state_tracker.get_episode_metric(("pokemon_core", "agent_state"))
        if state != AgentState.IN_BATTLE:
            return []
        return [{"option": option} for option in self._OPTIONS]
    
    def parameters_to_space(self, option: str):
        return self._OPTIONS.index(option)

    
    def space_to_parameters(self, space_action):
        if space_action < 0 or space_action >= len(self._OPTIONS):
            return None
        return {"option": self._OPTIONS[space_action]}
    
    
    def go_to_battle_menu(self):
        # assumes we are in battle menu or will get there with some B's
        state_reports = []
        for i in range(3):
            self._emulator.step(LowLevelActions.PRESS_BUTTON_B)
            state_reports.append(self._state_tracker.report())
        return state_reports

    def button_sequence(self, low_level_actions: List[LowLevelActions]):
        state_reports = self.go_to_battle_menu()
        for action in low_level_actions:
            self._emulator.step(action)
        self._emulator.step(LowLevelActions.PRESS_BUTTON_A) # confirm option
        return state_reports + [self._state_tracker.report()]


    def go_to_fight_menu(self):
        return self.button_sequence([LowLevelActions.PRESS_ARROW_UP, LowLevelActions.PRESS_ARROW_LEFT])

    def go_to_bag_menu(self):
        return self.button_sequence([LowLevelActions.PRESS_ARROW_DOWN, LowLevelActions.PRESS_ARROW_LEFT])

    def go_to_pokemon_menu(self):
        return self.button_sequence([LowLevelActions.PRESS_ARROW_UP, LowLevelActions.PRESS_ARROW_RIGHT])

    def go_to_run(self):
        return self.button_sequence([LowLevelActions.PRESS_ARROW_DOWN, LowLevelActions.PRESS_ARROW_RIGHT])
    
    def _execute(self, option):
        success = -1
        if option == "fight":
            state_reports = self.go_to_fight_menu()
            success = 0 if self._emulator.state_parser.is_in_fight_options_menu(self._emulator.get_current_frame()) else -1
        elif option == "bag":
            state_reports = self.go_to_bag_menu()
            success = 0 if self._emulator.state_parser.is_in_fight_bag(self._emulator.get_current_frame()) else -1
        elif option == "pokemon":
            state_reports = self.go_to_pokemon_menu()
            success = 0 if self._emulator.state_parser.is_in_pokemon_menu(self._emulator.get_current_frame()) else -1
        elif option == "run":
            state_reports = self.go_to_run()
            current_frame = self._emulator.get_current_frame()
            got_away_safely = self._emulator.state_parser.named_region_matches_multi_target(current_frame, "dialogue_box_middle", "got_away_safely")
            cannot_escape = self._emulator.state_parser.named_region_matches_multi_target(current_frame, "dialogue_box_middle", "cannot_escape")
            cannot_run_from_trainer = self._emulator.state_parser.named_region_matches_multi_target(current_frame, "dialogue_box_middle", "cannot_run_from_trainer")
            if got_away_safely:
                success = 0
                self._emulator.step(LowLevelActions.PRESS_BUTTON_B) # to clear the dialogue
            elif cannot_escape:
                success = 1
                self._emulator.step(LowLevelActions.PRESS_BUTTON_B) # to clear the dialogue
            elif cannot_run_from_trainer:
                success = 2
                self._emulator.step(LowLevelActions.PRESS_BUTTON_B)
                state_reports.append(self._state_tracker.report())
                self._emulator.step(LowLevelActions.PRESS_BUTTON_B) # Twice, to clear the dialogue
            else:
                pass # Should never happen, but might. 
            state_reports.append(self._state_tracker.report())
            return state_reports, success
        elif option == "progress":
            current_frame = self._emulator.get_current_frame()
            state_reports = self.go_to_battle_menu()
            new_frame = self._emulator.get_current_frame()
            if frame_changed(current_frame, new_frame):
                success = 0 # valid frame change, screen changed
            else:
                success = -1 # uneccesary progress press
        else:
            pass # Will never happen.
        return state_reports, success
    

class PickAttackAction(HighLevelAction):
    """
    Selects an attack option in the battle fight menu.

    Is Valid When: In Battle State, In Fight Menu

    Action Success Interpretation:
        - -1: Navigation Failure. Either could not get to the top of the attack menu (this should not happen) or the option index was too high (more likely the cause of failure).
        - 0: Used attack successfully.
        - 1: Tried to use a move with no PP remaining.
    """
    def get_action_space(self):
        return Discrete(4)
    
    def is_valid(self, option: int = None):
        option = option - 1
        if option is not None:
            if option < 0 or option >=4:
                return False        
        return self._emulator.state_parser.is_in_fight_options_menu(self._emulator.get_current_frame())
    
    def parameters_to_space(self, option: int):
        option = option - 1
        return option
    
    def space_to_parameters(self, space_action):
        return {"option": space_action + 1}
    
    def _execute(self, option: int):
        # assume we are in the attack menu already
        # first go to the top:
        option = option - 1
        flag = False
        for _ in range(4):
            self._emulator.step(LowLevelActions.PRESS_ARROW_UP)
            if self._emulator.state_parser.is_on_top_attack_option(self._emulator.get_current_frame()):
                flag = True
                break
        if not flag: # could not get to top. Some error
            return [self._state_tracker.report()], -1
        # then go down option times
        for time in range(option):
            self._emulator.step(LowLevelActions.PRESS_ARROW_DOWN)
            if self._emulator.state_parser.is_on_top_attack_option(self._emulator.get_current_frame()):
                # went back to top, means that option was invalid
                return [self._state_tracker.report()], -1
        state_reports = []
        self._emulator.step(LowLevelActions.PRESS_BUTTON_A) # confirm option
        state_reports.append(self._state_tracker.report())
        if self._emulator.state_parser.tried_no_pp_move(self._emulator.get_current_frame()):
            self._emulator.step(LowLevelActions.PRESS_BUTTON_B) # to clear the no PP dialogue
            state_reports.append(self._state_tracker.report())
            return state_reports, 1 # tried to use a move with no PP
        else:
            self._emulator.step(LowLevelActions.PRESS_BUTTON_B) # to get through any attack animation dialogue
            state_reports.append(self._state_tracker.report())
        return state_reports, 0


class PokemonCrystalBagActions:
    pass


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
        cells = self._emulator.state_parser.capture_grid_cells(self._emulator.get_current_frame())
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
        return [ret_dict], 0
    
    # TODO: Add the action to break down the grid into pieces and check if the target is in each piece and return the grid coordinates where it is found. 
