from typing import Optional, Dict, Any, List, Tuple

from gymnasium import spaces

from poke_worlds.emulation.pokemon.parsers import AgentState
from poke_worlds.interface.action import HighLevelAction
from poke_worlds.interface.pokemon.actions import InteractAction, MoveStepsAction
from poke_worlds.utils import load_parameters, log_dict, log_info, ocr
from poke_worlds.emulation.pokemon.emulators import PokemonEmulator
from poke_worlds.emulation.pokemon.trackers import CorePokemonTracker, PokemonRedStarterTracker, PokemonOCRTracker
from poke_worlds.interface.environment import DummyEnvironment, Environment
from poke_worlds.interface.controller import Controller

import gymnasium as gym
import numpy as np



class PokemonEnvironment(DummyEnvironment):
    REQUIRED_EMULATOR = PokemonEmulator
    


class PokemonRedChooseCharmanderFastEnv(Environment):
    REQUIRED_TRACKER = PokemonRedStarterTracker    

    def __init__(self, emulator: PokemonEmulator, controller: Controller, parameters: Optional[dict]=None):
        """
        Initializes the Environment with the given emulator and controller.
        
        """
        self._parameters = load_parameters(parameters)
        self._emulator = emulator
        self._controller = controller
        # observation space is a dictionary with "coords" key containing the (x, y) coordinates of the player and "facing" key containing the direction the player is facing
        coord_space = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint16)
        direction_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Dict({
            "facing": direction_space,
            "coords": coord_space
        })
        """ The observation space is the raw pixel values of the emulator's screen and the (x, y) coordinates of the player. """
        super().__init__()

    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        """
        Override default emulator keyword arguments for this environment.
        """
        emulator_kwargs["state_tracker_class"] = PokemonRedStarterTracker
        emulator_kwargs["init_state"] = "starter"
        return emulator_kwargs

    def get_observation(self, **kwargs):
        info = self.get_info()
        coords = info["pokemon_red_location"]["current_local_location"][:2]
        direction = info["pokemon_red_location"]["direction"]
        facing = None
        if direction == (1, 0):
            facing = 0  # Right
        elif direction == (-1, 0):
            facing = 1  # Left
        elif direction == (0, -1):
            facing = 2  # Up
        elif direction == (0, 1):
            facing = 3  # Down
        observation = {
            "facing": facing,
            "coords": np.array(coords, dtype=np.uint16)
        }
        return observation
    
    def determine_terminated(self, state):
        starter_chosen = state["pokemon_red_starter"]["current_starter"]
        return starter_chosen is not None    
    
    def determine_reward(self, start_state, action, action_kwargs, transition_states, action_success) -> float:
        """
        Reward the agent for choosing Charmander as quickly as possible.
        """
        current_state = transition_states[-1]
        starter_chosen = current_state["pokemon_red_starter"]["current_starter"]
        n_steps = current_state["core"]["steps"]
        if starter_chosen is None:
            if n_steps >= self._emulator.max_steps-2: # some safety
                return -5.0 # Penalty for not choosing a starter within max steps
            starter_spots = [(5, 3), (6, 4), (7, 4), (8, 4), (9, 3), (8, 2), (7, 2), (6, 2)]
            player_pos = current_state["pokemon_red_location"]["current_local_location"][:2]
            # compute manhattan distance to closest starter spot
            dists = [abs(player_pos[0]-spot[0]) + abs(player_pos[1]-spot[1]) for spot in starter_spots]
            min_dist = min(dists)
            if min_dist == 0:
                return 1 # Small reward for being on a starter spot
            elif min_dist < 2:
                return 0.0 # Neutral for being closish
            elif min_dist < 4:
                return -1.0 # Small penalty for being farther
            else:
                return -5.0 # Ridiculous penalty for being far away
        step_bonus = 100 / (n_steps+1)
        if starter_chosen == "charmander":
            return 500.0 + step_bonus
        else:
            return 100.0 + step_bonus# Penalty for choosing the wrong starter. For now, just less reward.


class PokemonHighLevelEnvironment(DummyEnvironment):
    """ A dummy environment that does nothing special. """
    REQUIRED_STATE_TRACKER = PokemonOCRTracker
    REQUIRED_EMULATOR = PokemonEmulator

    def __init__(self, action_buffer_max_size: int = 3, ocr_buffer_max_size: int = 10, **kwargs):
        """
        Initializes the DummyEnvironment with the given emulator and controller.

        It is safe to overwrite the self.observation_space in the subclass after calling this __init__ method.
        """
        super().__init__(**kwargs)
        self.action_buffer_max_size = action_buffer_max_size
        self.ocr_buffer_max_size = ocr_buffer_max_size
        screen_shape = self._emulator.screen_shape
        screen_space = spaces.Box(low=0, high=255, shape=(screen_shape[1], screen_shape[0], 1), dtype=np.uint8)
        action_buffer = spaces.Text(max_length=512)
        ocr = spaces.Text(max_length=512)
        state = spaces.Discrete(4) # In Dialogue, In Menu, Battle, Free Roam

        self.observation_space = spaces.Dict({
            "screen": screen_space,
            "action_buffer": action_buffer,
            "ocr": ocr,
            "state": state
        })
        """ The observation space is the raw pixel values of the emulator's screen and messages with OCR text and error signals from HighLevelActions"""
        self.action_buffer: List[Tuple[HighLevelAction, Dict[str, Any], int, dict]] = []
        """ Buffer of recent actions taken in the environment. Each entry is a tuple of (action, kwargs, success_code, action_returns)."""
        self.ocr_buffer: List[Tuple[int, Dict[str, str]]] = []
        """ Buffer of recent OCR results. Each entry is a tuple of (emulator_step_index (not the same as environment step), ocr_texts). Where ocr_texts is a dictionary of form {key: text} """        

    @staticmethod
    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        basic_tracker = PokemonOCRTracker
        incoming_tracker = emulator_kwargs.get("state_tracker_class", "default")
        if isinstance(incoming_tracker, str):
            emulator_kwargs["state_tracker_class"] = basic_tracker
        else:
            emulator_kwargs["state_tracker_class"] = Environment.safe_override_state_tracker_class(incoming_tracker, basic_tracker)
        return emulator_kwargs

    def add_to_action_buffer(self, action: HighLevelAction, action_kwargs, action_success: int, action_return: dict):
        """ Adds an action to the action buffer, maintaining the maximum size. """
        self.action_buffer.append((action, action_kwargs, action_success, action_return))
        if len(self.action_buffer) > self.action_buffer_max_size:
            self.action_buffer.pop(0)

    def action_buffer_to_str(self) -> str:
        """ Converts the action buffer to a simplestring representation. """
        buffer_str = ""
        for i, (action, action_kwargs, action_success, action_return) in enumerate(self.action_buffer):
            buffer_str += f"Action {i}: {action.__name__} | args {action_kwargs} | success code: {action_success} | return: {action_return}\n"
        return buffer_str

    def get_observation(self, *, action=None, action_kwargs=None, transition_states=None, action_success=None, add_to_buffers: bool=True):
        """
        WARNING: This method does NOT obey the gym API standards, since its returns do NOT match the observation space declared in __init__. 
        This is because the observation space definition is too restrictive. 

        In specific, this method returns "action_return" and other similar args, which is not part of the observation space.

        You can make this gym friendly by wrapping this in a class to remove the extra fields.
        """
        if transition_states is None:
            current_state = self.get_info()
            screen = current_state["core"]["current_frame"]
            # Will not try to add to OCR buffers, because no transition states should only be called on init. 
            if "ocr" in current_state and "ocr_texts" in current_state["ocr"]:
                ocr_texts = current_state["ocr"]["ocr_texts"] # is a dict with kind -> text
                ocr_combined = " | ".join([f"{kind}: {text}" for kind, text in ocr_texts.items()])
            else:
                ocr_combined = ""
            action_return = {}
        else:
            screen = transition_states[-1]["core"]["current_frame"]
            action_return = {}
            if "action_return" in transition_states[-1]:
                action_return = transition_states[-1]["action_return"]
            if add_to_buffers:
                self.add_to_action_buffer(action, action_kwargs, action_success, action_return)
            ocr_texts_all = []
            ocr_buffer_addition = []
            for state in transition_states:
                if "ocr" in state and "ocr_texts" in state["ocr"]:
                    ocr_texts = state["ocr"]["ocr_texts"] # is a dict with kind -> text
                    ocr_step = state["ocr"]["step"]
                    ocr_texts_all.append(ocr_texts)
                    ocr_buffer_addition.append((ocr_step, ocr_texts))
            # combine all ocr texts
            ocr_combined = ""
            for ocr_texts in ocr_texts_all:
                for kind, text in ocr_texts.items():
                    ocr_combined += f"{kind}: {text} | "
            if add_to_buffers:
                self.ocr_buffer.extend(ocr_buffer_addition)
                if len(self.ocr_buffer) > self.ocr_buffer_max_size:
                    self.ocr_buffer = self.ocr_buffer[-self.ocr_buffer_max_size:]
        current_state = self._emulator.state_parser.get_agent_state(screen)
        observation = {
            "screen": screen,
            "action_buffer": self.action_buffer_to_str(),
            "ocr": ocr_combined,
            "state": current_state,
            "action": action,
            "action_kwargs": action_kwargs,
            "action_success": action_success,
            "action_return": action_return
        }
        return observation
    
    def get_action_buffer(self):
        return self.action_buffer
    
    def get_ocr_buffer(self):
        return self.ocr_buffer

    def render_obs(self, action=None, action_kwargs=None, transition_states=None, action_success=None): # Might cause issues if you try to render() as well
        """
        Renders the observation space by displaying all the frames passed during the action execution.

        Args:
            action (Optional[HighLevelAction]): The previous action taken.
            action_kwargs (dict): The keyword arguments used for the action.
            transition_states (Optional[List[Dict[str, Dict[str, Any]]]]): The states observed during the action execution.
            action_success (Optional[int]): The success code of the action.
        """
        info = self.get_info()
        if transition_states is not None and len(transition_states) > 0:
            screens = transition_states[0]["core"]["passed_frames"]
            for transition_state in transition_states[1:]:
                screens = np.concatenate([screens, transition_state["core"]["passed_frames"]], axis=0)
        else:
            screens = info["core"]["passed_frames"]
        if screens is None:
            screens = [info["core"]["current_frame"]]
        for screen in screens:
            self._screen_render(screen)
        obs = self.get_observation(action=action, action_kwargs=action_kwargs, transition_states=transition_states, action_success=action_success, add_to_buffers=False)
        obs.pop("screen")
        log_info(f"Obs Strings:", self._parameters)
        log_dict(obs, parameters=self._parameters)

    def render_info(self, action=None, action_kwargs=None, transition_states=None, action_success=None):
        info = self.get_info()
        info["core"].pop("current_frame")
        info["core"].pop("passed_frames")
        log_info("State: ", self._parameters)
        log_dict(info, self._parameters)
    