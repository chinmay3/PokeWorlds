# Putting the environments that need transformers here only, so that technically the code runs without it:

from poke_worlds.utils import load_parameters, verify_parameters, log_error, log_dict, log_info, ocr
from poke_worlds.interface.environment import Environment, DummyEnvironment
from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController
from poke_worlds.interface.action import HighLevelAction
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction
from poke_worlds.emulation.pokemon.emulators import PokemonEmulator
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser, AgentState
from poke_worlds.emulation.pokemon.trackers import CorePokemonTracker
from typing import List, Tuple, Dict, Any
import numpy as np

from gymnasium import spaces

class PokemonHighLevelEnvironment(DummyEnvironment):
    """ A dummy environment that does nothing special. """
    REQUIRED_STATE_TRACKER = CorePokemonTracker
    REQUIRED_EMULATOR = PokemonEmulator

    def __init__(self, action_buffer_max_size: int = 3, **kwargs):
        """
        Initializes the DummyEnvironment with the given emulator and controller.

        It is safe to overwrite the self.observation_space in the subclass after calling this __init__ method.
        """
        super().__init__(**kwargs)
        self.action_buffer_max_size = action_buffer_max_size
        screen_shape = self._emulator.screen_shape
        screen_space = spaces.Box(low=0, high=255, shape=(screen_shape[1], screen_shape[0], 1), dtype=np.uint8)
        text_output = spaces.Text(max_length=512)
        self.observation_space = spaces.Dict({
            "screen": screen_space,
            "messages": text_output
        })
        """ The observation space is the raw pixel values of the emulator's screen and messages with OCR text and error signals from HighLevelActions"""
        self.action_buffer: List[Tuple[HighLevelAction, Dict[str, Any], int, str]] = []
        """ Buffer of recent actions taken in the environment. Each entry is a tuple of (action, kwargs, success_code, success_message)."""


    def add_to_action_buffer(self, action: HighLevelAction, action_kwargs, action_success: int, success_message: str):
        """ Adds an action to the action buffer, maintaining the maximum size. """
        self.action_buffer.append((action, action_kwargs, action_success, success_message))
        if len(self.action_buffer) > self.action_buffer_max_size:
            self.action_buffer.pop(0)


    def get_observation(self, *, action=None, action_kwargs=None, transition_states=None, action_success=None):
        if transition_states is None:
            screen = self.get_info()["core"]["current_frame"]
        else:
            screen = transition_states[-1]["core"]["current_frame"]
        dialogue_message = ""
        action_success_message = ""
        if transition_states is None:
            pass
        else:
            dialogue_frames = []    
            screens = transition_states[0]["core"]["passed_frames"]
            for transition_state in transition_states[1:]:
                screens = np.concatenate([screens, transition_state["core"]["passed_frames"]], axis=0)
            if screens is None: # then no passed_frames, rely on just the current screen
                screens = [transition_states[-1]["core"]["current_frame"]]
            for screen in screens:
                if self._emulator.state_parser.get_agent_state(screen) == AgentState.IN_DIALOGUE:
                    dialogue_frames.append(screen)
            if len(dialogue_frames) > 0:
                # get every second dialogue frame to reduce duplicates. 
                use_dialogue_frames = []
                for dialogue_frame in dialogue_frames[::2]:
                    self._emulator.state_parser.capture_named_region(dialogue_frame, "dialogue_box_full")
                    use_dialogue_frames.append(dialogue_frame)
                ocr_texts = ocr(use_dialogue_frames)
                dialogue_message = "There was some dialogue as a result of your actions: "
                for i, text in enumerate(ocr_texts):
                    dialogue_message = dialogue_message + f"[{i+1}] {text}\n"
            if action_success == 0:
                action_success_message = "The action you took was executed fully."
            else:
                if action == MoveStepsAction:
                    if action_success == 1:
                        action_success_message = "You moved until you hit a wall, object, NPC or obstacle. If it is an object or NPC, you can now interact with it. If it is an obstacle or wall, interacting will do nothing."
                    if action_success == -1:
                        action_success_message = "You could not move in that direction at all. There is most likely an obstacle in the way."
                    if action_success == 2:
                        action_success_message = "You moved, but before you could finish your steps, you were interupted by a battle, dialogue or cutscene."
                elif action == InteractAction:
                    if action_success == -1:
                        action_success_message = "There was nothing to interact with in front of you. Make sure you are facing an object or character and are right next to it. Move into an object or NPC to face them."
                    if action_success == 1:
                        action_success_message = "Your interaction led to something."
            self.add_to_action_buffer(action, action_kwargs, action_success, action_success_message)
        final_message = ""
        if transition_states is not None:
            for transition_state in transition_states:
                if "vlm_perception" in transition_state:
                    final_message = final_message + "\nVLM Perception: " + transition_state["vlm_perception"] + "\n"
        if len(self.action_buffer) > 0:
            final_message = final_message + "\nRecent actions taken in the environment: \n"
        for i, (buffered_action, buffered_kwargs, buffered_success, buffered_message) in enumerate(self.action_buffer):
            if buffered_message == "":
                buffered_message = "executed"
            additional = ""
            if i == len(self.action_buffer) - 1:
                additional = "(previous action you took)"
            final_message = final_message + f"[{i+1}] {additional} Action: {buffered_action.__name__} with arguments {buffered_kwargs}, System Response: {buffered_message}. \n"
        if dialogue_message != "":
            final_message = final_message + dialogue_message + "\n"
        current_state = self._emulator.state_parser.get_agent_state(screen)
        if current_state == AgentState.IN_BATTLE:
            final_message = final_message + "\nYou are currently in a battle."
        elif current_state == AgentState.IN_MENU:
            final_message = final_message + "\nYou are currently in a menu."
        observation = {
            "screen": screen,
            "messages": final_message.strip()
        }
        return observation


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
        obs = self.get_observation(action=action, action_kwargs=action_kwargs, transition_states=transition_states, action_success=action_success)
        log_info(f"Messages: {obs['messages']}", self._parameters)

    def render_info(self, action=None, action_kwargs=None, transition_states=None, action_success=None):
        info = self.get_info()
        info["core"].pop("current_frame")
        info["core"].pop("passed_frames")
        log_info("State: ", self._parameters)
        log_dict(info, self._parameters)
