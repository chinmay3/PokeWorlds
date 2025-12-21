from poke_worlds.emulation.emulator import Emulator, LowLevelActions
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser, AgentState
from poke_worlds.emulation.pokemon.trackers import CorePokemonTracker
from poke_worlds.utils import log_error
from typing import Tuple
import numpy as np


class PokemonEmulator(Emulator):
    """
    Almost the exact same as Emulator, but forces the agent to not mess with the menu options cursor.
    Also, skips all dialogue automatically (by clicking 'B'). # TODO: Make sure you never skip choices. 
    """
    REQUIRED_STATE_PARSER = PokemonStateParser
    REQUIRED_STATE_TRACKER = CorePokemonTracker
    _MAXIMUM_DIALOGUE_PRESSES = 2000 # For now set a crazy high value
    """ Maximum number of times the agent will click B to get through a dialogue. """

    def step(self, action) -> Tuple[np.ndarray, bool]:
        frames, done = super().step(action)
        self.state_parser: PokemonStateParser
        if self.state_parser.is_hovering_over_options_in_menu(self.get_current_frame()):
            # force the agent to click the up button to get off the options
            self.run_action_on_emulator(LowLevelActions.PRESS_ARROW_UP)
        current_state = self.state_parser.get_agent_state
        all_next_frames = []
        n_clicks = 0
        # Clicks through any dialogue popups. 
        while (n_clicks < self._MAXIMUM_DIALOGUE_PRESSES) and current_state == AgentState.IN_DIALOGUE:
            next_frames = self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_B)
            all_next_frames.append(next_frames)
            n_clicks += 1
        if len(all_next_frames) != 0:
            all_next_frames = np.stack(all_next_frames)
            frames = np.concatenate([frames, all_next_frames]) # TODO: Check
        return frames, done
    
    def run_action_on_emulator(self, *args, **kwargs):
        if not kwargs.get("render", True):
            log_error(f"PokemonEmulator requires render to be True", self._parameters)
        return super().run_action_on_emulator(*args, **kwargs)
