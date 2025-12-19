from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import MetricGroup, StateTracker
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser, AgentState
from typing import Optional, Type
import numpy as np


class CorePokemonMetrics(MetricGroup):
    """
    Pokémon-specific metrics.
    """
    NAME = "pokemon_core"
    REQUIRED_PARSER = PokemonStateParser

    def reset(self, first = False):
        self.current_state: AgentState = AgentState.IN_DIALOGUE # Start by default in dialogue because it has the least permissable actions. 
        """ The current state of the agent in the game. """
        self._previous_state = self.current_state

    def close(self):
        pass

    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        self._previous_state = self.current_state
        current_state = self.state_parser.get_agent_state(current_frame)
        self.current_state = current_state
        if self.current_state != self._previous_state:
            log_info(f"Agent state changed from {self._previous_state} to {self.current_state}", self._parameters)

    def report(self) -> dict:
        """
        Reports the current Pokémon core metrics.
        Returns:
            dict: A dictionary containing the current agent state.
        """
        return {
            "agent_state": self.current_state.name
        }
    
    def report_final(self) -> dict:
        return {}


class CorePokemonTracker(StateTracker):
    """
    StateTracker for core Pokémon metrics.
    """
    def start(self):
        super().start()
        self.metric_classes.extend([CorePokemonMetrics])
