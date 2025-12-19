from poke_worlds.emulation.parser import StateParser
from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import StateTracker
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser

class EmptyTracker(StateTracker):
    """ A tracker that does nothing. Used as a placeholder and for debugging. """

    def __init__(self, name: str, session_name: str, instance_id: str, state_parser: PokemonStateParser, parameters: dict):
        super().__init__(name, session_name, instance_id, state_parser, parameters)
        self.metrics["current_state"] = None
    
    def reset(self):
        """ Called once per environment reset. """
        self.metrics["current_state"] = None

    def step(self):
        """ Called once per environment step. """
        current_frame = self.state_parser.get_current_frame()
        new_state = self.state_parser.get_agent_state(current_frame)
        if self.metrics["current_state"] != new_state:
            previous_state = self.metrics["current_state"]
            self.metrics["current_state"] = new_state
            log_info(f"State changed from {previous_state} to: {self.metrics['current_state']}", self._parameters)


    def close(self):
        """ Called once when the environment is closed to finalize any tracked metrics. """
        pass