from poke_worlds.utils import log_error, log_info
from poke_worlds.interface.pokemon.actions import MoveSteps
from poke_worlds.interface.controller import Controller


class PokemonTestController(Controller):
    ACTIONS = [MoveSteps]

    def string_to_high_level_action(self, input_str):
        input_str = input_str.lower()
        if input_str.count(":") != 1:
            return None, None
        direction_str, steps = input_str.split(":")
        direction = None
        if direction_str == "u":
            direction = "up"
        elif direction_str == "d":
            direction = "down"
        elif direction_str == "l":
            direction = "left"
        elif direction_str == "r":
            direction = "right"
        else:
            return None, None
        if not steps.strip().isnumeric():
            return None, None
        else:
            return MoveSteps, {"direction": direction, "steps": int(steps.strip())}
    
    def get_action_strings(self):
        msg = f"""
        <direction(u,d,r,l)>: <steps(int)>
        """
        return msg    
