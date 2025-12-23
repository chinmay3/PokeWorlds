from poke_worlds.utils import log_error, log_info
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction
from poke_worlds.interface.controller import Controller


class PokemonStateWiseController(Controller):
    ACTIONS = [MoveStepsAction, MenuAction, InteractAction, PassDialogueAction]

    def _parse_distance(self, distance_str):
        if distance_str.count(":") != 1:
            return None, None
        direction_str, steps = distance_str.split(":")
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
            return MoveStepsAction, {"direction": direction, "steps": int(steps.strip())}        
            

    def string_to_high_level_action(self, input_str):
        input_str = input_str.lower().strip()
        if input_str == "a":
            return InteractAction, {}
        elif input_str == "p":
            return PassDialogueAction, {}
        if ":" in input_str:
            return self._parse_distance(input_str)
        else:
            if input_str == "m_u":
                return MenuAction, {"menu_action": "up"}
            elif input_str == "m_d":
                return MenuAction, {"menu_action": "down"}
            elif input_str == "m_a":
                return MenuAction, {"menu_action": "confirm"}
            elif input_str == "m_b":
                return MenuAction, {"menu_action": "exit"}
            elif input_str == "m_o":
                return MenuAction, {"menu_action": "open"}
            elif input_str == "m_l":
                return MenuAction, {"menu_action": "left"}
            elif input_str == "m_r":
                return MenuAction, {"menu_action": "right"}
        return None, None

        
    def get_action_strings(self):
        msg = f"""
        <direction(u,d,r,l)>: <steps(int)> or <menu(m_u, m_d, m_r, m_l, m_a, m_b, m_o)> or <interaction(a)> or <pass_dialogue(p)>
        """
        return msg    
