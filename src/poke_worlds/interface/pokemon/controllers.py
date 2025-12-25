from poke_worlds.utils import log_error, log_info
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction, TestAction, LocateAction
from poke_worlds.interface.controller import Controller


class PokemonStateWiseController(Controller):
    ACTIONS = [MoveStepsAction, MenuAction, InteractAction, PassDialogueAction, TestAction, LocateAction]

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
        elif input_str == "t":
            return TestAction, {}
        elif input_str.startswith("l"):
            return LocateAction, {"target": input_str.replace("l", "").strip()}
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
        <direction(u,d,r,l)>: <steps(int)> or <menu(m_u, m_d, m_r, m_l, m_a, m_b, m_o)> or <interaction(a)> or <pass_dialogue(p)> or <test(t)> or <locate(l)>
        """
        return msg    


class Thoughts:
    # Autobattler (simple, first check if there is an attacking move in any pokemons slots. If so and not active, switch into it. Then spam that attack)
    # This is meant to be used with nerf_opponents to allow simulation without fear of battles getting in the way. 
    # A* pathfinding towards a given coordinate. 
    # Menu: Open Items, Open Pokemon, 
    # Open Map, Exit Map
    # Throw Ball
    # Show inventory 
    # Use Specific Item (e.g. Antidote etc) (String Based) You must establish the mapping to game id and get that done
    # Use Item on Pokemon 
    # Run from Battle
    # Show other pokemon info
    # Check Pokemon Info
    # Switch to Pokemon
    # Maybe set up an OCR on the frame to catch some amount of string mapping (i.e. catch cant use that )
        # Then you can have OCR on a sign
    # Simple mappings of interact with NPC or sign. 
    # Move in direction (for a specified number of steps)
    # Try to Buy x Items
    # 
    pass
"""
Action Spaces: 
Open World, No Window open:
    Move:
        Move(direction, steps): either returns success message or failure and early exit (could be wild pokemon, could be trainer, could be obstacle). 
        Move(x, y): Try to move towards this coordinate using A* algorithm. 
    Interact: Basically press A
    Inventory:
        Search(Item Name): return [not an item or the count of item in bag]
        List items: List all items in bag
        Use(Item Name):
            For each item, perhaps have an argument input (i.e. Use Potion on [Pokemon Name])
    Pokemon:
        List: List all Pokemon
        Check(Pokemon Name)
        Check_all
        SwitchFirst(Pokemon): Switches a new pokemon to the first slot in the party
    Fly: (If there is a fly pokemon in inventory)


Battle:
    Fight:
        Attack(name): either says attack not found or uses the attack
    Inventory:
        List
        Search(Item Name)
        Use(Item Name)
    Throw Ball(Ball Kind)
    Pokemon:
        List
        Check(Pokemon Name)
        Switch(Pokemon Name)
    Run

Conversation: (Hard?)
    Continue

Menu / Options: (HARD)
    Select specific option

Manual:
    All controls (other than Enter)    
"""
