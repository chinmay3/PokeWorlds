from poke_worlds.utils import log_error, log_info
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction, TestAction, LocateAction, CheckInteractionAction, BattleMenuAction, PickAttackAction, MoveGridAction, SeekAction
from poke_worlds.interface.controller import Controller
from poke_worlds.interface.action import HighLevelAction
from poke_worlds.emulation.pokemon.parsers import AgentState
from typing import Dict, Any


class PokemonStateWiseController(Controller):
    ACTIONS = [MoveStepsAction, MenuAction, InteractAction, PassDialogueAction, TestAction, LocateAction, CheckInteractionAction, BattleMenuAction, PickAttackAction, MoveGridAction, SeekAction]

    def string_to_high_level_action(self, input_str):
        input_str = input_str.lower().strip()
        if "(" not in input_str or ")" not in input_str:
            return None, None # Invalid format
        action_name = input_str.split("(")[0].strip()
        action_args_str = input_str.split("(")[1].split(")")[0].strip()
        # First handle the no arg actions
        if action_name == "checkinteraction":
            return CheckInteractionAction, {}
        if action_name == "interact":
            return InteractAction, {}
        if action_name == "passdialogue":
            return PassDialogueAction, {}
        # Now handle the actions with fixed options
        if action_name == "seek":
            item = action_args_str.strip()
            if "," not in item:
                return None, None
            intent, target = item.split(",")
            intent = intent.strip()
            target = target.strip()
            return SeekAction, {"intent": intent, "target": target}
        if action_name == "locate":
            item = action_args_str.strip()
            # prefer the image_references if both match
            return LocateAction, {"target": item}
        if action_name == "battlemenu":
            option = action_args_str.strip()
            if option in ["fight", "pokemon", "bag", "run", "progress"]:
                return BattleMenuAction, {"option": option}
            else:
                return None, None
        if action_name == "pickattack":
            if not action_args_str.strip().isnumeric():
                return None, None
            option = int(action_args_str.strip())
            if option < 1 or option > 4:
                return None, None
            return PickAttackAction, {"option": option}
        if action_name == "menu":
            option = action_args_str.strip()
            if option in ["up", "down", "confirm", "back"]:
                return MenuAction, {"menu_action": option}
            else:
                return None, None
        if action_name == "move":
            x_move, y_move = action_args_str.split(",")
            x_move = x_move.strip()
            y_move = y_move.strip()
            if " " not in x_move or " " not in y_move:
                return None, None
            direction, steps = x_move.split(" ")
            direction = direction.strip()
            steps = steps.strip()
            if direction not in ["right", "left"]:
                return None, None
            if not steps.isnumeric():
                return None, None
            x_arg = int(steps) if direction == "right" else -int(steps)
            direction, steps = y_move.split(" ")
            direction = direction.strip()
            steps = steps.strip()
            direction, steps = y_move.split(" ")
            direction = direction.strip()
            steps = steps.strip()
            if direction not in ["up", "down"]:
                return None, None
            y_arg = int(steps) if direction == "up" else -int(steps)
            return MoveGridAction, {"x_steps": x_arg, "y_steps": y_arg}
        return None, None
        
    def get_action_strings(self, return_all: bool=False) -> str:
        available_actions = "Available Actions:\n"
        current_state = self._emulator.state_parser.get_agent_state(self._emulator.get_current_frame())
        all_options = set(LocateAction.image_references.keys()).union(LocateAction.pre_described_options.keys())
        locate_option_strings = ", ".join(all_options)
        free_roam_action_strings = {
            SeekAction: f"seek(<intent: extremely short string description of intent>, <{locate_option_strings}>): Move towards the nearest instance of the specified visual entity until you are right next to it and facing it. Only the entities specified in <> are valid options, anything else will return an error. ",
            #LocateAction: f"locate(<{locate_option_strings}>): Locate all instances of the specified visual entity in the current screen, and return their coordinates relative to your current position. Only the entities specified in <> are valid options, anything else will return an error. DO NOT use this action with an input that is not listed in <> (e.g. locate(pokemon) or locate(pokeball) will fail).",
            MoveGridAction: "move(<right or left> <steps: int>,<up or down> <steps: int>): Move in grid space by the specified right/left and up/down steps.",
            CheckInteractionAction: "checkinteraction(): Check if there is something to interact with in front of you.",
            InteractAction: "interact(): Interact with cell directly in front of you. Only works if there is something to interact with.",
        }
        dialogue_action_strings = {
            PassDialogueAction: "passdialogue(): Advance the dialogue by one step.",
        }
        battle_action_strings = {
            BattleMenuAction: "battlemenu(<fight, pokemon, bag, run or progress>): Navigate the battle menu to select an option. Fight to choose an attack, Pokemon to switch Pokemon, Bag to use an item, Run to attempt to flee the battle, and Progress to continue dialogue or other battle events.",
        }
        pick_attack_action_strings = {
            PickAttackAction: "pickattack(<1-4>): Select an attack option in the battle fight menu.",
        }
        menu_action_strings = {
            MenuAction: "menu(<up, down or confirm>): Navigate the game menu.", # don't let it go back. 
        }
        if return_all:
            actions = {**free_roam_action_strings, **dialogue_action_strings, **battle_action_strings, **pick_attack_action_strings, **menu_action_strings}
        else:
            if current_state == AgentState.FREE_ROAM:
                actions = free_roam_action_strings
            elif current_state == AgentState.IN_DIALOGUE:
                actions = dialogue_action_strings
            elif current_state == AgentState.IN_BATTLE:
                if self._emulator.state_parser.is_in_fight_options_menu(self._emulator.get_current_frame()):
                    actions = {**battle_action_strings, **pick_attack_action_strings}
                else:
                    actions = battle_action_strings
            elif current_state == AgentState.IN_MENU:
                actions = menu_action_strings
            else:
                log_error(f"Unknown agent state {current_state} when getting action strings.")
        for action_class, action_desc in actions.items():
            available_actions += f"- {action_desc}\n"
        return available_actions
    
    def get_action_success_message(self, action: HighLevelAction, action_kwargs: Dict[str, Any], action_success: int) -> str:
        action_success_message = ""
        if action_success == 0:
            action_success_message = "Action performed."
        if action == MoveStepsAction or action == MoveGridAction:
            if action_success == 1:
                action_success_message = "You moved until you hit a wall, object, NPC or obstacle. If it is an object or NPC, you can now interact with it or run checkinteraction() to see if its interactable. If it is an obstacle or wall, interacting will do nothing."
            if action_success == -1:
                action_success_message = "You could not move in that direction at all. There is most likely an obstacle in the way. If you are walking into an object or NPC you can now interact with them or run checkinteraction() to see if its interactable. If it is an obstacle or wall, interacting will do nothing."
            if action_success == 2:
                action_success_message = "You moved, but before you could finish your steps, you were interupted by a battle, dialogue or cutscene."
        elif action == InteractAction:
            if action_success == -1:
                action_success_message = "There was nothing to interact with in front of you. Make sure you are facing an object or character and are right next to it. Move into an object or NPC to face them."
            if action_success == 1:
                action_success_message = "Your interaction led to something."
        elif action == PassDialogueAction:
            if action_success == -1:
                action_success_message = "There was no dialogue to pass through. Check the state"
        elif action == MenuAction:
            if action_success == -1:
                action_success_message = "The menu action could not be performed. Check if you are in the menu and that the action is valid."
        elif action == BattleMenuAction:
            if action_success == -1:
                action_success_message = "The battle menu action could not be performed. Check if you are in a battle and that the action is valid."
            elif action_success == 1:
                action_success = "Tried to run, but the wild pokemon was too fast and you could not escape."
            elif action_success == 2:
                action_success = "Tried to run, but you cannot run from trainer battles."
        elif action == PickAttackAction:
            if action_success == -1:
                action_success_message = "Could not pick that attack. Check if you are in the attack menu and that the attack index is valid."
            if action_success == 1:
                action_success_message = "Insufficient pp for that move. Pick another move."
        else:
            action_success_message = f"UNHANDLED CASE: action={action}, args={action_kwargs}, action_success={action_success}"
        return action_success_message


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
