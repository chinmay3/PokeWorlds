from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController
from poke_worlds.interface import HighLevelAction
from poke_worlds.execution.executor import Executor
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction, MoveGridAction, BattleMenuAction, PickAttackAction, CheckInteractionAction, LocateAction, SeekAction


class PokemonExecutor(Executor):
    REQUIRED_CONTROLLER = PokemonStateWiseController

    def get_action_message(self, *, action: HighLevelAction, action_kwargs: dict, action_success: int, action_return: dict, last_action_hint: bool=False):
        interaction_advice = f"If you want to try to interact with it, use interact() or checkinteraction() to get an estimate on whether or not you can interact with it."
        path_blocked_advice = f"If you are trying to go somewhere, then this direction is blocked. Try moving around it or going a different way."
        locate_advice = f"You may need to move around, or trust your visual intuition from the screen."
        if last_action_hint is False:
            interaction_advice = ""
            path_blocked_advice = ""
            locate_advice = ""
        action_success_message = ""
        if action == MoveStepsAction or action == MoveGridAction:
            if action_success == 0:
                if action_return["rotated"] == True:
                    action_success_message = f"You did not actually move, but rotated to face the direction you wanted to move in. This means there is now either an obstacle or object / NPC in front of you. {interaction_advice} {path_blocked_advice}"
                else:
                    action_success_message = "You moved the exact number of steps you wanted to move. You are now facing the target location."
            if action_success == 1:
                action_success_message = f"You moved until you hit a wall, object, NPC or obstacle. {interaction_advice} {path_blocked_advice}"
            if action_success == -1:
                action_success_message = f"You could not move in that direction at all. There is an obstacle in the way. {interaction_advice} {path_blocked_advice}"
            if action_success == 2:
                action_success_message = "You moved, but before you could finish your steps, you were interupted by a battle, dialogue or cutscene."
        elif action == CheckInteractionAction:
            percieve_output = action_return["percieve_output"]
            action_success_message = percieve_output            
        elif action == LocateAction:
            if action_success == -1:
                action_success_message = f"Could not locate the target on screen. {locate_advice}"
            else:
                potential_cells_str = action_return["potential_cells_str"]
                definitive_cells_str = action_return["definitive_cells_str"]
                action_success_message = f"Located the target on screen. Potential cells: {potential_cells_str}. Definitive cells: {definitive_cells_str}. Trust that these are accurate. You can now either move() towards one of these coordinates to interact() or do something else."
        elif action == InteractAction:
            if action_success == -1:
                action_success_message = "There was nothing to interact with in front of you. Make sure you are facing an object or character and are right next to it. Move into an object or NPC to face them."
            if action_success == 1:
                action_success_message = "Your interaction led to something."
        elif action == SeekAction:
            if action_success == 0:
                action_success_message = "Success! You sought out your target and began an interaction. Procedure through the rest of the game as required."
            if action_success == -1:
                action_success_message = f"Could not find the object on screen to seek. {locate_advice}"
            elif action_success == 1:
                action_success_message = f"Could not move to the target location because there is an obstacle completely blocking the path. {path_blocked_advice}"
            elif action_success == 2:
                action_success_message = "You were interupted before reaching the target location by a battle, dialogue or cutscene."
            elif action_success == 3:
                action_success_message = "The CheckInteraction VLM failed when trying to see if you could interact with the target at the location. This may be a VLM error. Just try running Interact()"
            elif action_success == 4:
                action_success_message = "When you reached the target location, there was nothing to interact with. Make sure you are going to the right place. You could also just try interact() to be sure. "
            elif action_success == 5:
                action_success_message = "When you tried to interact with the target at the location, the interaction failed. Make sure you are right next to and facing the object or NPC. Reposition with move() and try again."
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
