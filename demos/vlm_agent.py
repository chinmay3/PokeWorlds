from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, LowLevelController, RandomPlayController, LowLevelPlayController, PokemonStateWiseController
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction
from poke_worlds.interface.action import LowLevelPlayAction
from poke_worlds.emulation.emulator import LowLevelActions
from tqdm import tqdm
from PIL import Image
import numpy as np

class VL:
    system_prompt = """
You are playing a Pokemon game, and the objective is to get as far into the game as possible. 
You will be provided with images of the game screen. Based on the current screen, output the next action to take.

The set of allowed actions are:
1. MoveSteps(direction: str, steps: int): Move the character in the specified direction ('up', 'down', 'left', 'right') for a certain number of steps (1-10). Can ONLY be used in the FREE ROAM state. Example usage: MoveSteps("up", 3)
2. Interact(): Interact with the object or character directly in front of the player. Will fail if the player is not facing the object or is even one grid space away. Can ONLY be used in the FREE ROAM state. Example usage: Interact()   
3. MenuAction(menu_action: str): Perform a menu action. The possible actions are: navigate menu options ("up", "down", "left", "right"), choose the highlighted option ("confirm"), and " go back to the previous menu or exit the menu ("exit"). Can ONLY be used when in a MENU or BATTLE state. Example usage: MenuAction("confirm")
4. PassDialogue(): Advance the dialogue or text box by pressing the confirm button. Can ONLY be used when in a DIALOGUE state. Example usage: PassDialogue()

You must respond with exactly one of the above actions in the right format. Any other action is invalid. 

First, think about what is happening in the current frame, and also consider your past actions. Make sure you are not getting stuck in a repetitive loop, and if you are, try something new to break out of it. 

You should format your action output as follows:
Input: frame image
Think: (your reasoning about the current situation). Should be extremely brief.
<action></action>

Additional Context About Game:
[PREV]
[ALLOWED]
Now, based on the current frame and the context, first think and reason about your situation. Then, output your next action in the proper format, do not forget to enclose it with action tags: <action>COMMAND</action>. 
    """
    def __init__(self, env):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.env = env
        self.actions = self.env.actions

    def infer(self, current_frame, prev_message, allowed_string=""):
        use_prompt = self.system_prompt
        use_prompt = use_prompt.replace("[ALLOWED]", allowed_string)
        use_prompt = use_prompt.replace("[PREV]", prev_message)
        current_frame = current_frame.reshape(current_frame.shape[0], current_frame.shape[1])
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(current_frame),
                    },
                    {"type": "text", "text": use_prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )   
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=356, stop_strings=["</action>"], tokenizer=self.processor.tokenizer)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        full_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(full_text[0])
        return output_text[0]
    
    def parse_validate_action(self, output_text):
        text = output_text.lower().strip()
        if "<action>" not in text or "</action>" not in text:
            return "Bad <action>", "tags wrong", False
        action_str = text.split("<action>")[1].split("</action>")[0].strip()
        if "interact" in action_str.lower():
            return InteractAction, {}, self.env._controller.is_valid(InteractAction)
        elif "movesteps" in action_str.lower():
            if action_str.count(",") != 1:
                return "Bad MoveSteps", "improper commas", False
            direction_part, int_part = action_str.split(",")
            int_part = int_part.replace(")", "").strip()
            direction_part = direction_part.replace("movesteps(", "").strip()
            if not int_part.isnumeric():
                return "Bad MoveSteps", "steps not int", False
            steps = int(int_part)
            if "right" in direction_part:
                return MoveStepsAction, {"direction": "right", "steps": steps}, self.env._controller.is_valid(MoveStepsAction, direction="right", steps=steps)
            elif "left" in direction_part:
                return MoveStepsAction, {"direction": "left", "steps": steps}, self.env._controller.is_valid(MoveStepsAction, direction="left", steps=steps)
            elif "up" in direction_part:
                return MoveStepsAction, {"direction": "up", "steps": steps}, self.env._controller.is_valid(MoveStepsAction, direction="up", steps=steps)
            elif "down" in direction_part:
                return MoveStepsAction, {"direction": "down", "steps": steps}, self.env._controller.is_valid(MoveStepsAction, direction="down", steps=steps)
            else:
                return "Bad MoveSteps", "direction not recognized", False
        elif "menuaction" in action_str.lower():
            if "up" in action_str:
                return MenuAction, {"menu_action": "up"}, self.env._controller.is_valid(MenuAction, menu_action="up")
            elif "down" in action_str:
                return MenuAction, {"menu_action": "down"}, self.env._controller.is_valid(MenuAction, menu_action="down")
            elif "left" in action_str:
                return MenuAction, {"menu_action": "left"}, self.env._controller.is_valid(MenuAction, menu_action="left")
            elif "right" in action_str:
                return MenuAction, {"menu_action": "right"}, self.env._controller.is_valid(MenuAction, menu_action="right")
            elif "confirm" in action_str or "select" in action_str:
                return MenuAction, {"menu_action": "confirm"}, self.env._controller.is_valid(MenuAction, menu_action="confirm")
            elif "exit" in action_str:
                return MenuAction, {"menu_action": "exit"}, self.env._controller.is_valid(MenuAction, menu_action="exit")
        elif "passdialogue" in action_str.lower():
            return PassDialogueAction, {}, self.env._controller.is_valid(PassDialogueAction)
        else:
            return "Unknown Action", "not recognized", False
        



    def act(self, observation):
        allowed_categories = self.env._controller.get_possibly_valid_high_level_actions()
        allowed_string = "The following action categories could possibly be valid now: "
        for ac in allowed_categories:
            allowed_string = allowed_string + f"{ac},\t"
        current_frame = observation["screen"]
        prev_message = observation["messages"]
        output_text = self.infer(current_frame, prev_message, allowed_string)
        action, action_kwargs, validated = self.parse_validate_action(output_text)
        allowed_string = allowed_string + f". You tried the action {action} with arguments {action_kwargs}. This is not valid in the current state. Assume the state is different and try again."
        max_out = 20
        counter = 0
        while not validated:
            output_text = self.infer(current_frame, prev_message, allowed_string=allowed_string)
            action, action_kwargs, validated = self.parse_validate_action(output_text)
            allowed_string = allowed_string + f"\nThen, you tried {action} with {action_kwargs}, which was also invalid. Try again. Think deeper."
            counter += 1
            if counter >= max_out:
                return None, None
        return action, action_kwargs
        
        
    

environment = get_pokemon_environment(game_variant="pokemon_red", controller=PokemonStateWiseController(), 
                                      environment_variant="high_level",
                                      save_video=True,
                                        init_state="starter", session_name="high_level", headless=True)
vl = VL(environment)
steps = 0
#max_steps = 10_000
max_steps = 10
pbar = tqdm(total=max_steps)
observation, info = environment.reset()
while steps < max_steps:
    action, kwargs = vl.act(observation)
    if action is None:
        print("VL agent failed to produce a valid action after multiple attempts. Exiting.")
        break
    observation, reward, terminated, truncated, info = environment.step_high_level_action(action, **kwargs)
    if terminated or truncated:
        break
    steps += 1
    pbar.update(1)
pbar.close()    
environment.close()
# Plot rewards over time