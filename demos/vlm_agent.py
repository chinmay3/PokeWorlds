from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, LowLevelController, RandomPlayController, LowLevelPlayController
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
- UP : Will either move the character up or navigate up in a menu.
- DOWN : Will either move the character down or navigate down in a menu.
- LEFT : Will either move the character left or navigate left in a menu.
- RIGHT : Will either move the character right or navigate right in a menu.
- A : Press the A button to interact, confirm, or select. You can only use this action to interact with an NPC or object if they are directly in front of you. Otherwise, use movement actions to position yourself.
- B : Press the B button to cancel or go back.
You must respond with exactly one of the above actions. Any other action is invalid. 

First, think about what is happening in the current frame, then decide on the best action to take next.

You should format your action output as follows:
Input: frame image
Think: (your reasoning about the current situation). Should be extremely brief.
<action>(one of UP, DOWN, LEFT, RIGHT, A, B)</action>

[PREV]
    """
    def __init__(self, env):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        self.env = env
        self.actions = self.env.actions

    def infer(self, current_frame, info):
        use_prompt = self.system_prompt
        if info.get("prev_action") is None:
            use_prompt = use_prompt.replace("[PREV]", "")
        else:
            prev_act = info["prev_action"]
            if info["core"]["frame_changed"]:
                use_prompt = use_prompt.replace("[PREV]", f"Previous action was {prev_act}. The frame changed to this after the action.")
            else:
                use_prompt = use_prompt.replace("[PREV]", f"Previous action was {prev_act}. The frame did not change after the action. This likely means the action was invalid or does nothing in this context.")
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, stop_strings=["</action>"], tokenizer=self.processor.tokenizer)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def act(self, current_frame):
        output_text = self.infer(current_frame)
        # parse the output text into an action here:
        action_c = self.actions[0] # there's only one action here. 
        kwargs = {} # "action key needs to map to a low level action"
        valid_actions = [LowLevelActions.PRESS_ARROW_DOWN, LowLevelActions.PRESS_ARROW_LEFT, LowLevelActions.PRESS_ARROW_UP, LowLevelActions.PRESS_ARROW_RIGHT, LowLevelActions.PRESS_BUTTON_A, LowLevelActions.PRESS_BUTTON_B]
        if "<action>" in output_text and "</action>" in output_text:
            action = output_text.split("<action>")[1].split("</action>")[0].strip()
            if "up" in action.strip().lower():
                kwargs["low_level_action"] = LowLevelActions.PRESS_ARROW_UP
            elif "down" in action.strip().lower():
                kwargs["low_level_action"] = LowLevelActions.PRESS_ARROW_DOWN
            elif "right" in action.strip().lower():
                kwargs["low_level_action"] = LowLevelActions.PRESS_ARROW_RIGHT
            elif "left" in action.strip().lower():
                kwargs["low_level_action"] = LowLevelActions.PRESS_ARROW_LEFT
            elif "a" in action.strip().lower():
                kwargs["low_level_action"] = LowLevelActions.PRESS_BUTTON_A
            elif "b" in action.strip().lower():
                kwargs["low_level_action"] = LowLevelActions.PRESS_BUTTON_B
            else:
                kwargs["low_level_action"] = None
        else:
            kwargs["low_level_action"] = None
        if kwargs["low_level_action"] is None:
            # pick random action
            print("Random Action")
            kwargs["low_level_action"] = np.random.choice(valid_actions)
        print(f"Chosen action: {kwargs['low_level_action']} from model output: {output_text}")
        return action_c, kwargs
    

environment = get_pokemon_environment(game_variant="pokemon_red", controller=LowLevelPlayController(), save_video=True,
                                        init_state="starter", max_steps=1000, headless=True)
vl = VL(environment)
steps = 0
max_steps = 500
pbar = tqdm(total=max_steps)
observation, info = environment.reset()
while steps < max_steps:
    action, kwargs = vl.act(observation, info)
    observation, reward, terminated, truncated, info = environment.step_high_level_action(action, **kwargs)
    if terminated or truncated:
        break
    steps += 1
    pbar.update(1)
pbar.close()    
environment.close()
# Plot rewards over time