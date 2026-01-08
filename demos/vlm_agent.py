from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, PokemonStateWiseController
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction, MoveGridAction, BattleMenuAction, PickAttackAction, CheckInteractionAction, LocateAction, SeekAction
from poke_worlds.utils import load_parameters
from poke_worlds.utils.vlm import HuggingFaceVLM
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import click
from poke_worlds.execution.pokemon.executors import PokemonExecutor
from poke_worlds.execution.pokemon.reports import PokemonExecutionReport




class VL:
    supervisor_start_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION]. You are to create a plan for a player agent to follow to achieve this mission.
The player agent can take the following actions (depending on the situation): [ALLOWED_ACTIONS]
Given the screen state of the game, come up with a high-level plan, with multiple steps to achieve this mission.
Then, craft a single, concise note for your player agent to follow to achieve the first step of this plan.
For each step, give a simple description of the step and a criteria for completion.
Format your response as:
High Level Plan: 
1. <STEP ONE>
[SEP]
2. <STEP TWO>
[SEP]
...
Note: <NOTE OR GUIDE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
    """
    def __init__(self, env, mission, model_name = f"Qwen/Qwen3-VL-32B-Instruct", max_steps_per_goal=15, max_action_attempts=3, max_total_steps=100):
        backbone_model_name = load_parameters()['backbone_vlm_model']
        HuggingFaceVLM.start()
        if backbone_model_name == model_name: # then load just one model
            self.model = HuggingFaceVLM._MODEL
            self.processor = HuggingFaceVLM._PROCESSOR
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
        self.env = env
        self.mission = mission
        self.actions = self.env.actions
        self.high_level_plan = None
        self.current_step_index = 0
        self.note = None
        self.agent_note = None
        self.max_steps_per_goal = max_steps_per_goal
        self.steps_taken_for_current_goal = 0
        self.agent_note_buffer = []
        self.observation = None
        self.max_action_attempts = max_action_attempts
        self.max_total_steps = max_total_steps
        self.prev_action = None
        self.all_inputs = []
        self.all_outputs = []

    def save_logs(self):
        path = self.env._emulator.session_path + "/vlm_logs.csv"
        df = pd.DataFrame({"inputs": self.all_inputs, "outputs": self.all_outputs})
        df.to_csv(path, index=False)
        return
        
    def infer(self, prompt, current_frame):
        current_frame = current_frame.reshape(current_frame.shape[0], current_frame.shape[1])
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(current_frame),
                    },
                    {"type": "text", "text": prompt},
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=500, stop_strings=["[STOP]"], tokenizer=self.processor.tokenizer, 
                                            do_sample=True, 
                                            temperature=0.9)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        full_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self.all_inputs.append(prompt)
        self.all_outputs.append(output_text)
        #self.save_logs()
        return output_text
    
    def parse_plan_steps(self, plan_text):
        steps = []
        for line in plan_text.lower().replace("high level plan:", "").split("[sep]"):
            line = line.strip()
            if line != "":
                steps.append(line.strip())
        return steps

    def parse_supervisor_start(self, output_text):
        if "note:" in output_text.lower():
            high_level_plan, note = output_text.lower().split("note:")
            steps = self.parse_plan_steps(high_level_plan)
            return steps, note.strip()
        else:
            print("Failed to parse supervisor start output: ", output_text)
            return None, None
            
    def do_supervisor_start(self):
        allowed_actions = self.env._controller.get_action_strings(return_all=True)
        allowed_actions_str = ""
        for class_name, action_str in allowed_actions.items():
            allowed_actions_str += f"{action_str}\n"
        prompt = self.supervisor_start_prompt.replace("[MISSION]", self.mission).replace("[ALLOWED_ACTIONS]", allowed_actions_str)
        current_frame = self.env.get_observation()["screen"]
        output_text = self.infer(prompt, current_frame)
        steps, note = self.parse_supervisor_start(output_text)
        self.high_level_plan = steps
        self.note = note
        return note
                
    
    def play(self):
        # First, get initial note from supervisor
        note = self.do_supervisor_start()
        print(f"Starting high level plan: {self.high_level_plan}")
        print(f"Starting note for first step: {note}")
        # game: str, environment: Environment, execution_report_class: Type[ExecutionReport], high_level_goal: str, immediate_task: str, initial_plan: str, visual_context: str, exit_conditions: List[str], 
        executor = PokemonExecutor(game="Pokemon", environment=self.env, 
                                   execution_report_class=PokemonExecutionReport, 
                                   high_level_goal=self.mission + f" with the overall steps being: {self.high_level_plan}", 
                                   immediate_task=self.high_level_plan[0], initial_plan=note, 
                                   visual_context="You are in Professor Oaks Lab along with Blue, your rival. There are 3 pokeballs on the bench to your right.", 
                                   exit_conditions=[])
        execution_report = executor.execute(step_limit=20, show_progress=True)
        print("Execution finished.")
        self.env.close()

        

        
        
@click.command()
@click.option("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", type=str)
def do(model_name):
    short_model = model_name.split("/")[-1]
    environment = get_pokemon_environment(game_variant="pokemon_red", controller=PokemonStateWiseController(), 
                                        environment_variant="high_level",
                                        save_video=True,
                                            init_state="starter", session_name=f"high_level_{short_model}", headless=True)
    mission = "Seek and select any one pokeball with a starter from the bench to your right, and then leave the building from the entrance below. HINT: You should typically try seek before relying on manual movement."
    vl = VL(environment, mission, model_name=model_name)
    vl.play()

# Plot rewards over time

if __name__ == "__main__":
    do()
