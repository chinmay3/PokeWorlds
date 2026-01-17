from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from poke_worlds import AVAILABLE_GAMES, get_environment
from poke_worlds.utils import load_parameters
from poke_worlds.execution.vlm import ExecutorVLM
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import click
from poke_worlds.execution.pokemon.executors import PokemonExecutor
from poke_worlds.execution.pokemon.reports import PokemonExecutionReport




class VL:
    supervisor_visual_context_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION]. 
Given the screen state of the game, come up with a brief and concise description of the current visual and game context that covers the most important details relevant to the mission and plan, while ignoring the irrelevant details.
Context: 
"""

    supervisor_start_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION]. You are to create a plan for a player agent to follow to achieve this mission.
The player agent can take the following actions (depending on the situation): [ALLOWED_ACTIONS]
The screen state of the game is given to you and described to you as [VISUAL_CONTEXT]. Now, come up with a high-level plan, with multiple steps to achieve this mission.
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

    supervisor_common_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION], with the allowed actions being [ALLOWED_ACTIONS]. For this, you developed the following high level plan:
[HIGH_LEVEL_PLAN]

"""

    executor_return_analysis_prompt = f"""
{supervisor_common_prompt}

As you played the game, you learned the lessons: [LESSONS_LEARNED]

You have now attempted to execute the following immediate task from this plan: [IMMEDIATE_TASK]

You took the following actions, and observed the following changes in the game:
[ACTION_AND_OBSERVATIONS_LOG]

Based on this, identify what deeper lessons you can learn about the game, and the executor, that will help you better construct tasks for the execution step. 
Answer with the following response format:
Analysis of Execution: <YOUR ANALYSIS OF WHAT HAPPENED, was execution successful or not, what went wrong if not>
Lessons Learned: <REFINE THE LESSONS YOU HAVE ALREADY LEARNED AND COMBINE IT WITH NEW LESSONS FROM THE ANALYSIS. BE BRIEF AND CONCISE>
Current Visual Context: <A BRIEF DESCRIPTION OF THE CURRENT VISUAL AND GAME CONTEXT, WITH THE MISSION, PREVIOUS ACTIONS AND PLAN IN MIND>
Mission Complete: Yes / No
    """

    executor_information_construction_prompt = f"""
{supervisor_common_prompt}

You have learned the following lessons so far: [LESSONS_LEARNED]

After taking some actions, you are now in the context: [VISUAL_CONTEXT]

Based on this, come up with an initial roadmap or low level plan for the executor to follow to achieve the next immediate task.
Format your response as:
IMMEDIATE TASK: <THE IMMEDIATE TASK TO ACHIEVE FOR THE MISSION> Must be a short, directly actionable task from the visual context and available actions. 
Plan: <YOUR PLAN FOR THE EXECUTOR TO FOLLOW TO ACHIEVE THE IMMEDIATE TASK>
"""
    def __init__(self, env, model_name = f"Qwen/Qwen3-VL-8B-Instruct", max_steps_per_executor=15):
        backbone_model_name = load_parameters()['executor_vlm_model']
        ExecutorVLM.start()
        if backbone_model_name == model_name: # then load just one model
            self.model = ExecutorVLM._MODEL
            self.processor = ExecutorVLM._PROCESSOR
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
        self.env = env
        self.actions = self.env.actions
        self.high_level_plan = None
        self.current_step_index = 0
        self.max_steps_per_executor = max_steps_per_executor
        self.steps_taken_for_current_goal = 0
        self.observation = None
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
        self.save_logs()
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
            
    
    def do_initial_visual_context(self):
        prompt = self.supervisor_visual_context_prompt.replace("[MISSION]", self.mission)
        current_frame = self.env.get_info()["core"]["current_frame"]
        output_text = self.infer(prompt, current_frame)
        return output_text.strip()

    
    def do_supervisor_start(self, visual_context):
        allowed_actions = self.env._controller.get_action_strings(return_all=True)
        allowed_actions_str = ""
        for class_name, action_str in allowed_actions.items():
            allowed_actions_str += f"{action_str}\n"
        prompt = self.supervisor_start_prompt.replace("[MISSION]", self.mission).replace("[ALLOWED_ACTIONS]", allowed_actions_str).replace("[VISUAL_CONTEXT]", visual_context)
        current_frame = self.env.get_info()["core"]["current_frame"]
        output_text = self.infer(prompt, current_frame)
        steps, note = self.parse_supervisor_start(output_text)
        self.high_level_plan = steps
        self.executor_return_analysis_prompt = self.executor_return_analysis_prompt.replace("[MISSION]", self.mission).replace("[ALLOWED_ACTIONS]", allowed_actions_str).replace("[HIGH_LEVEL_PLAN]", "\n".join(self.high_level_plan))
        self.executor_information_construction_prompt = self.executor_information_construction_prompt.replace("[MISSION]", self.mission).replace("[ALLOWED_ACTIONS]", allowed_actions_str).replace("[HIGH_LEVEL_PLAN]", "\n".join(self.high_level_plan))
        return note
    
    def parse_executor_return_analysis(self, output_text):
        analysis = None
        lessons_learned = None
        current_visual_context = None
        mission_complete = False
        output_text = output_text.lower()
        if output_text.count("lessons learned:") != 1:
            print("Failed to parse executor return analysis output: ", output_text)
            return None, None, None, False
        analysis_part, rest = output_text.split("lessons learned:")
        analysis = analysis_part.replace("analysis of execution:", "").strip()
        if rest.count("current visual context:") != 1:
            print("Failed to parse executor return analysis output: ", output_text)
            return analysis, None, None, False
        lessons_part, rest2 = rest.split("current visual context:")
        lessons_learned = lessons_part.strip()
        if rest2.count("mission complete:") != 1:
            print("Failed to parse executor return analysis output: ", output_text)
            return analysis, lessons_learned, None, False
        context_part, mission_part = rest2.split("mission complete:")
        current_visual_context = context_part.strip()
        mission_complete = "yes" in mission_part.lower()
        return analysis, lessons_learned, current_visual_context, mission_complete
    
    def parse_executor_information_construction(self, output_text):
        output_text = output_text.lower()
        if output_text.count("plan:") != 1:
            print("Failed to parse executor information construction output: ", output_text)
            return None, None
        task_part, plan_part = output_text.split("plan:")
        immediate_task = task_part.replace("immediate task:", "").strip()
        plan = plan_part.strip()
        return immediate_task, plan
                
    
    def play(self, mission: str, visual_context=None):
        assert mission is not None, "Mission must be provided to play()."
        self.mission = mission
        # First, get initial note from supervisor
        if visual_context is None:
            visual_context = self.do_initial_visual_context()
            print("Inferred initial visual context: ", visual_context)
        note = self.do_supervisor_start(visual_context)
        high_level_plan_str = "\n".join(self.high_level_plan)
        print(f"Starting high level plan: {high_level_plan_str}")
        print(f"Starting note for first step: {note}")
        initial_plan = note
        lessons_learned = "No Lessons Learned Yet."
        immediate_task = self.high_level_plan[0]
        mission_accomplished = False
        pbar = tqdm(total=self.env._emulator.max_steps, desc="Overall VLM Agent Progress")
        while not mission_accomplished:
            print(f"Starting execution of immediate task: {immediate_task} with initial_plan: {initial_plan}")
            executor = PokemonExecutor(game="Pokemon", environment=self.env, 
                                    execution_report_class=PokemonExecutionReport, 
                                    high_level_goal=self.mission,
                                    task=immediate_task, initial_plan=initial_plan, 
                                    visual_context=visual_context, 
                                    exit_conditions=[])
            execution_report = executor.execute(step_limit=self.max_steps_per_executor, show_progress=True)
            actions_and_observations = "\n".join(execution_report.get_execution_summary())
            prompt = self.executor_return_analysis_prompt.replace("[LESSONS_LEARNED]", lessons_learned).replace("[IMMEDIATE_TASK]", immediate_task).replace("[ACTION_AND_OBSERVATIONS_LOG]", actions_and_observations)
            current_frame = self.env.get_info()["core"]["current_frame"]
            output_text = self.infer(prompt, current_frame)
            analysis, lessons_learned, visual_context, mission_accomplished = self.parse_executor_return_analysis(output_text)
            if visual_context is None:
                break
            if mission_accomplished:
                print("Mission accomplished!")
                break
            print(f"Executor Analysis: {analysis}")
            print(f"Updated Lessons Learned: {lessons_learned}")
            print(f"Updated Visual Context: {visual_context}")
            if execution_report.exit_code == 1:
                print("Environment Steps Done")
                break
            prompt = self.executor_information_construction_prompt.replace("[LESSONS_LEARNED]", lessons_learned).replace("[VISUAL_CONTEXT]", visual_context)
            current_frame = self.env.get_info()["core"]["current_frame"]
            output_text = self.infer(prompt, current_frame)
            immediate_task, initial_plan = self.parse_executor_information_construction(output_text)
            if immediate_task is None or initial_plan is None:
                break
            pbar.update(1)
        print("Finished playing VLM agent.")
        self.env.close()

        

        
        
@click.command()
@click.option("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", type=str)
@click.option("--init_state", default="starter", type=str)
@click.option("--game_variant", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--mission", default="Professor oak has invited you into his lab and offered you a choice of starter pokemon from the bench. You are to select a starter, leave the building from the bottom, and keep playing until you get the first gym badge.", type=str)
@click.option("--visual_context", default=None, type=str)
@click.option("--max_steps", default=1000, type=int)
def do(model_name, init_state, game_variant, mission, visual_context, max_steps):
    short_model = model_name.split("/")[-1]
    environment = get_environment(game=game_variant, environment_variant="default", controller_variant="state_wise", 
                                        save_video=True, max_steps=max_steps,
                                            init_state=init_state, session_name=f"vlm_demo_{short_model}", headless=True)
    vl = VL(environment, model_name=model_name)
    vl.play(mission=mission, visual_context=visual_context)

# Plot rewards over time

if __name__ == "__main__":
    do()
