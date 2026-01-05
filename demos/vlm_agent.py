from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, LowLevelController, RandomPlayController, LowLevelPlayController, PokemonStateWiseController
from poke_worlds.interface.pokemon.actions import MoveStepsAction, MenuAction, InteractAction, PassDialogueAction
from poke_worlds.interface.action import LowLevelPlayAction
from poke_worlds.emulation.emulator import LowLevelActions
from poke_worlds.utils import load_parameters
from poke_worlds.utils.vlm import HuggingFaceVLM
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import click


class VL:
    system_prompt = """
You are playing a Pokemon game and you are trying to achieve the mission [STEP]. 
You left yourself the following note for this mission: [NOTE]
    """

    prev_information_prompt = """
Previous Actions and Results:
[ACTION_BUFFER]
Previous Thinking and Notes:
[NOTE_BUFFER]
Previous Screen Messages from OCR:
[OCR_BUFFER]
Message From Most Recent Observation:
[OCR]
    """

    action_instruction_prompt = """
Your instruction is to:
1. Given the results of your previous actions and the state of the current screen, decide whether your mission is complete or not.
2. If complete, give a concise summary of the important dialogues you saw on the way, and any important lessons learned.
2. If not complete, reflect on your previous actions and notes. Understand whether you are moving closer to your goal, or whether a change of strategy is needed.
3. If not complete, select an action you will perform.


You should format your action output as follows:
Mission Completed: <Yes/No>. 
If Yes: 
    Learnings: <CONCISE SUMMARY OF LESSONS LEARNED AND IMPORTANT DIALOGUES>.
If No: 
    Critique of Previous Action Failures: From the results of your previous actions, have you been moving closer to your immediate goal? If not, why do you think that is? What can you do differently this time to improve your chances of success?
    Note: Update your note with important information that will help guide you to your mission. Has your context changed from the previous note? What should your overall next step be? Have you learned anything from the dialogues or screens that you should track. 
    Action: <action>SELECTED ACTION COMMAND</action>. Current state is [STATE]. The allowed actions for this state are:\n[ALLOWED_ACTIONS]. 

Now, based on the current frame and the context, respond in the proper format:
    """

    failure_summary_prompt = """
However, you have been trying to achieve this goal, and have not done so for too long. 
Your job is to now, given the results of your previous actions and the state of the current screen, leave a final note on what went wrong. 
Was the overall mission itself unachievable? Would you have certainly succeeded if given more time? Would a slightly different mission have been more achievable? 
Provide a concise summary of what went wrong and any suggestions for future attempts.
    """

    full_action_prompt = f"""
{system_prompt}

Information from prior steps:
{prev_information_prompt}

{action_instruction_prompt}
"""
    
    full_failure_summary_prompt = f"""
{system_prompt}
Information from prior steps:
{prev_information_prompt}

{failure_summary_prompt}
    """

    supervisor_start_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION]. You are to create a plan for a player agent to follow to achieve this mission.
The player agent can take the following actions (depending on the situation): [ALLOWED_ACTIONS]
Given the screen state of the game, come up with a high-level plan, with multiple steps to achieve this mission.
Then, craft a single, concise note for your player agent to follow to achieve the first step of this plan.
Format your response as:
High Level Plan: 
1. STEP ONE (Simple description of step and Criteria for completion)
2. STEP TWO (Simple description of step and Criteria for completion)
...
Note: <CONCISE NOTE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
    """

    superviser_failure_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION] and you have come up with the following high-level plan:
[HIGH_LEVEL_PLAN]. The player agent can take the following actions (depending on the situation): [ALLOWED_ACTIONS]
Your agent has been trying to achieve [STEP_DESCRIPTION], but has not succeeded for too long.
It has left you the following message regarding its failure: 
[AGENT_MESSAGE]
Your job is to now, given the results of your agent's previous actions and the state of the current screen, revise the high-level plan if needed, and then craft a new concise note for your player agent to follow to achieve the next step of this (possibly revised) plan.
Format your response as:
Reasoning: <YOUR REASONING ABOUT WHAT WENT WRONG AND HOW TO FIX IT>
Revised High Level Plan:
1. STEP ONE (Simple description of step and Criteria for completion) DO NOT REPEAT STEPS THAT HAVE ALREADY BEEN COMPLETED. Start at what needs to be done next.
2. STEP TWO (Simple description of step and Criteria for completion)
...
Note: <CONCISE NOTE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
    """

    superviser_success_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION] and you have come up with the following high-level plan:
[HIGH_LEVEL_PLAN]
Your agent has successfully achieved [STEP_DESCRIPTION] and has left you with the following message: [AGENT_MESSAGE]
Your job is to now, given the results of your agent's previous actions and the state of the current screen, craft a new concise note for your player agent to follow to achieve the next step of this plan.
Format your response as:
Note: <CONCISE NOTE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE NEXT STEP>
    """


    def __init__(self, env, mission, size=32, max_steps_per_goal=15, max_action_attempts=3, max_total_steps=100):
        model_name = f"Qwen/Qwen3-VL-{size}B-Instruct"
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=500, stop_strings=["</action>"], tokenizer=self.processor.tokenizer, 
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
        print(full_text)
        return output_text
    
    def parse_supervisor_start(self, output_text):
        if "note:" in output_text.lower():
            high_level_plan, note = output_text.lower().split("note:").strip()
            # split high level plan into steps
            steps = []
            for line in high_level_plan.split("\n"):
                if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                    steps.append(line.strip())
            return steps, note.strip()
        else:
            print("Failed to parse supervisor start output: ", output_text)
            return None, None
        
    def parse_supervisor_failure(self, output_text):
        if "note:" in output_text.lower():
            if "revised high level plan:" in output_text.lower():
                reasoning, rest = output_text.lower().split("revised high level plan:")
                high_level_plan, note = rest.split("note:")
                # split high level plan into steps
                steps = []
                for line in high_level_plan.split("\n"):
                    if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                        steps.append(line.strip())
                return steps, note.strip()
            else:
                note = output_text.lower().split("note:")[-1].strip()
                return None, note
        else:
            print("Failed to parse supervisor failure output: ", output_text)
            return None, None

    def parse_supervisor_success(self, output_text):
        if "note:" in output_text.lower():
            note = output_text.lower().split("note:")[-1].strip()
            return note
        else:
            print("Failed to parse supervisor success output: ", output_text)
            return None
        
    def parse_failure_summary(self, output_text):
        return output_text.strip()
    
    def parse_action_instruction(self, output_text):
        output_text = output_text.strip().lower()
        if "mission complete: yes" in output_text:
            mission_complete = True
            learnings = output_text.lower().split("learnings:")[-1].strip()
            return mission_complete, learnings
        else: # assume no
            mission_complete = False
            if "critique" in output_text:
                remaining = output_text.split("critique")[-1]
            else:
                print(f"Warning: 'critique' not found in action instruction output. Output was: {output_text}")
                return mission_complete, None
            if "note:" in remaining:
                critique, rest = remaining.split("note:")
                if "action:" in rest:
                    note, action_part = rest.split("action:")
                    action = action_part.strip().split()[0]
                    return mission_complete, (critique.strip(), note.strip(), action.strip())
                else:
                    return mission_complete, None
            else:
                print(f"Warning: 'note:' not found in action instruction output. Output was: {output_text}")
                return mission_complete, None
            
    def do_supervisor_start(self):
        allowed_actions = self.env._controller.get_action_strings(return_all=True)
        prompt = self.supervisor_start_prompt.replace("[MISSION]", self.mission).replace("[ALLOWED_ACTIONS]", allowed_actions)
        current_frame = self.env.get_observation()["screen"]
        output_text = self.infer(prompt, current_frame)
        steps, note = self.parse_supervisor_start(output_text)
        self.high_level_plan = steps
        self.note = note
        return note
    
    def do_supervisor_failure(self, agent_message):
        note = self.note
        assert self.high_level_plan is not None, "High level plan must be set before calling supervisor failure."
        assert note is not None, "Note must be set before calling supervisor failure."
        step_description = self.high_level_plan[self.current_step_index] + " with note " + note
        if self.current_step_index > 0:
            step_description = f"Completed until step {self.current_step_index+1}. Now trying to achieve: {step_description}"
        allowed_actions = self.env._controller.get_action_strings(return_all=True)
        high_level_plan_str = "\n".join(self.high_level_plan)
        prompt = self.superviser_failure_prompt.replace("[MISSION]", self.mission).replace("[HIGH_LEVEL_PLAN]", high_level_plan_str).replace("[STEP_DESCRIPTION]", step_description).replace("[AGENT_MESSAGE]", agent_message).replace("[ALLOWED_ACTIONS]", allowed_actions)
        current_frame = self.env.get_observation()["screen"]
        output_text = self.infer(prompt, current_frame)
        steps, note = self.parse_supervisor_failure(output_text)
        if steps is not None:
            self.high_level_plan = steps
        self.current_step_index = 0 # Reset to Step 1 of new plan
        self.note = note
        self.steps_taken_for_current_goal = 0
        self.agent_note_buffer = []
        return note
    
    def do_supervisor_success(self, agent_message):
        assert self.high_level_plan is not None, "High level plan must be set before calling supervisor success."
        note = self.note
        assert note is not None, "Note must be set before calling supervisor success."
        step_description = self.high_level_plan[self.current_step_index] + " with note " + note
        high_level_plan_str = "\n".join(self.high_level_plan)
        prompt = self.superviser_success_prompt.replace("[MISSION]", self.mission).replace("[HIGH_LEVEL_PLAN]", high_level_plan_str).replace("[STEP_DESCRIPTION]", step_description).replace("[AGENT_MESSAGE]", agent_message)
        current_frame = self.env.get_observation()["screen"]
        output_text = self.infer(prompt, current_frame)
        note = self.parse_supervisor_success(output_text)
        self.current_step_index += 1
        self.note = note
        self.steps_taken_for_current_goal = 0
        self.agent_note_buffer = []
        return note
    
    def handle_action(self):
        assert self.note is not None, "Note must be set before calling handle_action."
        assert self.high_level_plan is not None, "High level plan must be set before calling handle_action."
        if self.observation is None:
            observation = self.env.get_observation()
        else:
            observation = self.observation
        allowed_actions = self.env._controller.get_action_strings(return_all=False)
        action_buffer_str = observation["action_buffer"]
        if action_buffer_str == "": # Force empty action buffer on first step
            action_buffer_str = "No previous actions taken."
        screen = observation["screen"]
        recent_ocr = observation["ocr"]
        if recent_ocr == "":
            recent_ocr = "No new messages."
        state = observation["state"]
        past_ocr = self.env.ocr_buffer
        ocr_buffer_str = ""
        for ocr_step, ocr_texts in past_ocr:
            ocr_text_combined = ""
            for kind, text in ocr_texts.items():
                ocr_text_combined += f"{kind}: {text} | "
            ocr_buffer_str += f"Step {ocr_step}: {ocr_text_combined}\n"
        if ocr_buffer_str == "":
            ocr_buffer_str = "No previous OCR messages."
        if self.steps_taken_for_current_goal > self.max_steps_per_goal:
            # Need to fail and ask supervisor for new note
            failure_summary_prompt = self.failure_summary_prompt
            prompt = self.full_failure_summary_prompt.replace("[STEP]", self.high_level_plan[self.current_step_index]).replace("[NOTE]", self.note).replace("[ACTION_BUFFER]", action_buffer_str).replace("[NOTE_BUFFER]", "\n".join(self.agent_note_buffer)).replace("[OCR_BUFFER]", ocr_buffer_str).replace("[OCR]", recent_ocr)
            failure_summary = self.infer(prompt, screen)
            return "fail", failure_summary
        else:
            self.steps_taken_for_current_goal += 1
            prompt = self.full_action_prompt.replace("[STEP]", self.high_level_plan[self.current_step_index]).replace("[NOTE]", self.note).replace("[ACTION_BUFFER]", action_buffer_str).replace("[NOTE_BUFFER]", "\n".join(self.agent_note_buffer)).replace("[OCR_BUFFER]", ocr_buffer_str).replace("[OCR]", recent_ocr).replace("[STATE]", state).replace("[ALLOWED_ACTIONS]", allowed_actions)
            action_valid = False
            previous_action_strings = ""
            n_attempts = 0
            while not action_valid and n_attempts < self.max_action_attempts:
                prompt = prompt + f"{previous_action_strings}"
                output_text = self.infer(prompt, screen)
                n_attempts += 1
                mission_complete, result = self.parse_action_instruction(output_text)
                if mission_complete:
                    learnings = result
                    return "success", learnings
                else:
                    critique, updated_note, action_str = result
                    self.agent_note_buffer.append(f"Critique: {critique}\nUpdated Note: {updated_note}")
                    # take action on environment and 
                    possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info = self.env.step_str(action_str)
                    if possible_obs is not None:
                        observation, reward, terminated, truncated, info = possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info
                        action, action_kwargs, transition_states, action_success = info["core"]["previous_action_details"]
                        self.observation = observation
                        done = terminated or truncated
                        return "continue", done
                    else:
                        # invalid action, try again
                        previous_action_strings += f"\nThe action '{action_str}' was invalid in the current state. Please choose a valid action next time."
            # If we reach here, all attempts failed
            return "invalid_action", None
                
    
    def play(self):
        # First, get initial note from supervisor
        note = self.do_supervisor_start()
        print(f"Starting high level plan: {self.high_level_plan}")
        print(f"Starting note for first step: {note}")
        done = False
        total_steps = 0
        pbar = tqdm(total=self.max_total_steps)
        while not done and total_steps < self.max_total_steps:
            total_steps += 1
            return_type, result = self.handle_action()
            if return_type == "success":
                learnings = result
                print(f"Step {self.current_step_index} completed! Learnings: {learnings}")
                if self.current_step_index >= len(self.high_level_plan):
                    print("All high level steps completed!")
                    done = True
                else:
                    note = self.do_supervisor_success(learnings)
                    print(f"New note for next step: {note}")
            elif return_type == "fail":
                failure_summary = result
                print(f"Step {self.current_step_index} failed! Failure Summary: {failure_summary}")
                note = self.do_supervisor_failure(failure_summary)
                print(f"New note for next step: {note}")
            elif return_type == "continue":
                done = result
            elif return_type == "invalid_action":
                print("VL agent failed to produce a valid action after multiple attempts. Exiting.")
                break
            pbar.update(1)
        pbar.close()
        self.env.close()

        

        
        
@click.command()
@click.option("--size", default=8, type=int)
def do(size):
    environment = get_pokemon_environment(game_variant="pokemon_red", controller=PokemonStateWiseController(), 
                                        environment_variant="high_level",
                                        save_video=True,
                                            init_state="starter", session_name=f"high_level_{size}", headless=True)
    mission = "Select a starter from the bench to your right, and then leave the building from the entrance below."
    vl = VL(environment, mission, size=size)
    vl.play()

# Plot rewards over time

if __name__ == "__main__":
    do()