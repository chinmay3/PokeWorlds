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


def get_action_success_message(action, action_kwargs, action_success, action_return):
    """
    Given an action and its result, return a string message indicating the success or failure of the action.
    Will come with hints to guide the next decision by the VLM. 
    """
    interaction_advice = f"If you want to try to interact with it, use interact() or checkinteraction() to get an estimate on whether or not you can interact with it."
    path_blocked_advice = f"If you are trying to go somewhere, then this direction is blocked. Try moving around it or going a different way."
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
            action_success_message = "Could not locate the target on screen. You may need to move around, or trust your visual intuition from the screen."
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
            action_success_message = "Could not find the object on screen to seek. You may need to move around, or trust your visual intuition from the screen."
        elif action_success == 1:
            action_success_message = "Could not move to the target location because there is an obstacle completely blocking the path. Try moving around it or going a different way."
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



class VL:
    system_prompt = """
You are playing a Pokemon game and the high level plan is as follows: [HIGH_LEVEL_PLAN]. 
But you are just trying to achieve one particular step as your mission [STEP]. 
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

    action_instruction_prompt = f"""
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
    Note: Update your note with important information that will help guide you to your mission. Has your context changed from the previous note? What should your overall next step be? Have you learned anything from the dialogues or screens that you should track. 
    Critique of Previous Action Failures: From the results of your previous actions, have you been moving closer to your immediate goal? If not, why do you think that is? What can you do differently this time to improve your chances of success? If you are stuck in a loop, what is the likely reason and how can you break out of it? Importantly, if you are moving towards a coordinate, remember that the coordinate will change as you move, so state the new target coordinate for the next step if you are using a move action. 
    Action: SELECTED ACTION COMMAND. 
    \nCurrent state is [STATE]. The allowed actions for this state are:\n[ALLOWED_ACTIONS]. 


Information from prior steps:
{prev_information_prompt}    
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
[SEP]
2. STEP TWO (Simple description of step and Criteria for completion)
[SEP]
...
Note: <CONCISE NOTE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
    """

    superviser_failure_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION] and you have come up with the following high-level plan:
[HIGH_LEVEL_PLAN]. The player agent can take the following actions (depending on the situation): [ALLOWED_ACTIONS]
Your agent has been trying to achieve [STEP_DESCRIPTION], but has not succeeded for too long.
It has left you the following message regarding its failure: 
[AGENT_MESSAGE]
Your job is to now, given the results of your agent's previous actions and the state of the current screen, revise the high-level plan and then craft a new concise note for your player agent to follow to achieve the next step of this plan.
Format your response as:
Reasoning: <YOUR REASONING ABOUT WHAT WENT WRONG AND HOW TO FIX IT>
Revised High Level Plan:
1. STEP ONE (Simple description of step and Criteria for completion) DO NOT REPEAT STEPS THAT HAVE ALREADY BEEN COMPLETED. Start at what needs to be done next.
[SEP]
2. STEP TWO (Simple description of step and Criteria for completion)
[SEP]
...
Note: <CONCISE NOTE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
    """

    superviser_success_prompt = """
You are playing a Pokemon game. Your overall mission is to [MISSION] and you have come up with the following high-level plan:
[HIGH_LEVEL_PLAN]
Your agent has successfully achieved [STEP_DESCRIPTION] and has left you with the following message: [AGENT_MESSAGE]
Your job is to now, given the results of your agent's previous actions and the state of the current screen, revise the high-level plan, and then craft a new concise note for your player agent to follow to achieve the next step of this plan.
Format your response as:
Reasoning: <YOUR REASONING ABOUT WHAT HAS OCCURED AND WHICH PARTS OF THE MISSION TO FOCUS ON NEXT>
Revised High Level Plan:
1. STEP ONE (Simple description of step and Criteria for completion) DO NOT REPEAT STEPS THAT HAVE ALREADY BEEN COMPLETED. Start at what needs to be done next.
[SEP]
2. STEP TWO (Simple description of step and Criteria for completion)
[SEP]
...
Note: <CONCISE NOTE FOR PLAYER AGENT TO FOLLOW TO ACHIEVE STEP ONE>
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
        self.all_inputs.append(prompt)
        self.all_outputs.append(output_text)
        self.save_logs()
        return output_text
    
    def parse_plan_steps(self, plan_text):
        steps = []
        for line in plan_text.lower().split("[sep]"):
            line = line.strip()
            if line != "":
                steps.append(line)
        return steps

    def parse_supervisor_start(self, output_text):
        if "note:" in output_text.lower():
            high_level_plan, note = output_text.lower().split("note:")
            steps = self.parse_plan_steps(high_level_plan)
            return steps, note.strip()
        else:
            print("Failed to parse supervisor start output: ", output_text)
            return None, None
        
    def parse_supervisor_failure(self, output_text):
        if "note:" in output_text.lower():
            if "revised high level plan:" in output_text.lower():
                reasoning, rest = output_text.lower().split("revised high level plan:")
                high_level_plan, note = rest.split("note:")
                steps = self.parse_plan_steps(high_level_plan)
                return steps, note.strip()
            else:
                note = output_text.lower().split("note:")[-1].strip()
                return None, note
        else:
            print("Failed to parse supervisor failure output: ", output_text)
            return None, None

    def parse_supervisor_success(self, output_text):
        return self.parse_supervisor_failure(output_text)
        
    def parse_failure_summary(self, output_text):
        return output_text.strip()
    
    def parse_action_instruction(self, output_text):
        output_text = output_text.strip().lower()
        if "mission complete: yes" in output_text or "learnings" in output_text:
            mission_complete = True
            learnings = output_text.lower().split("learnings:")[-1].strip()
            learnings = learnings.split("Action:")[0].strip()
            return mission_complete, learnings
        else: # assume no
            mission_complete = False
            if "note" in output_text:
                remaining = output_text.split("note:")[-1]
            else:
                print(f"Warning: 'note' not found in action instruction output. Output was: {output_text}")
                return mission_complete, ("unable to parse critique, come up with a new one", "unable to parse note, come up with a new one", "passdialogue()")
            if "critique" in remaining:
                note, rest = remaining.split("critique", 1)
                if "action:" in rest:
                    critique, action_part = rest.rsplit("action:", 1)
                    action_part = action_part.strip()
                    action_part = action_part.replace("<action>", "")
                    action_part = action_part.replace("</action>", "")
                    action_part = action_part.strip("<")
                    action_part = action_part.strip(">")
                    return mission_complete, ("critique " + critique.strip(), note.strip(), action_part.strip())
                else:
                    return mission_complete, ("unable to parse critique, come up with a new one", note.strip(), "passdialogue()")
            else:
                print(f"Warning: 'critique:' not found in action instruction output. Output was: {output_text}")
                return mission_complete, ("unable to parse critique, come up with a new one", "unable to parse note, come up with a new one.", "passdialogue()")
            
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
    
    def get_previous_action_string(self):
        action_buffer = self.env.action_buffer
        msgs = ""
        total_actions = len(action_buffer)
        for i, (action, action_kwargs, action_success, action_return) in enumerate(action_buffer):
            msg = get_action_success_message(action, action_kwargs, action_success, action_return)
            t = total_actions - 1 - i 
            if t != 0:
                msgs += f"Previous Action: {action.__name__} with kwargs {action_kwargs}. Status message: {msg}"
            msgs += f"Step T-{t}: Action: {action.__name__} with kwargs {action_kwargs}. Status message: {msg}\n"
        return msgs
    
    def handle_action(self):
        assert self.note is not None, "Note must be set before calling handle_action."
        assert self.high_level_plan is not None, "High level plan must be set before calling handle_action."
        if self.observation is None:
            observation = self.env.get_observation()
        else:
            observation = self.observation
        allowed_actions = self.env._controller.get_action_strings(return_all=False)
        action_buffer_str = self.get_previous_action_string()
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
        high_level_plan_str = "\n".join(self.high_level_plan)
        if self.steps_taken_for_current_goal > self.max_steps_per_goal:
            # Need to fail and ask supervisor for new note
            prompt = self.full_failure_summary_prompt.replace("[HIGH_LEVEL_PLAN]", high_level_plan_str).replace("[STEP]", self.high_level_plan[self.current_step_index]).replace("[NOTE]", self.note).replace("[ACTION_BUFFER]", action_buffer_str).replace("[NOTE_BUFFER]", "\n".join(self.agent_note_buffer)).replace("[OCR_BUFFER]", ocr_buffer_str).replace("[OCR]", recent_ocr)
            failure_summary = self.infer(prompt, screen)
            return "fail", failure_summary
        else:
            self.steps_taken_for_current_goal += 1
            prompt = self.full_action_prompt.replace("[HIGH_LEVEL_PLAN]", high_level_plan_str).replace("[STEP]", self.high_level_plan[self.current_step_index]).replace("[NOTE]", self.note).replace("[ACTION_BUFFER]", action_buffer_str).replace("[NOTE_BUFFER]", "\n".join(self.agent_note_buffer[-2:])).replace("[OCR_BUFFER]", ocr_buffer_str).replace("[OCR]", recent_ocr).replace("[STATE]", state.name).replace("[ALLOWED_ACTIONS]", allowed_actions)
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
                    if self.prev_action is None:
                        self.prev_action = action_str
                    else:
                        if "locate" in action_str.lower() and self.prev_action == action_str:
                            # prevent repeated locate actions
                            previous_action_strings += f"\n[VITAL SYSTEM MESSAGE] Don't you dare use '{action_str}' again. You just did that and got a result too. Stop overthinking it, trust the previous result and move towards it."
                            continue
                        if "interact" in action_str.lower() and self.prev_action == action_str:
                            previous_action_strings += f"\n[VITAL SYSTEM MESSAGE] Don't you dare use '{action_str}' again. You just did that. If it didn't work the first time, or gave you a message that theres nothing to interact with, it means you need to move towards and face something to do it. "
                        if "move" in action_str.lower() and "move" in self.prev_action.lower():
                            move_parse = action_str.lower().split(",")[:-1] # should just get the int
                            prev_move_parse = self.prev_action.lower().split(",")[:-1]
                            match_step = move_parse == prev_move_parse
                            msg = f"\nYou moved up previously and now you are trying to move down with the same number of steps. This is likely to return you to the same position. Stop overthinking your movement and take a second to re-understand your surroundings. Perhaps check for interaction or locate something, but stop moving back and forth."
                            if match_step:
                                if "up" in action_str.lower() and "down" in self.prev_action.lower():
                                    previous_action_strings += msg
                                    continue
                                if "right" in action_str.lower() and "left" in self.prev_action.lower():
                                    previous_action_strings += msg
                                    continue
                                if "left" in action_str.lower() and "right" in self.prev_action.lower():
                                    previous_action_strings += msg
                                    continue
                                if "down" in action_str.lower() and "up" in self.prev_action.lower():
                                    previous_action_strings += msg
                                    continue
                    
                    self.agent_note_buffer.append(f"Critique: {critique}\nUpdated Note: {updated_note}")
                    # take action on environment and log
                    self.prev_action = action_str
                    possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info = self.env.step_str(action_str)
                    if possible_obs is not None:
                        observation, reward, terminated, truncated, info = possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info
                        action, action_kwargs, transition_states, action_success = info["core"]["previous_action_details"]
                        self.observation = observation
                        done = terminated or truncated
                        return "continue", done
                    else:
                        # invalid action, try again
                        previous_action_strings += f"\nThe action '{action_str}' was invalid. Please choose a valid action next time."
                        if "locate" in action_str.lower():
                            previous_action_strings += " Remember to only use 'locate' actions with the allowed input options, and not arbitrary text."
                        if n_attempts == self.max_action_attempts - 1:
                            print("Max action attempts reached. Last invalid action was: ", action_str)
                            breakpoint()

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
@click.option("--model_name", default="Qwen/Qwen3-VL-32B-Instruct", type=str)
def do(model_name):
    short_model = model_name.split("/")[-1]
    environment = get_pokemon_environment(game_variant="pokemon_red", controller=PokemonStateWiseController(), 
                                        environment_variant="high_level",
                                        save_video=True,
                                            init_state="starter", session_name=f"high_level_{short_model}", headless=True)
    mission = "Seek and select any one pokeball with a starter from the bench to your right, and then leave the building from the entrance below."
    vl = VL(environment, mission, model_name=model_name)
    vl.play()

# Plot rewards over time

if __name__ == "__main__":
    do()
