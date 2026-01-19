from poke_worlds.utils import load_parameters, log_error, log_info, verify_parameters
from poke_worlds.interface import Environment
from poke_worlds.execution.vlm import SupervisorVLM
from poke_worlds.execution.report import ExecutionReport, SupervisorReport, SimpleSupervisorReport
from poke_worlds.execution.executor import Executor, SimpleExecutor
from abc import ABC, abstractmethod
from typing import Type, List, Dict, Tuple, Any
from tqdm import tqdm


class Supervisor(ABC):
    REQUIRED_EXECUTOR = Executor
    """ The type of Executor required by this Supervisor."""

    REQUIRED_REPORT = SupervisorReport
    """ The type of Report produced by this Supervisor."""

    def __init__(self, *, game: str, environment: Environment, model_name: str = None, vlm_kind: str = None, executor_class: Type[Executor] = None, execution_report_class: Type[ExecutionReport] = None, executor_max_steps=None, parameters: dict = None):
        self._parameters = load_parameters(parameters)
        if executor_class is None:
            executor_class = self.REQUIRED_EXECUTOR
        if not issubclass(executor_class, self.REQUIRED_EXECUTOR):
            log_error(f"Executor class {executor_class} is not a subclass of required {self.REQUIRED_EXECUTOR}.")
        self._environment = environment
        self._EXECUTOR_CLASS = executor_class
        self._EXECUTION_REPORT_CLASS = execution_report_class
        self._vlm = SupervisorVLM(model_name=model_name, vlm_kind=vlm_kind)
        self._game = game
        self._report = self._create_report()
        if executor_max_steps is None:
            self.executor_max_steps = self._parameters["executor_max_steps"]
        else:
            self.executor_max_steps = executor_max_steps


    def _create_report(self, **play_kwargs) -> SupervisorReport:
        """
        Creates the SupervisorReport instance. Override this method to customize report creation.

        :param play_kwargs: All keyword arguments passed to the play method. May or may not be used.
        :type play_kwargs: dict
        :return: The created SupervisorReport instance.        
        :rtype: SupervisorReport
        """
        return self.REQUIRED_REPORT(parameters=self._parameters)


    def call_executor(self, **executor_kwargs) -> ExecutionReport:
        """
        Calls the Executor with provided arguments, logs the call and its report.
        You should generally never call the executor directly, but use this method instead to ensure proper logging.
                
        :param executor_kwargs: Keyword arguments to pass to the Executor.
        :type executor_kwargs: dict
        :return: The ExecutionReport returned by the Executor.
        :rtype: ExecutionReport
        """
        game = executor_kwargs.pop("game", self._game)
        environment = executor_kwargs.pop("environment", self._environment)
        parameters = executor_kwargs.pop("parameters", self._parameters)
        call_args = executor_kwargs.copy()
        executor = self._EXECUTOR_CLASS(game=game, environment=environment, execution_report_class=self._EXECUTION_REPORT_CLASS, parameters=parameters, **executor_kwargs)
        report = executor.execute(step_limit=self.executor_max_steps, show_progress=True)
        self._report.log_executor_call(call_args, report)
        return report
    

    @abstractmethod
    def _play(self, **kwargs):
        """
        The main loop of the Supervisor. Should take in a reset environment and implement the logic to interact with the environment using the Executor.
        Always call executor with the call_executor method to ensure proper logging.

        
        :param kwargs: Additional keyword arguments for customization.
        :type kwargs: dict
        """
        pass


    def play(self, **kwargs) -> SupervisorReport:
        """
        Public method to start the Supervisor's play loop. Initializes the environment and calls the internal _play method.
        
        :param kwargs: Additional keyword arguments for customization.
        :type kwargs: dict
        :return: The final SupervisorReport after execution.
        :rtype: SupervisorReport
        """
        self._environment.reset()
        self._report = self._create_report(**kwargs)
        self._play(**kwargs)
        return self._report


class SimpleSupervisor(Supervisor):
    REQUIRED_EXECUTOR = SimpleExecutor

    REQUIRED_REPORT = SimpleSupervisorReport

    supervisor_visual_context_prompt = """
You are playing [GAME]. Your overall mission is to [MISSION]. 
Given the screen state of the game, come up with a brief and concise description of the current visual and game context that covers the most important details relevant to the mission and plan, while ignoring the irrelevant details.
Context: 
"""

    supervisor_start_prompt = """
You are playing [GAME]. Your overall mission is to [MISSION]. You are to create a plan for a player agent to follow to achieve this mission.
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
You are playing [GAME]. Your overall mission is to [MISSION], with the allowed actions being [ALLOWED_ACTIONS]. For this, you developed the following high level plan:
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
    def __init__(self, **kwargs):
        game = kwargs["game"]
        # replace [GAME] in prompts
        self.supervisor_visual_context_prompt = self.supervisor_visual_context_prompt.replace("[GAME]", game)
        self.supervisor_start_prompt = self.supervisor_start_prompt.replace("[GAME]", game)
        self.supervisor_common_prompt = self.supervisor_common_prompt.replace("[GAME]", game)
        self.executor_return_analysis_prompt = self.executor_return_analysis_prompt.replace("[GAME]", game)
        self.executor_information_construction_prompt = self.executor_information_construction_prompt.replace("[GAME]", game)
        super().__init__(**kwargs)

    def _create_report(self, **play_kwargs):
        mission = play_kwargs["mission"]
        initial_visual_context = play_kwargs["initial_visual_context"]
        return SimpleSupervisorReport(mission=mission, initial_visual_context=initial_visual_context, parameters=self._parameters)
    
    
    def parse_supervisor_start(self, output_text: str) -> Tuple[List[str], str]:
        if "note:" in output_text.lower():
            high_level_plan, note = output_text.lower().split("note:")
            steps = self.parse_plan_steps(high_level_plan)
            return steps, note.strip()
        else:
            print("Failed to parse supervisor start output: ", output_text)
            return None, None
        
    
    def do_supervisor_start(self, visual_context: str):
        allowed_actions = self.environment.get_action_strings(return_all=True)
        allowed_actions_str = ""
        for class_name, action_str in allowed_actions.items():
            allowed_actions_str += f"{action_str}\n"
        prompt = self.supervisor_start_prompt.replace("[MISSION]", self.mission).replace("[ALLOWED_ACTIONS]", allowed_actions_str).replace("[VISUAL_CONTEXT]", visual_context)
        current_frame = self.environment.get_info()["core"]["current_frame"]
        output_text = self._vlm.infer(prompt, current_frame)[0]
        steps, note = self.parse_supervisor_start(output_text)
        self.high_level_plan = steps
        self._report.high_level_plan = steps
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
    
    def _play(self, mission: str, initial_visual_context: str):
        assert mission is not None, "Mission must be provided to play()."
        assert initial_visual_context is not None, "Initial visual context must be provided to play()."
        self.mission = mission
        visual_context = initial_visual_context
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
            execution_report = self.call_executor(high_level_goal=self.mission,
                                                  task=immediate_task,
                                                  initial_plan=initial_plan,
                                                  visual_context=visual_context,
                                                  exit_conditions=[])
            actions_and_observations = "\n".join(execution_report.get_execution_summary())
            prompt = self.executor_return_analysis_prompt.replace("[LESSONS_LEARNED]", lessons_learned).replace("[IMMEDIATE_TASK]", immediate_task).replace("[ACTION_AND_OBSERVATIONS_LOG]", actions_and_observations)
            current_frame = self._environment.get_info()["core"]["current_frame"]
            output_text = self._vlm.infer(prompt, current_frame)[0]
            analysis, lessons_learned, visual_context, mission_accomplished = self.parse_executor_return_analysis(output_text)
            self._report.update_before_loop(executor_analysis=analysis, lessons_learned=lessons_learned, visual_context=visual_context)
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
            current_frame = self._environment.get_info()["core"]["current_frame"]
            output_text = self._vlm.infer(prompt, current_frame)[0]
            immediate_task, initial_plan = self.parse_executor_information_construction(output_text)
            if immediate_task is None or initial_plan is None:
                break
            pbar.update(1)
        print("Finished playing VLM agent.")