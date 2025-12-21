from abc import abstractmethod, ABC
from typing import Optional, Type, Dict, Any, List, Tuple

from poke_worlds.utils import load_parameters, log_error, log_info, log_warn, get_lowest_level_subclass, verify_parameters, log_dict


from poke_worlds.emulation import Emulator, StateTracker
from poke_worlds.interface.controller import Controller, LowLevelController
from poke_worlds.interface.action import HighLevelAction

import numpy as np
import gymnasium as gym
import warnings 
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html")
# This is to ignore deprecation warnings from pygame about pkg_resources
import pygame
import matplotlib.pyplot as plt




class Environment(gym.Env, ABC):
    """ Base class for environments interfacing with the emulator. """

    REQUIRED_EMULATOR = Emulator
    """ The highest level emulator that the environment can interface with. """

    REQUIRED_STATE_TRACKER = StateTracker
    """ The state tracker that tracks the minimal state information required for the environment to function. """
    
    @staticmethod
    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        """
        Override default emulator keyword arguments for this environment.

        Override this method in subclasses to modify the default emulator keyword arguments.

        You may want to use safe_override_state_tracker_class to ensure compatibility of state tracker classes.

        Args:
            emulator_kwargs (dict): Incoming emulator keyword arguments.
        Returns:
            dict: The overridden emulator keyword arguments.
        """
        return emulator_kwargs

    @staticmethod
    def safe_override_state_tracker_class(incoming_state_tracker_class: Type[StateTracker], required_state_tracker_class: Type[StateTracker]) -> Type[StateTracker]:
        """
        Safely overrides the state tracker class for the environment.

        Use this in override_emulator_kwargs to ensure that the lowest level state tracker class is chosen.
        
        Args:
            incoming_state_tracker_class (Type[StateTracker]): The incoming state tracker class.
            required_state_tracker_class (Type[StateTracker]): Usually the required state tracker class for the environment.
        Returns:
            Type[StateTracker]: The chosen state tracker class.
        """
        if issubclass(incoming_state_tracker_class, required_state_tracker_class):
            return incoming_state_tracker_class
        elif issubclass(required_state_tracker_class, incoming_state_tracker_class):
            return required_state_tracker_class
        else:
            return incoming_state_tracker_class # Don't know which one to pick, so just go with the incoming one.
    
    def __init__(self):
        """
        Ensures that the environment has the required attributes.
        All subclasses must call this __init__ method AFTER setting up the required attributes.

        If you are implementing a subclass, ensure that the following attributes are set:
            - _parameters: dict of config parameters
            - _emulator: emulator instance
            - _controller: controller instance
            - observation_space: gym space defining observation space structure
        """
        if not hasattr(self, "_parameters"):
            raise ValueError("Environment must have a '_parameters' attribute.")
        self._parameters: dict = self._parameters
        required_attributes = ["_emulator", "_controller", "observation_space"]
        for attr in required_attributes:
            if not hasattr(self, attr):
                log_error(f"Environment requires attribute '{attr}' to be set. Implement this in the subclass __init__", self._parameters)
        self._emulator: Emulator = self._emulator
        self._controller: Controller = self._controller
        self.observation_space: gym.spaces.Space = self.observation_space
        # For Intellisence lol
        if not issubclass(type(self._emulator), self.REQUIRED_EMULATOR):
            log_error(f"Environment requires an Emulator of type {self.REQUIRED_EMULATOR.NAME}, but got {type(self._emulator).NAME}", self._parameters)
        if not isinstance(self._controller, Controller):
            log_error(f"Environment requires a Controller instance, but got {type(self._controller)}", self._parameters)
        self.REQUIRED_STATE_TRACKER = get_lowest_level_subclass([self.REQUIRED_STATE_TRACKER, self._controller.REQUIRED_STATE_TRACKER])
        if not issubclass(type(self._emulator.state_tracker), self.REQUIRED_STATE_TRACKER):
            log_error(f"Environment requires a StateTracker of type {self.REQUIRED_STATE_TRACKER.NAME}, but got {type(self._emulator.state_tracker).NAME}", self._parameters)
        self._controller.assign_emulator(self._emulator)
        self.action_space = self._controller.get_action_space()
        """ The Gym action Space provided by the controller. """
        self.actions = self._controller.ACTIONS
        """ A list of HighLevelAction Types provided by the controller. """
        self.render_mode = "human"
        """ The render mode of the environment. Supports 'human' and 'rgb_array', but strongly assumes 'human' as can just read the emulator screen from `get_info` """ 
        self._window = None
        """ The pygame window for rendering in 'human' mode. Initialized on first render call. """
        self._clock = None
        """ The pygame clock for rendering in 'human' mode. Initialized on first render call. """

    @abstractmethod
    def get_observation(self) -> gym.spaces.Space:
        """
        Returns the current observation from the emulator.
        Returns:
            observation (gym.spaces.Space): The current observation.
        """
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the full state information as defined by the emulator's state tracker.
        Returns:
            info (dict): The full state information from the state tracker.
        """
        return self._emulator.state_tracker.report()

    def get_final_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the final state information from the emulator when all episodes are done.
        Will involve summaries over all episodes played.
        Returns:
            info (dict): The final state information from the state tracker.
        """
        return self._emulator.state_tracker.report_final()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[gym.spaces.Space, Dict[str, Dict[str, Any]]]:
        """
        Resets the environment and emulator to the initial state.
        Args:
            seed (int, optional): Seed for random number generators.
            options (dict, optional): Additional options for resetting the environment.
        Returns:
            observation (object): The initial observation of the environment.

            info (dict): Additional information about the reset.
        """
        super().reset(seed=seed, options=options)

        self._emulator.reset()
        self._controller.seed(seed)
        return self.get_observation(), self.get_info()
    
    @abstractmethod
    def determine_reward(self, start_state: Dict[str, Dict[str, Any]], action: gym.spaces.Space, transition_states: List[Dict[str, Dict[str, Any]]], action_success: bool) -> float:
        """
        Determines the reward based on the transition from start_state through transition_states.
        Args:
            start_state (Dict[str, Dict[str, Any]]): The state before the action was taken.
            action (gym.spaces.Space): The action taken.
            transition_states (List[Dict[str, Dict[str, Any]]]): A list of states observed during the action execution.
            action_success (bool): Whether the action was successful.
        Returns:
            float: The computed reward.
        """
        raise NotImplementedError
    
    @abstractmethod
    def determine_terminated(self, state: Dict[str, Dict[str, Any]]) -> bool:
        """
        Determines whether the episode reaches the goal / terminal state based on the transition from start_state through transition_states.
        This method is NOT meant to be used to determine if the step count has exceeded the maximum. 

        Args:
            state (Dict[str, Dict[str, Any]]): The current state after the action was taken.
        Returns:
            bool: Whether the episode is terminated.
        """
        pass
    
    def step(self, action: gym.spaces.OneOf) -> Tuple[gym.spaces.Space, float, bool, bool, Dict[str, Dict[str, Any]]]:
        """
        Executes the given Gym Space action in the environment via the controller.
        Use step_high_level_action to execute high level actions directly.

        Args:
            action (gym.spaces.OneOf): The action to execute. Must be a valid action in the controller's action space.

        Returns:
            observation (gym.spaces.Space): The observation after executing the action.
            reward (float): The reward obtained from executing the action.
            terminated (bool): Whether the episode has ended (reached the terminal state of the MDP).
            truncated (bool): Whether the episode was truncated (exceeded the maximum allowed steps).
            info (Dict[str, Dict[str, Any]]): Full state information.
        """
        high_level_action, kwargs = self._controller._space_action_to_high_level_action(action)
        return self.step_high_level_action(high_level_action, **kwargs)

    def step_high_level_action(self, action: HighLevelAction, **kwargs) -> Tuple[gym.spaces.Space, float, bool, bool, Dict[str, Dict[str, Any]]]:
        """
        Executes the given High Level action in the environment via the controller.

        Args:
            action (HighLevelAction): The high level action to execute.
            **kwargs: Additional keyword arguments to pass to the action.

        Returns:
            observation (gym.spaces.Space): The observation after executing the action.
            reward (float): The reward obtained from executing the action.
            terminated (bool): Whether the episode has ended (reached the terminal state of the MDP).
            truncated (bool): Whether the episode was truncated (exceeded the maximum allowed steps).
            info (Dict[str, Dict[str, Any]]): Full state information.
        """
        if self._emulator.check_if_done():
            log_error("Cannot step environment because emulator indicates done. Please reset the environment.", self._parameters)
        start_state = self.get_info()
        transition_states, action_success = self._controller.execute(action, **kwargs)
        if transition_states is None: # then the action was not a valid one according to the controller. Will return Nones for all
            return None, None, None, None, None
        truncated = self._emulator.check_if_done()
        observation = self.get_observation()
        current_state = self.get_info()
        terminated = self.determine_terminated(current_state)
        reward = self.determine_reward(start_state, action, transition_states, action_success)
        return observation, reward, terminated, truncated, current_state    
    
    def step_str(self, input_str: str)  -> Tuple[gym.spaces.Space, float, bool, bool, Dict[str, Dict[str, Any]]]:
        """
        Attempts to execute an input string representation of an action
        """
        action, kwargs = self._controller.string_to_high_level_action(input_str)
        if action is None: # not a valid action, will not perform an action and will simply return Nones
            return None, None, None, None, None
        return self.step_high_level_action(action, **kwargs)
    
    def close(self):
        """
        Closes the environment and the underlying emulator.
        """
        log_info("Closing environment and emulator.", self._parameters)
        self._emulator.close()

    
    def _screen_render(self, screen: np.ndarray):
        """
        TODO: Docstring
        """
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._emulator.screen_shape[0], self._emulator.screen_shape[1])
            )
        if self._clock is None:
            self._clock = pygame.time.Clock()        
        rgb = np.stack([screen[:, :, 0], screen[:, :, 0], screen[:, :, 0]], axis=2)
        pygame.surfarray.blit_array(self._window, rgb.swapaxes(0,1))
        pygame.display.flip()
        self._clock.tick(60)  # Limit to 60 FPS

    
    def render(self) -> Optional[np.ndarray]:
        """
        Gets the current screen from the emulator and renders it. 

        Use this method only if you want to generally run the emulator in headless mode but still want to see the screen occasionally.

        Do not call this method if the emulator is not headless, you should already have a PyBoy interactive window open in that case.

        Returns:
            If render_mode is 'rgb_array', returns the current screen as a numpy array. However this is always accessible via self.get_info()['core']['current_frame'], so this is mostly for Gym compatibility.
        """
        if self._emulator.headless == False:
            log_error("You probably don't want to call render() when the emulator is not headless.", self._parameters)
        screen = self._emulator.get_current_frame() # shape: 144, 160, 1
        if self.render_mode == "human":
            self._screen_render(screen)
        elif self.render_mode == "rgb_array":
            return screen
        else:
            log_error(f"Unsupported render mode: {self.render_mode}", self._parameters)

    def seed(self, seed: Optional[int] = None):
        """
        Seeds the environment's random number generator and the controller's RNG.

        Args:
            seed (int, optional): The seed value.
        """
        self._controller.seed(seed)

    def render_obs(self):
        """
        Provide a way to render the output of get_observation to a human. 
        Implement if you want to use the human_step_play method.
        """
        raise NotImplementedError
    
    def render_info(self):
        """
        Provide a way to render the output of get_info to a human. 
        Implement if you want to use the human_step_play method with show_info=True
        """
        raise NotImplementedError
    
    
    def human_step_play(self, max_steps: int=50, show_info: bool=False):
        """
        Opens a render window and allow the human to play through the environment as an agent would

        Args:
            max_steps (int): max steps to take
            show_info (bool): whether to show the state space (as opposed to just observation space)
        """
        observation, info = self.reset()
        self.render_mode = "human"
        log_info(f"Doing human step play for {max_steps} max steps...")
        steps = 0
        done = False
        action_input_str = self._controller.get_action_strings()
        rewards = []
        log_info(f"Allowed Actions: \n{action_input_str}", self._parameters)
        while not done and steps < max_steps:
            self.render()
            if show_info:
                self.render_info()
            input_str = input("Enter Action: ").strip()
            possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info = self.step_str(input_str)
            if possible_obs is not None:
                observation, reward, terminated, truncated, info = possible_obs, possible_reward, possible_terminated, possible_truncated, possible_info
            else:
                log_warn("That was not a valid input. did nothing", self._parameters)
            rewards.append(reward)
            if terminated or truncated:
                break
            steps += 1



class DummyEnvironment(Environment):
    """ A dummy environment that does nothing special. """

    def __init__(self, emulator: Emulator, controller: Controller, parameters: Optional[dict]=None):
        """
        Initializes the DummyEnvironment with the given emulator and controller.

        It is safe to overwrite the self.observation_space in the subclass after calling this __init__ method.
        """
        self._parameters = load_parameters(parameters)
        self._emulator = emulator
        self._controller = controller
        screen_shape = self._emulator.screen_shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(screen_shape[1], screen_shape[0], 1), dtype=np.uint8)
        """ The observation space is the raw pixel values of the emulator's screen. """
        super().__init__()

    def get_observation(self) -> gym.spaces.Space:
        return self._emulator.get_current_frame()
    
    def determine_reward(self, start_state: Dict[str, Dict[str, Any]], action: gym.spaces.Space, transition_states: List[Dict[str, Dict[str, Any]]], action_success: bool) -> float:
        return 0.0
    
    def determine_terminated(self, state):
        return 0.0
    
    def render_obs(self): # Might cause issues if you try to render() as well
        info = self.get_info()
        screen = info["core"]["current_frame"]
        #self._screen_render(screen)

    def render_info(self):
        info = self.get_info()
        info["core"].pop("current_frame")
        info["core"].pop("passed_frames")
        log_info("State: ", self._parameters)
        log_dict(info, self._parameters)
