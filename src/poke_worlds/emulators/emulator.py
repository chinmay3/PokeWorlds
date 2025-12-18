from enum import Enum
from abc import ABC, abstractmethod
from typing import Type, Optional, Tuple


import os
import re
from time import perf_counter
import sys
import shutil
import uuid
from poke_worlds.utils import load_parameters, log_error, log_warn, file_makedir, log_info, is_none_str, nested_dict_to_str, verify_parameters


import cv2
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
import numpy as np
from tqdm import tqdm


class LowLevelActions(Enum):
    """
    Enum for low-level actions that can be performed on the GameBoy emulator.
    """
    PRESS_ARROW_DOWN = WindowEvent.PRESS_ARROW_DOWN
    PRESS_ARROW_LEFT = WindowEvent.PRESS_ARROW_LEFT
    PRESS_ARROW_RIGHT = WindowEvent.PRESS_ARROW_RIGHT
    PRESS_ARROW_UP = WindowEvent.PRESS_ARROW_UP
    PRESS_BUTTON_A = WindowEvent.PRESS_BUTTON_A
    PRESS_BUTTON_B = WindowEvent.PRESS_BUTTON_B
    PRESS_BUTTON_START = WindowEvent.PRESS_BUTTON_START
    
class ReleaseActions(Enum):
    """
    Enum for release actions corresponding to low-level actions.
    """
    release_actions = {
        LowLevelActions.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
        LowLevelActions.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
        LowLevelActions.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
        LowLevelActions.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
        LowLevelActions.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
        LowLevelActions.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
        LowLevelActions.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START}


class NamedScreenRegion:
    """
    Saves a reference to a named screen region (always a rectangle) for easy access.
    """
    def __init__(self, name: str, start_x: int, start_y: int, width: int, height: int, parameters: dict, target_path: Optional[str] = None):
        """
        Initializes a named screen region.

        Args:
            name (str): The name of the screen region.
            start_x (int): The starting x-coordinate of the region in pixel space of the full resolution game screen. 
            start_y (int): The starting y-coordinate of the region in pixel space of the full resolution game screen.
            width (int): The width of the region in pixels.
            height (int): The height of the region in pixels.
            target (str): Optional path to a .npy file containing a screen capture of this region. Non-existent paths are only allowed if parameters['debug_mode'] (from configs/project_vars.yaml) is set to True. 
        """
        self.name = name
        """ Name of the screen region. """
        self.start_x = start_x
        """ The starting x-coordinate of the region. """
        self.start_y = start_y
        """ The starting y-coordinate of the region. """
        self.width = width
        """ The width of the region. """
        self.height = height
        """ The height of the region. """
        self._parameters = parameters
        self.target_path = target_path
        """ Path to npy file of a screen capture that we will be comparing this region against. Optional. """
        self.target : Optional[np.ndarray]= None
        if target_path is not None:
            if not target_path.endswith(".npy"):
                target_path = target_path + ".npy"
            if not os.path.exists(target_path):
                if not self._parameters['debug_mode']:
                    log_error(f"Target file {target_path} does not exist. This is only allowed in debug_mode (can be set in configs/project_vars.yaml)", self._parameters)
                else:
                    log_warn(f"Target file {target_path} does not exist. Continuing since debug_mode is enabled.", self._parameters)
            else:
                self.target = np.load(target_path)

    def get_end_x(self) -> int:
        """
        Returns the end x-coordinate of the named screen region.
        
        Returns:
            int: The end x-coordinate of the named screen region.
        """
        return self.start_x + self.width
    
    def get_end_y(self) -> int:
        """
        Returns the end y-coordinate of the named screen region.
        Returns:
            int: The end y-coordinate of the named screen region.
        """
        return self.start_y + self.height
    
    def get_corners(self) -> Tuple[int, int, int, int]:
        """
        Returns the corners of the named screen region as (start_x, start_y, end_x, end_y).

        Returns:
            Tuple[int, int, int, int]: The corners of the named screen region.
        """
        return (self.start_x, self.start_y, self.get_end_x(), self.get_end_y())
    
    def __str__(self) -> str:
        return f"NamedScreenRegion(name={self.name}, start_x={self.start_x}, start_y={self.start_y}, width={self.width}, height={self.height})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    
    def matches_target(self, reference: np.ndarray, strict_shape: bool=True, epsilon=0.01) -> bool:
        """
        Compares the given reference image to the target image using Absolute Error (AE).
        Args:
            reference (np.ndarray): The reference image to compare.
            strict_shape (bool, optional): Whether to error out if the array shapes do not match. 
            epsilon (float, optional): The threshold for considering a match. 
        Returns:
            bool: True if the MSE is below the epsilon threshold, False otherwise.
        """
        if self.target is None:
            if self._parameters["debug_mode"]:
                return False
            log_error(f"No target image set for NamedScreenRegion {self.name}. Cannot compare.", self._parameters)
        if reference.shape != self.target.shape:
            if strict_shape:
                log_error(f"Reference image shape {reference.shape} does not match target image shape {self.target.shape} for NamedScreenRegion {self.name}.", self._parameters)
            else:
                return False
        diff = np.abs(reference.astype(np.float32) - self.target.astype(np.float32))
        mae = np.mean(diff)
        if mae <= epsilon:
            return True
        return False


class StateParser(ABC):
    """
    Abstract base class for parsing game state variables from the GameBoy emulator.
    """
    def __init__(self, pyboy, parameters, named_screen_regions: Optional[list[NamedScreenRegion]] = None):
        """
        Initializes the StateParser.
        Args:
            pyboy: An instance of the PyBoy emulator.
            parameters: A dictionary of parameters for configuration.
            named_screen_regions (Optional[list[NamedScreenRegion]]): A list of NamedScreenRegion objects for easy access to specific screen regions.
        """
        verify_parameters(parameters)
        self._parameters = parameters
        if not isinstance(pyboy, PyBoy):
            log_error("pyboy must be an instance of PyBoy", self._parameters)
        self._pyboy = pyboy
        self.named_screen_regions: dict[str, NamedScreenRegion] = {}
        """ Dictionary of NamedScreenRegion objects for easy access to specific screen regions. """
        if named_screen_regions is not None:
            for region in named_screen_regions:
                if not isinstance(region, NamedScreenRegion):
                    log_error(f"named_screen_regions must be a list of NamedScreenRegion objects. Found {type(region)}", self._parameters)
                if region.name in self.named_screen_regions:
                    log_error(f"Duplicate named screen region: {region.name}", self._parameters)
                self.named_screen_regions[region.name] = region

    def bit_count(self, bits: int) -> int:
        """
        Counts the number of set bits (1s) in the given integer.
        Args:
            bits (int): The integer to count set bits in.
        Returns:
            int: The number of set bits.
        """
        return bin(bits).count("1")    
    
    def read_m(self, addr: bytes) -> int:
        """
        Reads a byte from the specified memory address.
        Args:
            addr (int): The memory address to read from.
        Returns:
            int: The byte value at the specified memory address.
        """
        #return self.pyboy.get_memory_value(addr)
        return self._pyboy.memory[addr]

    def read_bits(self, addr) -> str:
        """
        Reads a memory address and returns the result as a binary string. Adds padding so that reading bit 0 works correctly. 
        Args:
            addr (int): The memory address to read from.
        Returns:
            str: The binary string representation of the byte at the specified memory address.
        """
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))

    def read_bit(self, addr, bit: int) -> bool:
        """
        Reads a specific bit from a memory address.
        Args:
            addr (int): The memory address to read from.
            bit (int): The bit position to read (0-7).
        Returns:
            bool: True if the bit is set (1), False otherwise.
        """
        # add padding so zero will read '0b100000000' instead of '0b0'
        return self.read_bits(addr)[-bit - 1] == "1"
    
    def read_m_bit(self, addr_bit: str) -> bool:
        """
        Reads a specific addr-bit string from a memory address. 
        Args:
            addr_bit (str): The - concatenation of a memory address and the bit position (e.g. '0xD87D-5')
        Returns:
            bool: True if the bit at that memory address is set (1), False otherwise
        """
        if "-" not in addr_bit:
            log_error(f"Incorrect format addr_bit: {addr_bit}", self._parameters)
        addr, bit = addr_bit.split("-")        
        flag = False
        try:
            addr = eval(addr)
        except:
            flag = True
        if flag:
            log_error(f"Could not eval byte string: {addr}. Check format", self._parameters)
        if not bit.isdigit():
            log_error(f"bit {bit} is not digit", self._parameters)
        bit = int(bit)
        return self.read_bit(addr, bit)

    def get_raised_flags(self, item_dict: dict) -> set:
        """
        Reads a dictionary of the form {flag_name: memory_address-bit} and returns a set of all flag names that are currently raised (i.e. the bit at the memory address is 1).
        Args:
            item_dict (dict): A dictionary mapping flag names to memory address-bit strings.
        Returns:
            set: A set of flag names that are currently raised.
        """
        items = set()
        for item_name, slot in item_dict.items():
            if self.read_m_bit(slot):
                items.add(item_name)
        return items
    
    def get_current_frame(self) -> np.ndarray:
        """
        Reads the pyboy screen and returns a full resolution numpy array
        
        Returns:
            np.ndarray: The rendered image as a numpy array.
        """
        screen = self._pyboy.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        return screen
    
    def capture_box(self, current_frame: np.ndarray, start_x: int, start_y: int, width: int, height: int) -> np.ndarray:
        """
        Captures a rectangular region from the current frame.

        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            start_x (int): The starting x-coordinate of the region.
            start_y (int): The starting y-coordinate of the region.
            width (int): The width of the region.
            height (int): The height of the region.
        Returns:
            np.ndarray: The captured rectangular region.
        """
        # first check that the box is within the frame
        end_x = start_x + width
        end_y = start_y + height
        if start_x < 0 or start_y < 0 or end_x > current_frame.shape[1] or end_y > current_frame.shape[0]:
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(current_frame.shape[1], end_x)
            end_y = min(current_frame.shape[0], end_y)
        return current_frame[start_y:end_y, start_x:end_x, :]
     
    def capture_square_centered(self, current_frame: np.ndarray, center_x: int, center_y: int, box_size: int) -> np.ndarray:
        """
        Captures a square region from the current frame centered at (center_x, center_y) with the given box size.
        
        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            center_x (int): The x-coordinate of the center of the square.
            center_y (int): The y-coordinate of the center of the square.
            box_size (int): The size of the square box to capture.
        
        Returns:
            np.ndarray: The captured square region.
        """
        half_box = box_size // 2
        start_x = max(center_x - half_box, 0)
        end_x = min(center_x + half_box, current_frame.shape[1])
        start_y = max(center_y - half_box, 0)
        end_y = min(center_y + half_box, current_frame.shape[0])
        return current_frame[start_y:end_y, start_x:end_x, :]
    
    def draw_box(self, current_frame: np.ndarray, start_x: int, start_y: int, width: int, height: int, color: tuple = (0, 0, 0), thickness: int = 1) -> np.ndarray:
        """
        Draws a rectangle on the current frame.

        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            start_x (int): The starting x-coordinate of the rectangle.
            start_y (int): The starting y-coordinate of the rectangle.
            width (int): The width of the rectangle.
            height (int): The height of the rectangle.
            color (tuple, optional): The color of the rectangle in BGR format.
            thickness (int, optional): The thickness of the rectangle border. 

        Returns:
            np.ndarray: The frame with the drawn rectangle.
        """
        end_x = start_x + width
        end_y = start_y + height
        if start_x < 0 or start_y < 0 or end_x > current_frame.shape[1] or end_y > current_frame.shape[0]:
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(current_frame.shape[1], end_x)
            end_y = min(current_frame.shape[0], end_y)
        frame_with_box = current_frame.copy()
        cv2.rectangle(frame_with_box, (start_x, start_y), (end_x, end_y), color, thickness)
        return frame_with_box
    
    def draw_square_centered(self, current_frame: np.ndarray, center_x: int, center_y: int, box_size: int, color: tuple = (0, 0, 0), thickness: int = 1) -> np.ndarray:
        """
        Draws a square on the current frame centered at (center_x, center_y) with the given box size.
        
        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            center_x (int): The x-coordinate of the center of the square.
            center_y (int): The y-coordinate of the center of the square.
            box_size (int): The size of the square box to draw.
            color (tuple, optional): The color of the square in BGR format. 
            thickness (int, optional): The thickness of the square border.
        
        Returns:
            np.ndarray: The frame with the drawn square.
        """
        half_box = box_size // 2
        start_x = max(center_x - half_box, 0)
        end_x = min(center_x + half_box, current_frame.shape[1])
        start_y = max(center_y - half_box, 0)
        end_y = min(center_y + half_box, current_frame.shape[0])
        frame_with_square = current_frame.copy()
        cv2.rectangle(frame_with_square, (start_x, start_y), (end_x, end_y), color, thickness)
        return frame_with_square
    
    def capture_named_region(self, current_frame: np.ndarray, name: str) -> np.ndarray:
        """
        Captures a named region from the current frame.

        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            name (str): The name of the region to capture.

        Returns:
            np.ndarray: The captured region.
        """
        if name not in self.named_screen_regions:
            log_error(f"Named screen region {name} not found.", self._parameters)
        region = self.named_screen_regions[name]
        x, y, w, h = region.start_x, region.start_y, region.width, region.height
        return self.capture_box(current_frame, x, y, w, h)
    
    def named_region_matches_target(self, current_frame: np.ndarray, name: str) -> bool:
        """
        Compares a named region from the current frame to its target image using Absolute Error (AE).

        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            name (str): The name of the region to compare.
        Returns:
            bool: True if the region matches the target image, False otherwise.
        """
        if name not in self.named_screen_regions:
            log_error(f"Named screen region {name} not found.", self._parameters)
        region = self.named_screen_regions[name]
        captured_region = self.capture_named_region(current_frame, name)
        return region.matches_target(captured_region)

    def draw_named_region(self, current_frame: np.ndarray, name: str, color: tuple = (0, 0, 0), thickness: int = 1) -> np.ndarray:
        """
        Draws a named region on the current frame.

        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            name (str): The name of the region to draw.
            color (tuple, optional): The color of the rectangle in BGR format. 
            thickness (int, optional): The thickness of the rectangle border.

        Returns:
            np.ndarray: The frame with the drawn rectangle.
        """
        if name not in self.named_screen_regions:
            log_error(f"Named screen region {name} not found.", self._parameters)
        region = self.named_screen_regions[name]
        x, y, w, h = region.start_x, region.start_y, region.width, region.height
        return self.draw_box(current_frame, x, y, w, h, color, thickness)
    
    def draw_grid_overlay(self, current_frame: np.ndarray, grid_skip: int=20) -> np.ndarray:
        """
        Draws a grid overlay on the current frame for easier region identification.
        Args:
            current_frame (np.ndarray): The current frame from the emulator.
            grid_skip (int, optional): The number of pixels between grid lines. 
        Returns:
            np.ndarray: The frame with the grid overlay.            
        """
        frame_with_grid = current_frame.copy()
        for x in range(0, current_frame.shape[1], grid_skip):
            cv2.line(frame_with_grid, (x, 0), (x, current_frame.shape[0]), (0, 0, 255), 1, lineType=cv2.LINE_AA)
        for y in range(0, current_frame.shape[0], grid_skip):
            cv2.line(frame_with_grid, (0, y), (current_frame.shape[1], y), (0, 0, 255), 1, lineType=cv2.LINE_AA)
        return frame_with_grid
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Name of the parser for logging purposes.
        :return: string name of the parser
        """
        raise NotImplementedError    

# TODO: Think about design for this.
class StateTracker(ABC):
    def __init__(self, name: str, session_name: str, instance_id: str, state_parser: StateParser, parameters: dict):
        """
        Initializes the StateTracker.
        Args:
            name (str): Name of the game.
            session_name (str): Name of the session.
            instance_id (str): Unique identifier for this environment instance.
            state_parser (StateParser): An instance of the StateParser to parse game state variables.
            parameters (dict): A dictionary of parameters for configuration.
        """
        verify_parameters(parameters)
        self.name = name
        """ Name of the game. """
        self.session_name = session_name
        """ Name of the session. """
        self.instance_id = instance_id
        """ Unique identifier for this environment instance. """
        self.state_parser = state_parser
        """ An instance of the StateParser to parse game state variables. """
        self._parameters = parameters
        self.metrics = {}
        self.reset()
        """ Dictionary to store tracked metrics. TODO: Review design. """

    def reset(self):
        """
        Is called once per environment reset.

        At this level, only manages the step counter
        """
        self.metrics["steps"] = 0

    def step(self):
        """
        Is called once per environment step to update any tracked metrics.

        At this level, only manages the step counter
        """
        self.metrics["steps"] =  self.metrics.get("steps", 0) + 1
    
    @abstractmethod
    def close(self):
        """
        Is called once when the environment is closed to finalize any tracked metrics.
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"<StateTracker(name={self.name}, session_name={self.session_name}, instance_id={self.instance_id})>"

    def __str__(self) -> str:
        start = f"***\t{self.__repr__()}\t***"
        body = nested_dict_to_str(self.metrics, indent=1)
        return f"{start}\n{body}"



class Emulator():
    def __init__(self, name: str, gb_path: str, state_parser_class: Type[StateParser], state_tracker_class: Type[StateTracker], init_state: str, parameters: dict, headless: bool = True, max_steps: int = None, save_video: bool = None, session_name: str = None, instance_id: str = None):
        """
        Start the GameBoy emulator with the given ROM file and initial state.

        Args:
            name (str): Name of the emulator instance.
            gb_path (str): Path to the GameBoy ROM file.
            state_parser_class (Type[StateParser]): A class that inherits from StateParser to parse game state variables.
            state_tracker_class (Type[StateTracker]): A class that inherits from StateTracker to track game state metrics.
            init_state (str): Path to the initial state file to load.
            parameters (dict): Dictionary of parameters for the environment.
            headless (bool, optional): Whether to run the environment in headless mode. 
            max_steps (int, optional): Maximum number of steps per episode. 
            save_video (bool, optional): Whether to save video of the episodes.          
            session_name (str, optional): Name of the session. If None, a new session name will be allocated.   
        """
        verify_parameters(parameters)
        self._parameters = parameters
        if name is None or name == "":
            log_error("You must provide a name for the emulator instance.", self._parameters)
        if gb_path is None:
            log_error("You must provide a path to the GameBoy ROM file.", self._parameters)
        if not issubclass(state_parser_class, StateParser):
            log_error("state_parser_class must be a subclass of StateParser.", self._parameters)
        if not issubclass(state_tracker_class, StateTracker):
            log_error("state_tracker_class must be a subclass of StateTracker.", self._parameters)
        if init_state is None:
            log_error("You must provide an initial state file to load.", self._parameters)
        if headless not in [True, False]:
            log_error("headless must be a boolean.", self._parameters)
        self.name = name
        """ Name of the emulator (does not need to be unique across instances, e.g. 'PokemonRed'). """
        self._gb_path = gb_path
        self._set_init_state(init_state)
        # validate init_state exists and ends with .state
        if not os.path.exists(self._gb_path):
            log_error(f"GameBoy ROM file {self._gb_path} does not exist. You must obtain a ROM through official means, and then place it in the path: {self._gb_path}", self._parameters)
        if not self._gb_path.endswith(".gb") and not self._gb_path.endswith(".gbc"):
            log_error(f"GameBoy ROM file {self._gb_path} is not a .gb or .gbc file.", self._parameters)
        self.headless = headless
        """ Whether to run the environment in headless mode."""
        if max_steps is None:
            max_steps = self._parameters["gameboy_max_steps"]
        if max_steps > self._parameters["gameboy_hard_max_steps"]:
            log_warn(f"max_steps {max_steps} exceeds gameboy_hard_max_steps {self._parameters['gameboy_hard_max_steps']}. Setting to hard max.", self._parameters)
            max_steps = self._parameters["gameboy_hard_max_steps"]
        self.max_steps = max_steps
        """ Maximum number of steps per episode. """
        if save_video is None:
            save_video = self._parameters["gameboy_default_save_video"]
        self.save_video = save_video
        """ Whether to save video of the episodes. """
        self.session_name = None
        if session_name is None:
            session_name = self._allocate_new_session_name()
        elif not isinstance(session_name, str) or session_name == "":
            log_error(f"session_name must be a non-empty string. Recieved {session_name}", self._parameters)
        self.session_name = session_name
        """ Name of the session. Decides the directory where artifacts are saved. """
        self.session_path = os.path.join(self.get_session_path(), self.session_name)
        """ Path to the session directory. This is where all artifacts for this session are saved. """
        os.makedirs(self.session_path, exist_ok=True)
        will_save_artifacts = self.save_video # may add more artifact types later. Tracker will also save metrics to disk perhaps
        if will_save_artifacts and self._is_digit_session(self.session_name):
            log_warn(f"Session name {self.session_name} is in the form session_X where X is an integer. This is the pattern for an automatically assigned session and it will be cleared every time you start a new emulator. Name the session to make sure its saved artifacts persist.", self._parameters)
        if instance_id is None:
            instance_id = str(uuid.uuid4())[:8]
        self.instance_id = instance_id
        """ Unique identifier for this environment instance. Useful for distinguishing multiple environments running in parallel. """
        self.act_freq = parameters["gameboy_action_freq"]
        """ Number of emulator ticks per action. Defaults to value specified in config files. """
        self.press_step = parameters["gameboy_press_step"]
        """ Number of emulator ticks to hold down a button press. Defaults to value specified in config files. """
        self.render_headless = parameters["gameboy_headless_render"]
        """ Whether to render the emulator screen even in headless mode. This must be true for methods that rely on image observations (e.g. VLMs) to access the screen. Defaults to value specified in config files. """
        if not self.render_headless:
            log_error("render_headless cannot be set to False. In the Pokemon environments, screen captures are used aggressively to determine state. ", self._parameters)

        self._full_frame_writer = None
        self._model_frame_writer = None
        self.reset_count = 0
        """ Number of times the environment has been reset. """
        self.step_count = 0
        """ Number of steps taken in the current episode. """
        self._reduce_video_resolution = parameters["gameboy_reduce_video_resolution"]
        pokemon_frame_size = (160, 144) # Confirm this is universal if you want other GB games. 
        if self._reduce_video_resolution:
            self.output_shape = (pokemon_frame_size[0]//2, pokemon_frame_size[1]//2)
        else:
            self.output_shape = (pokemon_frame_size[0], pokemon_frame_size[1])
            """ Shape of the output observations. This is the resolution of the rendered screen. """

        head = "null" if self.headless else "SDL2"

        self._pyboy = PyBoy(
            self._gb_path,
            window=head,
        )
        self.state_parser = state_parser_class(self._pyboy, self._parameters)
        """ Instance of the StateParser to parse game state variables. """

        self.state_tracker = state_tracker_class(self.name, self.session_name, self.instance_id, self.state_parser, self._parameters)
        """ Instance of the StateTracker to track game state metrics. """

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not self.headless:
            if not is_none_str(self._parameters["gameboy_headed_emulation_speed"]):
                self._pyboy.set_emulation_speed(int(self._parameters["gameboy_headed_emulation_speed"]))        
            
    @staticmethod
    def create_first_state(gb_path: str, state_path: str):
        """
        Creates a basic state for the emulator. This can be used to create an initial, default state file for a new game.

        Warning: This method uses parameter free logging, so if you override the log_file with a command prompt argument, it will be ignored here.

        Args:
            gb_path (str): Path to the GameBoy ROM file.
            state_path (str): Path to save the initial state file.
        """
        # error out if gb_path does not exist or is not a .gb or .gbc file
        if not os.path.exists(gb_path):
            log_error(f"GameBoy ROM file {gb_path} does not exist. You must obtain a ROM through official means, and then place it in the path: {gb_path}")
        if not gb_path.endswith(".gb") and not gb_path.endswith(".gbc"):
            log_error(f"GameBoy ROM file {gb_path} is not a .gb or .gbc file.")
        if not state_path.endswith(".state"):
            state_path = state_path + ".state"
        if os.path.exists(state_path):
            log_error(f"State file {state_path} already exists. Will not overwrite...")
        file_makedir(state_path)
        pyboy = PyBoy(
            gb_path,
            window="null",
        )
        with open(state_path, "wb") as f:
            pyboy.save_state(f)
        pyboy.stop()
        log_info(f"Created initial state file at {state_path}")
    
    def _is_digit_session(self, session_name: str) -> bool:
        """
        Checks if the given session name is in the form session_X where X is an integer.

        Args:
            session_name (str): The session name to check.
        """
        if not session_name.startswith("session_"):
            return False
        pattern = r"^session_(\d+)$" # This is AI generated. I hope it works. It knows regex better than I do.
        match = re.match(pattern, session_name)
        if match:
            return True
        return False


    def _allocate_new_session_name(self) -> str:
        """
        Allocates a new session name based on existing sessions in the session directory.
        :return: new session name
        """
        self.clear_unnamed_sessions()
        storage_dir = self._parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
        os.makedirs(session_path, exist_ok=True)
        existing_sessions = os.listdir(session_path)
        # session names are always in the form session_X where X is an integer starting from 0
        session_indices = [int(name.split("_")[-1]) for name in existing_sessions if name.startswith("session_") and name.split("_")[-1].isdigit()]
        if len(session_indices) == 0:
            new_index = 0
        else:
            new_index = max(session_indices) + 1
        new_session_name = f"session_{new_index}"
        return new_session_name

    def clear_unnamed_sessions(self):
        """
        Clears all unnamed (integer based) sessions from the session directory.
        """
        storage_dir = self._parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
        if not os.path.exists(session_path):
            return
        existing_sessions = os.listdir(session_path)
        saved_sessions = []
        if self.session_name is not None:
            saved_sessions.append(self.session_name)
        for session in existing_sessions:
            if session in saved_sessions:
                continue
            if self._is_digit_session(session):
                full_path = os.path.join(session_path, session)
                # delete the session directory and all its contents
                import shutil
                shutil.rmtree(full_path)
                log_info(f"Deleted unnamed session {full_path}", self._parameters)
                continue
            saved_sessions.append(session)
        log_info(f"Deleted {len(existing_sessions) - len(saved_sessions)} sessions. Kept {len(saved_sessions)} sessions: \n{saved_sessions}", self._parameters)

    def get_session_path(self) -> str:
        """
        Returns the path to the session directory for this environment variant.
        :return: path to the session directory
        """
        storage_dir = self._parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
        os.makedirs(session_path, exist_ok=True)
        return session_path

    def set_init_state(self, init_state: str):
        """Sets a new initial state file for the environment. and resets the environment.

        Args:
            init_state (str): Path to the new initial state file.
        """
        self._set_init_state(init_state)
        self.reset()
    
    def _set_init_state(self, init_state: str):
        """
        Sets a new initial state file for the environment to eventually load. 
        Does not reset the environment. 

        Args:
            init_state (str): Path to the new initial state file.
        """
        if init_state is None:
            return
        if not os.path.exists(init_state):
            log_error(f"New initial state file {init_state} does not exist.", self._parameters)
        if not init_state.endswith(".state"):
            log_error(f"New initial state file {init_state} is not a .state file.", self._parameters)
        self.init_state = init_state
        log_info(f"Set new initial state file to {self.init_state}", self._parameters)

    def reset(self, new_init_state: str = None):
        """
        Resets the environment to the initial state. Optionally loads a new initial state file.

        Args:
            new_init_state (str, optional): Path to a new initial state file to load. 
        """
        # validate the new_init_state if provided
        if new_init_state is not None:
            self._set_init_state(new_init_state)
        # restart game, skipping to init_state 
        with open(self.init_state, "rb") as f:
            self._pyboy.load_state(f)

        self.reset_count += 1
        self.step_count = 0
        self.state_tracker.reset()
        self.close_video()
        return

    def get_current_frame(self, reduce_res: bool = None) -> np.ndarray:
        """
        Renders the currently rendered screen of the emulator and returns it as a numpy array.
        Args:
            reduce_res (bool, optional): Whether to reduce the resolution of the rendered image. Defaults to gameboy_reduce_video_resolution.
        Returns:
            np.ndarray: The rendered image as a numpy array.
        """ 
        game_pixels_render = self._pyboy.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        if reduce_res is None:
            reduce_res = self._reduce_video_resolution
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def step(self, action: LowLevelActions = None) -> bool:
        """ 
        
        Takes a step in the environment by performing the given action on the emulator.
        If saving video, starts the video recording on the first step.

        Args:
            action (LowLevelActions, optional): Lowest level action to perform on the emulator.

        Returns:
            bool: Whether the max_steps limit is reached.
        """
        if action is not None:
            if action not in LowLevelActions:
                log_error(f"Invalid action {action}. Must be one of {list(LowLevelActions)} or None", self._parameters)
        if self.step_count >= self.max_steps:
            log_error("Step called after max_steps reached. Please reset the environment.", self._parameters)
            
        if self.save_video and self.step_count == 0:
            self.start_video()
        
        if self.save_video or self.video_running: # TODO: Consider alternative ways of handling this
            self.add_video_frame()

        frames = self.run_action_on_emulator(action)

        self.step_count += 1
        self.state_tracker.step()
        return frames, self.check_if_done()

    def get_state_parser(self) -> StateParser:
        """
        Returns the current game state parser instance.

        Returns:
            StateParser: The current game state parser.
        """
        return self.state_parser

    def run_action_on_emulator(self, action: LowLevelActions = None, profile: bool = False, render: bool = True) -> Optional[np.ndarray]:
        """ 
        
        Performs the given action on the emulator by pressing and releasing the corresponding button.

        Args:
            action (LowLevelActions): Lowest level action to perform on the emulator.
            profile (bool, optional): Whether to profile the action execution time. 
            render (bool, optional): Whether to render the emulator screen during action execution. 
        Returns:
            Optional[np.ndarray]: The stack of frames that passed while performing the action, if rendering is enabled. Is of shape [n_frames (3 right now), height, width, channels]. Otherwise, None.
        """
        #log_info(f"Running action: {action}", self.parameters)        
        frames = None
        if action is not None:
            if render == True:
                frames = []
            if profile:
                start_time = perf_counter()
            self._pyboy.send_input(action.value)
            if profile:
                end_time = perf_counter()
                # Action LowLevelActions.PRESS_ARROW_LEFT took 0.01 ms or 0.00 s

            # disable rendering when we don't need it
            if render is not None:
                render_screen = render
            else:
                render_screen = self.save_video or not self.headless or self.render_headless 
            press_step = self.press_step
            self._pyboy.tick(press_step, render_screen)
            if render:
                frames.append(self.get_current_frame(reduce_res=False))
            if profile:
                start_time = perf_counter()
            self._pyboy.send_input(ReleaseActions.release_actions.value[action])
            if profile:
                mid_time = perf_counter()
            self._pyboy.tick(self.act_freq - press_step - 1, render_screen)
            if render:
                frames.append(self.get_current_frame(reduce_res=False))
            if profile:
                end_time = perf_counter()
            # Releasing action LowLevelActions.PRESS_ARROW_LEFT took 0.00 ms, followed by 16.13 ms for remaining ticks
            self._pyboy.tick(1, True)
            if render:
                frames.append(self.get_current_frame(reduce_res=False))
                frames = np.stack(frames, axis=0)
        else:
            log_warn("No action provided to run_action_on_emulator. Skipping action. You probably should only ever use this in debugging mode. Idk maybe wait is a command.", self._parameters)
            self._pyboy.tick(self.act_freq, True)
        return frames
            
    def get_free_video_id(self) -> str:
        """
        Returns a new unique video ID for saving video files.

        Returns:
            str: A new unique video ID.
        """
        base_dir = os.path.join(self.session_path, "videos")
        videos = os.listdir(base_dir) if os.path.exists(base_dir) else []
        # all will be something.mp4, if its int.mp4, get the int
        video_ints = []
        for video in videos:
            if video.endswith(".mp4"):
                video_name = video[:-4]
                if video_name.isdigit():
                    video_ints.append(int(video_name))
        if len(video_ints) == 0:
            return "0.mp4"
        return str(max(video_ints) + 1)+".mp4"
        
    def start_video(self, video_id: str = None):
        """ 
        Starts recording video of the emulator's screen.
        Args:
            video_id (str, optional): Name of the video file to save. If None, a new unique name will be generated. 
        """
        if video_id is not None:
            if not isinstance(video_id, str):
                log_error("video_id must be a string (not digits) if provided.", self._parameters)
            if not video_id.endswith(".mp4"):
                log_error("video_id must end with .mp4 if provided.", self._parameters)
            if os.path.exists(os.path.join(self.session_path, "videos", video_id)):
                log_warn(f"video_id {video_id} already exists. Overwriting...", self._parameters)                       
        else:
            video_id = self.get_free_video_id()
        base_dir = os.path.join(self.session_path, "videos")
        os.makedirs(base_dir, exist_ok=True)
        video_path = os.path.join(base_dir, f"{video_id}")
        self.close_video()
        self.frame_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (self.output_shape[0], self.output_shape[1]), isColor=False)
        self.video_running = True
        log_info(f"Started recording video to: {video_path}", self._parameters)

    def add_video_frame(self):
        """
        Adds the current frame of the emulator to the video being recorded.
        """
        current_frame = self.get_current_frame(reduce_res=self._reduce_video_resolution)[:, :, 0]
        # frame_size = (current_frame.shape[1], current_frame.shape[0]) # Width, Height, should be equal to self.output_shape
        # Create VideoWriter object
        frame = current_frame
        self.frame_writer.write(frame)
                    
    def check_if_done(self):
        """
        Checks if the max_steps limit has been reached.
        """
        done = self.step_count >= self.max_steps - 1
        return done

    def close_video(self):
        """
        Closes the video writer and stops recording video.
        """
        if self._full_frame_writer is not None:
            self._full_frame_writer.release()
            self._full_frame_writer = None
        if self._model_frame_writer is not None:
            self._model_frame_writer.release()
            self._model_frame_writer = None
        self.video_running = False
    
    def close(self):
        """
        Closes the emulator and any associated resources.
        If the session directory is empty after closing, it will be deleted.
        """
        self._pyboy.stop(save=False)
        self.close_video()
        self.state_tracker.close()
        # check if session directory is empty, and if so delete it
        if os.path.exists(self.session_path) and len(os.listdir(self.session_path)) == 0:
            os.rmdir(self.session_path)
    
    def human_play(self, max_steps: int = None):
        """ 
        Allows a human to play the emulator using keyboard inputs.
        Args:
            max_steps (int, optional): Maximum number of steps to play. Defaults to gameboy_hard_max_steps in configs.
        """
        if max_steps is None:
            max_steps = self._parameters["gameboy_hard_max_steps"]
        log_info("Starting human play mode. Use arrow keys and A(a)/B(s)/Start(enter) buttons to play. Close the window to exit.", self._parameters)
        if self.headless:
            log_error("Human play mode requires headless=False. Change the initialization", self._parameters)    
        self.reset()
        while True:
            self._pyboy.tick(1, True)
            self.state_tracker.step()
            if self.step_count >= max_steps:
                break
        self.close()

    def random_play(self, max_steps: int = None):
        """ 
        Allows the emulator to play itself using (sort of) random actions.
        Args:
            max_steps (int, optional): Maximum number of steps to play. Defaults to gameboy_hard_max_steps in configs.
        """
        if max_steps is None:
            max_steps = self._parameters["gameboy_random_play_max_steps"]
        log_info("Starting random play mode.", self._parameters)
        self.reset()
        pbar = tqdm(total=max_steps, desc="Random Play Steps")
        allowed_actions = list(LowLevelActions)
        # remove the Start and Select actions from allowed actions to avoid menu spamming. 
        allowed_actions.remove(LowLevelActions.PRESS_BUTTON_START)
        while self.step_count < max_steps:
            action = np.random.choice(allowed_actions)
            frames, done = self.step(action)
            pbar.update(1)
            if done:
                break
        pbar.close()
        self.close()
        log_info("Random play mode ended.", self._parameters)

    def _dev_play(self, max_steps: int = None):
        """
        Allows a human to play the emulator using keyboard inputs. Does not route through step function. 
        This function continuously reads from the parameters in the configs directory and if it detects a change in the `gameboy_dev_play_stop` parameter, will enter a breakpoint

        Args:
            max_steps (int, optional): Maximum number of steps to play. Defaults to gameboy_hard_max_steps in configs.
        """
        if not hasattr(self.state_parser, "rom_data_path"):
            log_error("Development play mode requires a StateParser with rom_data_path attribute.", self._parameters)
        if max_steps is None:
            max_steps = self._parameters["gameboy_hard_max_steps"]
        log_info("Starting human play mode. Use arrow keys and A(a)/B(s)/Start(enter) buttons to play. Close the window to exit. Open configs/gameboy_vars.yaml and set gameboy_dev_play_stop to true to enable development mode.", self._parameters)
        if self.headless:
            log_error("Human play mode requires headless=False. Change the initialization", self._parameters)
        self.reset()
        valid_regions = list(self.state_parser.named_screen_regions.keys())
        unassigned_regions = [name for name in valid_regions if self.state_parser.named_screen_regions[name].target is None]
        log_info(f"Unassigned regions (target array not set) are: {unassigned_regions}", self._parameters)
        while True:
            self._parameters = load_parameters()
            if not self._parameters["gameboy_dev_play_stop"]:
                self._pyboy.tick(1, True)
                self.state_tracker.step()
            else:
                dev_instructions = f"""
                In development mode.
                Enter 'e' to close the emulator.
                Enter '' to re-enter normal play mode (remember to change gameboy_dev_play_stop back to false in configs or it'll stop again). 
                Enter 's <state_name>' to save the current state as a .state file.
                Enter 'l <state_name>' to load a .state file.
                Enter 'c <region_name> <save_name / None if region.target_path is set>' to capture a named region and save it as a .npy file.
                Enter 'd <None / region_name>' to draw a named region and display the current screen with the region drawn.
                Enter 'b' to enter a breakpoint.
                Valid region names are: {valid_regions}
                Initially unassigned regions (target array not set) were: {unassigned_regions}\n\t Note: This list does not update as you assign targets during this session.
                Current State: 
                {str(self.state_tracker)}
                """
                log_info(dev_instructions, self._parameters)
                user_input = input("Dev mode input: ")
                user_input = user_input.lower().strip()
                first_char = user_input[0] if len(user_input) > 0 else ""
                allowed_inputs = ["e", "", "c", "s", "l", "d", "b"]
                if first_char not in allowed_inputs:
                    log_warn(f"Invalid input {user_input}. Valid inputs are: {allowed_inputs}", self._parameters)
                    continue
                if first_char == "e":
                    log_info("Exiting human play mode.", self._parameters)
                    break
                elif first_char == "":
                    log_info("Exiting development mode. Resuming normal play.", self._parameters)
                    continue
                elif first_char == "s" or first_char == "l":
                    parts = user_input.split(" ")
                    if len(parts) != 2:
                        log_warn(f"Invalid input {user_input}.", self._parameters)
                        continue
                    state_name = parts[1]
                    if not state_name.endswith(".state"):
                        state_name = state_name + ".state"
                    state_path = os.path.join(self.state_parser.rom_data_path, "states", state_name)
                    if first_char == "s":
                        if os.path.exists(state_path):
                            confirm_input = input(f"State file {state_path} already exists. Overwrite? (y/n): ")
                            if confirm_input.lower().strip() != "y":
                                log_info("Aborting save state.", self._parameters)
                                continue
                        self.save_state(state_path)
                    else:
                        if not os.path.exists(state_path):
                            log_warn(f"State file {state_path} does not exist. Cannot load.", self._parameters)
                            continue
                        self.set_init_state(state_path)
                elif first_char == "b":
                    breakpoint()
                else:
                    current_frame = self.get_current_frame(reduce_res=False)                    
                    # draw it even if c, so we can see what we're capturing
                    save_path = None
                    if first_char == "c":
                        parts = user_input.split(" ")
                        if len(parts) != 3:
                            if len(parts) != 2:
                                log_warn(f"Invalid input {user_input}.", self._parameters)
                                continue
                            else:                                
                                region_name = parts[1]
                                region = self.state_parser.named_screen_regions[region_name]
                                save_path = region.target_path
                                if save_path is None:
                                    log_warn(f"Region {region_name} does not have a target path specified. Please provide a save name.", self._parameters)
                                    continue
                        else:
                            save_name = parts[2]
                            if not save_name.endswith(".npy"):
                                save_name = save_name + ".npy"
                                save_path = os.path.join(self.state_parser.rom_data_path, "captures", save_name)
                        file_makedir(save_path)
                    elif first_char == "d":
                        parts = user_input.split(" ")
                        if len(parts) == 2:
                            region_name = parts[1]
                        elif len(parts) == 1:
                            region_name = "Full Screen"
                        else:
                            log_warn(f"Invalid input {user_input}.", self._parameters)
                            continue
                    if region_name == "Full Screen":
                        drawn_frame = current_frame
                    else:
                        drawn_frame = self.state_parser.draw_named_region(current_frame, region_name)
                    plt.imshow(drawn_frame[:, :, 0], cmap="gray")
                    plt.title(f"Region: {region_name}")
                    plt.show()
                    if first_char == "c":
                        captured_region = self.state_parser.capture_named_region(current_frame, region_name)
                        plt.imshow(captured_region[:, :, 0], cmap="gray")
                        plt.title(f"Captured Region: {region_name}")
                        plt.show()
                        existing_file = os.path.exists(save_path)
                        existing_str = "" if not existing_file else " (will overwrite existing file)"
                        confirmation_input = input(f"Save captured region {region_name} to {save_path}? (y/n) {existing_str}: ")
                        if confirmation_input.lower().strip() != "y":
                            log_info("Aborting capture region.", self._parameters)
                            continue
                        np.save(save_path, captured_region)
                        log_info(f"Saved captured region {region_name} to {save_path}", self._parameters)
            if self.step_count >= max_steps:
                break
        self.close()
        
    def save_render(self):
        """
        Saves the current rendered screen of the emulator as a JPEG image in the renders directory.
        """
        render_path = os.path.join(self.session_path, "renders", f"step_{self.step_count}_id{self.instance_id}.jpeg")
        file_makedir(render_path)
        plt.imsave(render_path, self.render(reduce_res=False)[:,:, 0])

    def save_state(self, state_path: str):
        """
        Saves the current state of the emulator to a .state file.
        Args:
            state_path (str): Path to save the .state file.

        """
        if not state_path.endswith(".state"):
            state_path = state_path + ".state"
        file_makedir(state_path)
        with open(state_path, "wb") as f:
            self._pyboy.save_state(f)
        log_info(f"Saved state to {state_path}", self._parameters)

    def _sav_to_state(self, sav_file: Optional[str], state_file: str):
        """
        Loads a .sav file into the emulator and saves the corresponding .state file. 
        Use this if you want to manually create .sav files and convert them to .state files for use as initial states.

        Args:
            save_file (str or None): Path to the .sav file to load. If None, looks for a .sav file in the same directory as the ROM with the same base name.
            state_file (str): Path to save the .state file
        """
        log_info("Trying to find .sav file and convert to .state file. This is a breaking operation, so the program will terminate after its completion.", self._parameters)
        if sav_file is not None:
            expected_sav = sav_file
        else:
            if ".gbc" in self._gb_path:
                expected_sav = self._gb_path.replace(".gbc", ".sav")
            else:
                expected_sav = self._gb_path.replace(".gb", ".sav")
        if not os.path.exists(expected_sav):
            log_error(f"Expected .sav file at {expected_sav} to convert to .state file, but it does not exist.", self._parameters)
        if state_file is None or state_file == "":
            log_error("You must provide a state_file to save the .state file.", self._parameters)
        if not state_file.endswith(".state"):
            state_file = state_file + ".state"
        if os.path.exists(state_file):
            log_error(f"state_file {state_file} already exists. Please provide a new path to avoid overwriting.", self._parameters)
        file_makedir(state_file)
        # copy the .sav file to self._gb_path.gb.ram file
        save_destination = self._gb_path.replace(".gb", ".gb.ram")
        shutil.copyfile(expected_sav, save_destination)
        self.close()
        self._pyboy = PyBoy(
            self._gb_path,
            window="null",
        )
        self._pyboy.set_emulation_speed(0)
        self._pyboy.tick(10000, False) # get to opening menu
        self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_A, render=False) # press A to get past opening menu
        self._pyboy.tick(1000, False) # wait for load
        self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_A, render=False) # press A to load game
        self._pyboy.tick(1000, False) # wait for file select
        self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_A, render=False) # press A to confirm load
        self._pyboy.tick(5000, False) # wait for game to load
        self.save_state(state_file)
        self._pyboy.stop(save=False)
        log_info("State saved successfully. Exiting now to avoid issues ...", self._parameters)
        # remove the .gb.ram file
        if os.path.exists(save_destination):
            os.remove(save_destination)
        sys.exit(0)
            
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        return self.name
    

def bytes_to_padded_hex_string(integer_value):
    """
    Converts a bytes object into a padded, '0x'-prefixed hexadecimal string.
    """
    # 1. Convert the bytes object back into an integer
    # Assumes big-endian order for the example '0x00a' -> 10
    # 2. Format the integer into a string with padding and the '0x' prefix
    # The 'x' specifier for hex, '#' adds '0x', '04' pads to 4 hex characters total
    # (not including the '0x' prefix for simple formatters like this, but managing width)
    
    # A robust approach to match your exact output '0x00a':
    # You generally want enough width for your bytes. b'\n' is 1 byte, 2 hex chars.

    return f'0x{integer_value:04x}' # {0:04x} pads to 4 digits specifically