from poke_worlds.emulation.parser import StateParser
from poke_worlds.utils import nested_dict_to_str, verify_parameters, log_info, log_error, log_warn, log_dict, show_frames


import numpy as np
from typing import Optional, Type, Dict, Any, Tuple, List

from abc import ABC, abstractmethod
from poke_worlds.utils.vlm import ocr

EPSILON = 0.0001
""" Default epsilon for frame change detection. """

class MetricGroup(ABC):
    """Abstract Base class for organizing related metrics."""
    NAME = "base"
    """ Name of the MetricGroup. """

    REQUIRED_PARSER = StateParser
    """ The StateParser which implements the minimum required functionality for this MetricGroup to work. """

    def __init__(self, state_parser: StateParser, parameters: dict):
        verify_parameters(parameters)
        if not issubclass(type(state_parser), self.REQUIRED_PARSER):
            log_error(f"StateParser of type {type(state_parser)} is not compatible with MetricGroup requiring {self.REQUIRED_PARSER}.")
        self.state_parser = state_parser
        """ An instance of the StateParser to parse game state variables. """
        self._parameters = parameters
        self.start()
        self.final_metrics: Dict[str, Any] = None
        """ Dictionary to store final metrics after environment close. """

    def start(self):
        """
        Called once when environment starts. 
        All subclasses should call super() AFTER initializing their own variables.
        Only variables that will persist across episodes should be initialized here.
        """
        self.reset(first=True)

    @abstractmethod    
    def reset(self, first: bool = False):
        """Called when environment resets.
        
        Args:
            first (bool): Whether this is the first reset of the environment. If True, might need to aggregate metrics into running final totals. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        """
        Called when environment closes. Good for computing summary stats.

        Step will not be called after this.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        """
        Called each environment step to update metrics.
        Args:
            current_frame (np.ndarray): The current frame rendered by the emulator.
            recent_frames (Optional[np.ndarray]): The stack of frames that were rendered during the last action. Shape is [n_frames, height, width, channels]. Can be None if rendering is disabled.
        """
        raise NotImplementedError
    
    @abstractmethod
    def report(self) -> dict:
        """
        Return metrics as dictionary for instantaneous variable tracking.

        Subclasses must specify the keys used in the returned dictionary.
        Returns:
            dict: A dictionary containing the current metrics.
        """
        raise NotImplementedError
    
    @abstractmethod
    def report_final(self) -> dict:
        """
        Return metrics as dictionary for logging. Called at end of environment (before close).
        Will never be called before `self.close`.

        Subclasses must specify the keys used in the returned dictionary.
        Returns:
            dict: A dictionary containing the final metrics.
        """
        raise NotImplementedError
    
    def log_info(self, message: str):
        """
        Logs with MetricGroup's name
        """
        log_info(f"[Metric({self.NAME})]: {message}", self._parameters)

    def log_warn(self, message: str):
        """
        Logs with MetricGroup's name
        """
        log_warn(f"[Metric({self.NAME})]: {message}", self._parameters)

    def log_report(self):
        """
        Logs the current metrics report with MetricGroup's name. Primarily for debugging.
        """
        log_info(f"Metric({self.NAME}):\n")
        log_dict(self.report(), parameters=self._parameters) 


class CoreMetrics(MetricGroup):
    """
    Tracks basic metrics available in all games:
    1. steps: Number of steps taken in the episode.
    2. frame_changed:
            For every available recent frame, sees if any of them differ from the previous frame. If so sets frame_changed to True. Is not robust to jitter on screen. 
    3. current_frame
    4. passed_frames (last element is equal to current_frame)
    """
    NAME = "core"

    def start(self):
        self.steps_per_episode = []
        """ List of steps taken in each episode. """
        super().start()
    
    def reset(self, first=False):
        if not first:
            self.steps_per_episode.append(self.steps)
        else:    
            self.steps = 0
            """ Number of steps taken in the episode. """
            self.previous_frame = None
            """ Previous frame for detecting changes. """
            self.current_frame = None
            """ Current frame. """
            self.frame_changed = True
            """ Whether the frame has changed at all since last step. """
            self.passed_frames = None
            """ Stack of frames since the last step """

    def close(self):
        if len(self.steps_per_episode) > 0:
            total_episodes = len(self.steps_per_episode)
            average_steps = np.mean(self.steps_per_episode)
            max_steps = np.max(self.steps_per_episode)
            min_steps = np.min(self.steps_per_episode)
            std_steps = np.std(self.steps_per_episode)
        else:
            total_episodes = 0
            average_steps = 0.0
            max_steps = 0
            min_steps = 0
            std_steps = 0.0
        self.final_metrics = {
            "total_episodes": int(total_episodes),
            "average_steps_per_episode": float(average_steps),
            "max_steps": int(max_steps),
            "min_steps": int(min_steps),
            "std_steps": float(std_steps)
        }

    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        self.steps += 1
        self.current_frame = current_frame
        self.passed_frames = recent_frames
        if self.previous_frame is None:
            self.previous_frame = current_frame
            self.frame_changed = True
        else:
            frame_changed = False
            comparison_frame = self.previous_frame
            if recent_frames is None:
                recent_frames = np.array([current_frame])
            for frame in recent_frames:
                if np.abs(frame - comparison_frame).mean() > EPSILON:
                    frame_changed = True
                else:
                    frame_changed = False
                    comparison_frame = frame
                if frame_changed:
                    break
            self.frame_changed = frame_changed
        self.previous_frame = current_frame    

    def report(self) -> dict:
        """
        Provides the following metrics:
        - `steps`: Number of steps taken in the episode.
        - `frame_changed`: Whether the frame has changed since the last step.
        - `current_frame`: The current frame.
        - `passed_frames`: All frames that have passed since the last step.
        Returns:
            dict: A dictionary containing the current metrics.
        """
        return {
            "steps": self.steps,
            "frame_changed": self.frame_changed,
            "current_frame": self.current_frame,
            "passed_frames": self.passed_frames
        }
    
    def report_final(self) -> dict:
        """
        Provides the following metrics:
        - `total_episodes`: Total number of episodes completed.
        - `average_steps_per_episode`: Average number of steps taken per episode.
        - `max_steps`: Maximum number of steps taken in any episode.
        - `min_steps`: Minimum number of steps taken in any episode.
        - `std_steps`: Standard deviation of steps taken across episodes.

        Returns:
            dict: A dictionary containing the final metrics.
        """
        return self.final_metrics


class OCRMetric(MetricGroup, ABC):
    """
    MetricGroup for OCR results.
    """
    NAME = "ocr"
    
    def start(self):
        """
        Assumes the child has initialized a dict called self.kinds which tracks the various kinds of OCR that could be done. 
                self.kinds should be in the form: {kind: (region_name, prompt/None)} where region_name is the name of the region to OCR for that kind and prompt is the specialized prompt used to elicit an OCR response, or None for the default.
                Will track ocr results in form of list of dictionaries where these kinds are keys. 
        """
        super().start()
        if self.NAME != "ocr":
            log_error("OCRMetric subclasses must have NAME equal to 'ocr' for the environment get_info() aggregation step to work.", self._parameters)
        if not hasattr(self, 'kinds'):
            log_error("OCRMetrics must declare self.kinds dictionary", self._parameters)
        elif not isinstance(self.kinds, dict):
            log_error("self.kinds must be a dictionary", self._parameters)
        self.kinds: dict
        for item in self.kinds:
            if not isinstance(item, str):
                log_error("self.kinds keys must be strings", self._parameters)
            region_info, prompt = self.kinds[item]
            if not isinstance(region_info, str):
                log_error("self.kinds values must be region names (strings)", self._parameters)
            if region_info not in self.state_parser.named_screen_regions:
                log_error(f"OCR region name {region_info} not found in state parser named regions. Available options: {self.state_parser.named_screen_regions}", self._parameters)
            if prompt is not None and not isinstance(prompt, str):
                log_error("self.kinds values must be region names (strings) and prompts (strings or None)", self._parameters)


    def read_kind(self, current_frame: np.ndarray, kind: str) -> Optional[str]:
        """
        Reads text of the given kind from the current frame.
        """
        if kind not in self.kinds:
            log_error(f"OCR kind {kind} not found in self.kinds", self._parameters)
        region = self.kinds[kind]
        captured_region = self.state_parser.capture_named_region(current_frame=current_frame, name=region)
        ocr_result = ocr([captured_region])[0]
        if ocr_result.strip() != "":
            return ocr_result.strip()
        return None
    
    def batch_read_kind(self, frames: np.ndarray, kinds: str, merge: bool = True) -> Optional[List[str]]:
        """
        Reads text from the frames in batch style and does optional merge aggregatoin over results. 
        If merge is True, the list size can be anything less than the number of kinds
        else it will be equal to the number of kinds, with Nones included. 
        """
        if kinds not in self.kinds:
            log_error(f"OCR kind {kinds} not found in self.kinds", self._parameters)
        region, prompt = self.kinds[kinds]
        captured_regions = []
        for frame in frames:
            captured_region = self.state_parser.capture_named_region(current_frame=frame, name=region)
            captured_regions.append(captured_region)
        ocr_results = ocr(captured_regions, text_prompt=prompt, do_merge=merge)
        if merge:
            ocr_results = [res.strip() for res in ocr_results if res.strip() != ""]
            if len(ocr_results) == 0:
                return None
            return ocr_results
        else:
            return ocr_results

    @staticmethod
    def can_read_kind(self, frame: np.ndarray, kind: str) -> bool:
        """
        Checks if the frame has text for the given kind.

        Args:
            frame (np.ndarray): The frame to check.
            kind (str): The kind of text to check for.
        """
        raise NotImplementedError

    def reset(self, first = False):
        """
        ocr_texts will track a list of the form List[Tuple[int, Dict[str, str]]]
        which is a list of (step_number, {kind: ocr_text}) dictionaries.
        """
        self.ocr_texts = []
        self.steps = 0
        self.prev_has_ocr = False

    def simple_merge(self, existing: Optional[str], new: str) -> str:
        if existing is None:
            return new
        if new in existing:
            return existing
        if existing in new:
            return new
        return existing + " " + new
    
    
    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        """
        By default we don't use the batch functionality, but this can be overridden in child classes.
        """
        all_frames = None
        if recent_frames is not None:
            all_frames = recent_frames # Current frame is included in recent frames
        else:
            all_frames = np.array([current_frame])
        # remove all frames that are identical to the previous frame to reduce OCR calls
        true_all_frames = []
        previous_frame = None
        for frame in all_frames:
            if previous_frame is None:
                true_all_frames.append(frame)
                previous_frame = frame
            else:
                if np.abs(frame - previous_frame).mean() > EPSILON:
                    true_all_frames.append(frame)
                    previous_frame = frame
        ocr_dict = {}
        # Aggregate results for all frames and separate per kind. 
        for kind in self.kinds.keys():
            all_frames = []
            for frame in true_all_frames:
                if self.can_read_kind(frame, kind):
                    all_frames.append(frame)
            if len(all_frames) > 0:
                all_frames = np.array(all_frames)
                ocr_result = self.batch_read_kind(all_frames, kind, merge=True)
                if ocr_result is not None:
                    ocr_dict[kind] = ocr_result
        if len(ocr_dict) > 0:
            self.ocr_texts.append((self.steps, ocr_dict))
            self.prev_has_ocr = True
        else:
            self.prev_has_ocr = False
        self.steps += 1    


    def report(self) -> dict:
        """
        Reports just the previous OCR texts extracted in the episode.
        Returns:
            dict: A dictionary containing the OCR texts.
        """
        if self.prev_has_ocr:
            return {
                "ocr_texts": self.ocr_texts[-1][1],
                "step": self.ocr_texts[-1][0]
            }
        else:
            return {}
    
    def report_final(self) -> dict:
        """
        Reports all the OCR texts extracted in the episode.
        """
        return {
            "ocr_texts": self.ocr_texts
        }
    
    def close(self):
        pass


class StateTracker():
    """
Tracks and provides API access to the game state / metrics over time and across episodes.
The most hassle-free way to read from the StateTracker is to use the `report()` and `report_final()` methods to get nested dictionaries of all metrics tracked.

**Example Usage:**

```python
import numpy as np
from poke_worlds import get_pokemon_emulator
emulator = get_pokemon_emulator(variant="pokemon_red")

# We can access the StateTracker via the emulator
state_tracker = emulator.state_tracker

# Run a random action on the emulator
emulator.reset()
allowed_actions = list(LowLevelActions)
action = np.random.choice(allowed_actions)
_, _ = emulator.step(action) # also updates the StateTracker internally
# We can access the current episode metrics via the StateTracker
episode_metrics = state_tracker.report() # access all of them as a nested dict
specific_metric = state_tracker.get_episode_metric(("core", "steps")) # access specific metrics

# If we reset the emulator, the StateTracker will reset its inter-episode metrics as well
emulator.reset()
action = np.random.choice(allowed_actions)
_, _ = emulator.step(action)
emulator.close() # StateTracker will finalize its metrics internally
final_metrics = state_tracker.report_final() # access all of them as a nested dict
specific_final_metric = state_tracker.get_final_metric(("core", "average_steps_per_episode")) # access specific final metrics
```
    """
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
        self.start()
        if self.metric_classes[0] != CoreMetrics:
            log_error("First metric class must be CoreMetrics. Make sure to call `super().start()` first in child class overrides of `start()`.", parameters)
        self.metrics = {}
        """ Dictionary to store MetricGroup instances. """
        for metric_group_class in self.metric_classes:
            metric_group_instance: MetricGroup = metric_group_class(state_parser, parameters)
            self.metrics[metric_group_instance.NAME] = metric_group_instance
        self.episode_metrics: Dict[str, Dict[str, Any]] = None
        """ Dictionary to store metrics running during episode. """
        self.final_metrics: Dict[str, Dict[str, Any]] = None

    def start(self):
        """
        Sets up the metrics for the tracker by creating the list `self.metric_classes`

        Child classes must FIRST call super().start() and THEN set up their own metric classes.
        """
        self.metric_classes: List[Type[MetricGroup]] = [CoreMetrics]

    def reset(self):
        """
        Is called once per environment reset to reset any tracked metrics.
        """
        for metric_group in self.metrics.values():
            metric_group.reset()
        self.step()

    def step(self, recent_frames: Optional[np.ndarray] = None):
        """
        Is called once per environment step to update any tracked metrics.

        Args:
            recent_frames (Optional[np.ndarray]): The stack of frames that were rendered during the last action. Shape is [n_frames, height, width, channels]. Can be None if rendering is disabled.
            epsilon (float, optional): The threshold for considering a frame change.
        """
        current_frame = None
        if recent_frames is None:
            current_frame = self.state_parser.get_current_frame()
        else:
            current_frame = recent_frames[-1]
        self.episode_metrics = {}
        for metric_group in self.metrics.values():
            metric_group.step(current_frame, recent_frames)
            self.episode_metrics[metric_group.NAME] = metric_group.report()

    def close(self):
        """
        Is called once when the environment is closed to finalize any tracked metrics.
        """
        for metric_group in self.metrics.values():
            metric_group.close()
        self.final_metrics = {name: mg.report_final() for name, mg in self.metrics.items()}

    def report(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the current episode metrics.

        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary containing the current episode metrics.
        """
        return self.episode_metrics
    
    def report_final(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the final metrics after environment close.

        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary containing the final metrics.
        """
        if self.final_metrics is None:
            return {}
        return self.final_metrics

    def _get_specific_metric(self, metrics_dict, key: Tuple[str, str]) -> Dict[str, Any]:
        if metrics_dict is None:
            log_error("No metrics available. Have you called step() or close()?")
        metric_group_name, metric_name = key
        if metric_group_name not in metrics_dict:
            log_error(f"Metric group {metric_group_name} not found in metrics. Available groups: {list(metrics_dict.keys())}")
        if metric_name not in metrics_dict[metric_group_name]:
            log_error(f"Metric {metric_name} not found in metric group {metric_group_name}. Available metrics: {list(metrics_dict[metric_group_name].keys())}")
        return metrics_dict[metric_group_name][metric_name]
    
    def get_episode_metric(self, key: Tuple[str, str]) -> Dict[str, Any]:
        """
        Returns the metrics for a specific episode and metric group.

        Does not give final metrics at any point. 

        Args:
            key (Tuple[str, str]): A tuple containing the episode ID and metric group name.

        Returns:
            Dict[str, Any]: A dictionary containing the metrics for the specified episode and metric group.
        """
        return self._get_specific_metric(self.episode_metrics, key)
    
    def get_final_metric(self, key: Tuple[str, str]) -> Dict[str, Any]:
        """
        Returns the final metrics for a specific metric group.

        Args:
            key (Tuple[str, str]): A tuple containing the metric group name and metric name.


        Returns:
            Dict[str, Any]: A dictionary containing the final metrics for the specified metric group.
        """
        return self._get_specific_metric(self.final_metrics, key)

    def __repr__(self) -> str:
        metric_names = [mg.NAME for mg in self.metrics.values()]
        return f"<StateTracker(name={self.name}, session_name={self.session_name}, instance_id={self.instance_id}), metrics=({', '.join(metric_names)})>"