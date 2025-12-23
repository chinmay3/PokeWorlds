from poke_worlds.utils import log_error
from poke_worlds.interface.action import HighLevelAction
from poke_worlds.emulation.pokemon.parsers import AgentState
from poke_worlds.emulation.pokemon.trackers import CorePokemonTracker
from poke_worlds.emulation import LowLevelActions
from abc import ABC
from typing import List, Tuple
import numpy as np

from gymnasium.spaces import Box
import matplotlib.pyplot as plt

HARD_MAX_STEPS = 20
""" The hard maximum number of steps we'll let agents take in a sequence """

def frame_changed(past: np.ndarray, preset: np.ndarray, epsilon=0.01):
    return np.abs(past -preset).mean() > epsilon 

def _plot(past: np.ndarray, current: np.ndarray):
    # plot both side by side for debug
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(past)
    axs[1].imshow(current)
    plt.show()


class BaseMovementAction(HighLevelAction, ABC):
    REQUIRED_STATE_TRACKER = CorePokemonTracker

    def move(self, direction: str, steps: int) -> Tuple[np.ndarray, int]:
        """
        TODO: Docstring
        """
        action_dict = {"right": LowLevelActions.PRESS_ARROW_RIGHT, 
                       "down": LowLevelActions.PRESS_ARROW_DOWN, 
                       "up": LowLevelActions.PRESS_ARROW_UP,
                       "left": LowLevelActions.PRESS_ARROW_LEFT}
        if direction not in action_dict.keys():
            log_error(f"Got invalid direction to move {direction}", self._parameters)
        action = action_dict[direction]
        # keep trying the action.
        # exit status 0 -> finished steps
        # 1 -> took some steps, but not all, and then frame stopped changing OR the frame starts oscillating (trying to check for jitter)
        # 2 -> took some steps, but agent state changed from free roam
        # -1 -> frame didn't change, even on the first step 
        action_success = -1
        transition_frames = []
        previous_frame = self._emulator.get_current_frame() # Do NOT get the state tracker frame, as it may have a grid on it. 
        n_step = 0
        agent_state = AgentState.FREE_ROAM
        while n_step < steps and agent_state == AgentState.FREE_ROAM:
            frames = self._emulator.run_action_on_emulator(action)
            transition_frames.extend(frames)
            # check if frames changed. If not, break out
            current_frame = frames[-1]
            if not frame_changed(previous_frame, current_frame):
                break
            agent_state = self._emulator.state_parser.get_agent_state(self._emulator.get_current_frame())
            if agent_state != AgentState.FREE_ROAM:
                break
            n_step += 1
            previous_frame = current_frame

        transition_frames = np.stack(transition_frames, axis=0)
        self._emulator.update_listeners_after_actions(transition_frames)
        if agent_state != AgentState.FREE_ROAM:
            action_success = 2
        else:
            if n_step <= 0:
                action_success = -1
            elif n_step == steps:
                action_success = 0
            else:
                action_success = 1
        return transition_frames, action_success 
    
    def chain_move(self, direction_steps: List[Tuple[str, int]]):
        """
        Chain together several move operations. Exit at the first action_success != 0
        # TODO: DOcstring and implement
        """
        raise ValueError

    def move_until_stop(self, direction, ind_step=5):
        """
        Move in a direction until action_success is no longer 0. 
        MAJOR VULNERABILITY: What if frame change detection is poor?
            You should have a hard max to guard against this. 
        """
        raise ValueError


    def is_valid(self, **kwargs):
        """
        Just checks if the 
        # TODO: Docstring        
        """
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM


class MoveSteps(BaseMovementAction):

    def get_action_space(self):
        # modelling this as a Box in 2D
        return Box(low=-HARD_MAX_STEPS//2, high=HARD_MAX_STEPS//2, shape=(2,), dtype=np.int8)

    def space_to_parameters(self, space_action):
        direction = None
        if space_action[0] > 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)
            direction = "up"
        if space_action[0] < 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)            
            direction = "down"
        if space_action[1] > 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)
            direction = "right"
        if space_action[1] < 0:
            if direction is not None:
                log_error(f"Weird space vector: {space_action}", self._parameters)
            direction = "left"
        if direction is None:
            log_error(f"Weird space vector: {space_action}", self._parameters)
        steps = abs(int(space_action.sum()))
        return {"direction": direction, "steps": steps}
        

    def parameters_to_space(self, direction: str, steps: int):
        move_vec = np.zeros(2) # x, y
        if direction == "up":
            move_vec[1] = 1
        elif direction == "right":
            move_vec[0] = 1
        elif direction == "down":
            move_vec[1] = -1
        elif direction == "left":
            move_vec[0] = -1
        else:
            log_error(f"Unrecognized direction {direction}", self._parameters)
        if steps <= 0:
            log_error(f"Bro. What you trying here. Don't step weirdly: {steps}", self._parameters)
        move_vec *= steps
        return move_vec


    def _execute(self, direction, steps):
        res = self.move(direction=direction, steps=steps)
        state_report = self._state_tracker.report()
        return [state_report], True



class MenuAction:
    pass # navigate up and down, hit b to exit and a to select. 


class BattleActions:
    pass