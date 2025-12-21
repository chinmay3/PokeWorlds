from typing import Optional, Dict, Any, List

from poke_worlds.utils import load_parameters
from poke_worlds.emulation.pokemon import PokemonEmulator
from poke_worlds.emulation.pokemon.trackers import PokemonRedStarterTracker
from poke_worlds.interface.environment import DummyEnvironment, Environment
from poke_worlds.interface.controller import Controller

import gymnasium as gym
import numpy as np



class PokemonEnvironment(DummyEnvironment):
    REQUIRED_EMULATOR = PokemonEmulator



class PokemonRedChooseCharmanderFastEnv(Environment):
    REQUIRED_TRACKER = PokemonRedStarterTracker    

    def __init__(self, emulator: PokemonEmulator, controller: Controller, parameters: Optional[dict]=None):
        """
        Initializes the Environment with the given emulator and controller.
        
        """
        self._parameters = load_parameters(parameters)
        self._emulator = emulator
        self._controller = controller
        # observation space is a dictionary with "coords" key containing the (x, y) coordinates of the player and "facing" key containing the direction the player is facing
        coord_space = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint16)
        direction_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Dict({
            "facing": direction_space,
            "coords": coord_space
        })
        """ The observation space is the raw pixel values of the emulator's screen and the (x, y) coordinates of the player. """
        super().__init__()

    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        """
        Override default emulator keyword arguments for this environment.
        """
        emulator_kwargs["state_tracker_class"] = PokemonRedStarterTracker
        emulator_kwargs["init_state"] = "starter"
        return emulator_kwargs

    def get_observation(self) -> gym.spaces.Space:
        info = self.get_info()
        coords = info["pokemon_red_location"]["current_local_location"][:2]
        direction = info["pokemon_red_location"]["direction"]
        facing = None
        if direction == (1, 0):
            facing = 0  # Right
        elif direction == (-1, 0):
            facing = 1  # Left
        elif direction == (0, -1):
            facing = 2  # Up
        elif direction == (0, 1):
            facing = 3  # Down
        observation = {
            "facing": facing,
            "coords": np.array(coords, dtype=np.uint16)
        }
        return observation
    
    def determine_terminated(self, state):
        starter_chosen = state["pokemon_red_starter"]["current_starter"]
        return starter_chosen is not None    
    
    def determine_reward(self, start_state: Dict[str, Dict[str, Any]], action: gym.spaces.Space, transition_states: List[Dict[str, Dict[str, Any]]], action_success: bool) -> float:
        """
        Reward the agent for choosing Charmander as quickly as possible.
        """
        current_state = transition_states[-1]
        starter_chosen = current_state["pokemon_red_starter"]["current_starter"]
        n_steps = current_state["core"]["steps"]
        if starter_chosen is None:
            if n_steps >= self._emulator.max_steps-2: # some safety
                return -5.0 # Penalty for not choosing a starter within max steps
            starter_spots = [(5, 3), (6, 4), (7, 4), (8, 4), (9, 3), (8, 2), (7, 2), (6, 2)]
            player_pos = current_state["pokemon_red_location"]["current_local_location"][:2]
            dists = [np.linalg.norm(np.array(player_pos) - np.array(spot)) for spot in starter_spots]
            min_dist = min(dists)
            return -0.1 * min_dist
        step_bonus = 100 / (n_steps+1)
        if starter_chosen == "charmander":
            return 500.0 + step_bonus
        else:
            return 100.0 + step_bonus# Penalty for choosing the wrong starter. For now, just less reward.