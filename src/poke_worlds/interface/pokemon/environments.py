from typing import Optional, Dict, Any, List, Tuple

from gymnasium import spaces

from poke_worlds.emulation.pokemon.base_metrics import CorePokemonMetrics
from poke_worlds.utils import load_parameters, log_dict, log_info
from poke_worlds.emulation.pokemon.emulators import PokemonEmulator
from poke_worlds.emulation.pokemon.trackers import (
    PokemonRedStarterTracker,
    PokemonOCRTracker,
)
from poke_worlds.interface.environment import (
    DummyEnvironment,
    Environment,
    TestEnvironmentMixin,
)
from poke_worlds.interface.controller import Controller

import gymnasium as gym
import numpy as np


class PokemonEnvironment(DummyEnvironment):
    """
    A basic Pokemon Environment.
    """

    REQUIRED_EMULATOR = PokemonEmulator
    REQUIRED_STATE_TRACKER = CorePokemonMetrics


class PokemonOCREnvironment(PokemonEnvironment):
    """
    A Pokemon Environment that includes OCR region captures and agent state.
    """

    REQUIRED_STATE_TRACKER = PokemonOCRTracker
    REQUIRED_EMULATOR = PokemonEmulator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        screen_shape = self._emulator.screen_shape
        screen_space = spaces.Box(
            low=0, high=255, shape=(screen_shape[1], screen_shape[0], 1), dtype=np.uint8
        )

        self.observation_space = screen_space
        """ The observation space is the raw pixel values of the emulator's screen only. """

    @staticmethod
    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        Environment.override_state_tracker_class(
            emulator_kwargs, PokemonOCREnvironment.REQUIRED_STATE_TRACKER
        )
        return emulator_kwargs

    def get_observation(
        self,
        *,
        action=None,
        action_kwargs=None,
        transition_states=None,
        action_success=None,
    ):
        if transition_states is None:
            current_state = self.get_info()
            screen = current_state["core"]["current_frame"]
        else:
            screen = transition_states[-1]["core"]["current_frame"]
        current_state = self._emulator.state_parser.get_agent_state(screen)
        return screen


class PokemonTestEnvironment(TestEnvironmentMixin, PokemonOCREnvironment):
    pass


class PokemonRedStarterChoiceEnvironment(PokemonOCREnvironment):
    """
    An environment which starts at the starter selection point in PokemonRed and terminates when the player selects a starter Pokemon.
    """

    REQUIRED_TRACKER = PokemonRedStarterTracker

    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        """
        Override default emulator keyword arguments for this environment.
        """
        Environment.override_state_tracker_class(
            emulator_kwargs, PokemonRedStarterTracker
        )
        emulator_kwargs["init_state"] = "starter"
        return emulator_kwargs

    def determine_terminated(
        self, start_state, *, action=None, action_kwargs=None, transition_states=None, action_success=None
    ) -> bool:
        super_terminated = super().determine_terminated(
            start_state=start_state,
            action=action,
            action_kwargs=action_kwargs,
            transition_states=transition_states,
            action_success=action_success,
        )
        if transition_states is None:
            return super_terminated
        states = transition_states
        for state in states:
            starter_chosen = state["pokemon_red_starter"]["current_starter"]
            if starter_chosen is not None:
                return True
        return super_terminated


class PokemonRedChooseCharmanderEnvironment(PokemonRedStarterChoiceEnvironment):
    """
    Reward the agent for choosing Charmander as quickly as possible.
    """

    def determine_reward(
        self, start_state, *, action=None, action_kwargs=None, transition_states=None, action_success=None
    ) -> float:
        """
        Reward the agent for choosing Charmander as quickly as possible.
        """
        if transition_states is None:
            return 0.0
        current_state = transition_states[-1]
        starter_chosen = current_state["pokemon_red_starter"]["current_starter"]
        n_steps = current_state["core"]["steps"]
        if starter_chosen is None:
            if n_steps >= self._emulator.max_steps - 2:  # some safety
                return -5.0  # Penalty for not choosing a starter within max steps
            else:
                return 0.0
        step_bonus = 100 / (n_steps + 1)
        if starter_chosen == "charmander":
            return 500.0 + step_bonus
        else:
            return (
                100.0 + step_bonus
            )  # Penalty for choosing the wrong starter. For now, just less reward.


class PokemonRedExploreStartingSceneEnvironment(PokemonRedStarterChoiceEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from poke_worlds.execution.retrieval import Index

        self._visual_index = Index(modality="image")

    def determine_reward(
        self, start_state, *, action=None, action_kwargs=None, transition_states=None, action_success=None
    ) -> float:
        """
        Reward the agent for seeing a screen it has not seen before.
        """
        # take the current frame, embed it, compare to index, and reward based on novelty
        screen = self._emulator.get_current_frame()  # avoids the grid
        similarity_scores = self._visual_index.add_compare(screen)
        if similarity_scores is None:
            return 1.0  # first frame
        novelty_score = 1.0 - float((similarity_scores.max()).item())
        return novelty_score
