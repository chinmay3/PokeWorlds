from poke_worlds.emulation.pokemon.base_metrics import (
    CorePokemonMetrics,
    PokemonOCRMetric,
    PokemonRedLocation,
    PokemonRedStarter,
)
from poke_worlds.emulation.pokemon.test_metrics import (
    PokemonCenterTerminateMetric,
    MtMoonTerminateMetric,
    SpeakToBillCompleteTerminateMetric,
    PickupPokeballTerminateMetric,
    ReadTrainersTipsSignTerminateMetric,
    SpeakToCinnabarGymAideCompleteTerminateMetric,
    DefeatedBrockTerminateMetric,
    DefeatedLassTerminateMetric,
    CaughtPidgeyTerminateMetric,
    CaughtPikachuTerminateMetric,
    BoughtPotionAtPewterPokemartTerminateMetric,
    UsedPotionOnCharmanderTerminateMetric,
    OpenMapTerminateMetric,
)

from poke_worlds.emulation.pokemon.base_metrics import (
    PokemonTestMetric,
)
from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import (
    StateTracker,
    TestTrackerMixin,
)
from poke_worlds.emulation.pokemon.parsers import (
    AgentState,
)
from typing import Optional


class CorePokemonTracker(StateTracker):
    """
    StateTracker for core Pokémon metrics.
    """

    def start(self):
        super().start()
        self.metric_classes.extend([CorePokemonMetrics, PokemonTestMetric])

    def step(self, *args, **kwargs):
        """
        Calls on super().step(), but then modifies the current frame to overlay the grid if the agent is in FREE ROAM.
        """
        super().step(*args, **kwargs)
        state = self.episode_metrics["pokemon_core"]["agent_state"]
        # if agent_state is in FREE ROAM, draw the grid, otherwise do not
        if state == AgentState.FREE_ROAM:
            screen = self.episode_metrics["core"]["current_frame"]
            screen = self.state_parser.draw_grid_overlay(current_frame=screen)
            self.episode_metrics["core"]["current_frame"] = screen
            previous_screens = self.episode_metrics["core"]["passed_frames"]
            if previous_screens is not None:
                self.episode_metrics["core"]["passed_frames"][-1, :] = screen


class PokemonRedStarterTracker(CorePokemonTracker):
    """
    Example StateTracker that tracks the starter Pokémon chosen in Pokémon Red.
    """

    def start(self):
        super().start()
        self.metric_classes.extend([PokemonRedStarter, PokemonRedLocation])


class PokemonOCRTracker(CorePokemonTracker):
    def start(self):
        super().start()
        self.metric_classes.extend([PokemonOCRMetric])


class PokemonTestTracker(TestTrackerMixin, PokemonOCRTracker):
    """
    Inherit this class and set TERMINATION_TRUNCATION_METRIC to create a TestTracker for Pokémon games.
    """

    TERMINATION_TRUNCATION_METRIC = PokemonCenterTerminateMetric


class PokemonRedCenterTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent reaches the entrance to the Viridian City Pokémon Center.
    """

    TERMINATION_TRUNCATION_METRIC = PokemonCenterTerminateMetric


class PokemonRedMtMoonTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent reaches the entrance to Mt. Moon.
    """

    TERMINATION_TRUNCATION_METRIC = MtMoonTerminateMetric


class PokemonRedSpeakToBillTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent speaks to Bill.
    """

    TERMINATION_TRUNCATION_METRIC = SpeakToBillCompleteTerminateMetric


class PokemonRedPickupPokeballTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent picks up the Pokéball in Professor Oak's Lab.
    """

    TERMINATION_TRUNCATION_METRIC = PickupPokeballTerminateMetric


class PokemonRedReadTrainersTipsSignTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent reads the Trainer's Tips sign.
    """

    TERMINATION_TRUNCATION_METRIC = ReadTrainersTipsSignTerminateMetric


class PokemonRedSpeakToCinnabarGymAideCompleteTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent speaks to the Cinnabar Gym aide.
    """

    TERMINATION_TRUNCATION_METRIC = SpeakToCinnabarGymAideCompleteTerminateMetric


class PokemonRedDefeatedBrockTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent defeats Brock.
    """

    TERMINATION_TRUNCATION_METRIC = DefeatedBrockTerminateMetric


class PokemonRedDefeatedLassTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent defeats the Lass trainer.
    """

    TERMINATION_TRUNCATION_METRIC = DefeatedLassTerminateMetric


class PokemonRedCaughtPidgeyTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent catches a Pidgey.
    """

    TERMINATION_TRUNCATION_METRIC = CaughtPidgeyTerminateMetric


class PokemonRedCaughtPikachuTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent catches a Pikachu.
    """

    TERMINATION_TRUNCATION_METRIC = CaughtPikachuTerminateMetric


class PokemonRedBoughtPotionAtPewterPokemartTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent buys a Potion at the Pewter City Poké Mart.
    """

    TERMINATION_TRUNCATION_METRIC = BoughtPotionAtPewterPokemartTerminateMetric


class PokemonRedUsedPotionOnCharmanderTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent uses a Potion on Charmander.
    """

    TERMINATION_TRUNCATION_METRIC = UsedPotionOnCharmanderTerminateMetric


class PokemonRedOpenMapTestTracker(PokemonTestTracker):
    """
    A TestTracker for Pokémon Red that ends an episode when the agent opens the map.
    """

    TERMINATION_TRUNCATION_METRIC = OpenMapTerminateMetric
