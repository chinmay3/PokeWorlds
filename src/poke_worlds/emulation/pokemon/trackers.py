from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import MetricGroup, StateTracker
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser, AgentState, PokemonRedStateParser
from typing import Optional, Type
import numpy as np


class CorePokemonMetrics(MetricGroup):
    """
    Pokémon-specific metrics.
    """
    NAME = "pokemon_core"
    REQUIRED_PARSER = PokemonStateParser
    BATTLE_EXIT_STABLE_COOLDOWN = 3 
    """ Number of frames to wait after exiting battle to confirm exit. 
        Overall this mechanism absolutely does not work, but somehow sort of does?
        It does not actually catch when you exit battle, but it triggers in the very start of a battle most of the time.
        This means on aggregate, it still counts battles reasonably. 
        But it should not be used for anything requiring precise per-episode battle counts.    
    """

    def start(self):
        self.n_battles_total = []
        super().start()

    def reset(self, first = False):
        if not first:
            self.n_battles_total.append(self.n_battles_completed)
        self.current_state: AgentState = AgentState.IN_DIALOGUE # Start by default in dialogue because it has the least permissable actions. 
        """ The current state of the agent in the game. """
        self._previous_state = self.current_state
        self.n_battles_completed = 0
        """ Number of battles completed in the current episode. Does not count the number of battles started. """
        self.surely_exited_battle_counter = self.BATTLE_EXIT_STABLE_COOLDOWN

    def close(self):
        self.reset()
        return

    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        self._previous_state = self.current_state
        current_state = self.state_parser.get_agent_state(current_frame)
        self.current_state = current_state
        if self.surely_exited_battle_counter < self.BATTLE_EXIT_STABLE_COOLDOWN:
            if self.current_state == AgentState.IN_BATTLE:
                # false alarm, still in battle
                self.surely_exited_battle_counter = self.BATTLE_EXIT_STABLE_COOLDOWN
            else:
                self.surely_exited_battle_counter -= 1
                if self.surely_exited_battle_counter <= 0:
                    # confirmed exited battle
                    self.n_battles_completed += 1
                    self.surely_exited_battle_counter = self.BATTLE_EXIT_STABLE_COOLDOWN            
        if self.current_state != self._previous_state:
            if self._previous_state == AgentState.IN_BATTLE:
                self.surely_exited_battle_counter -= 1
            #self.log_report()
        
    def report(self) -> dict:
        """
        Reports the current Pokémon core metrics:
        - Agent state
        - Number of battles completed in the current episode
        Returns:
            dict: A dictionary containing the current agent state.
        """
        return {
            "agent_state": self.current_state.name,
            "n_battles_completed": self.n_battles_completed
        }
    
    def report_final(self) -> dict:
        """
        Reports:
        - Total number of battles completed across all episodes
        - Mean battles per episode
        - Min/max battles per episode
        - Standard deviation of battles per episode
        """
        n_battles_total = np.array(self.n_battles_total)
        if len(n_battles_total) == 0:
            return {
                "total_battles_completed": 0,
                "mean_battles_per_episode": 0.0,
                "min_battles_per_episode": 0,
                "max_battles_per_episode": 0,
                "std_battles_per_episode": 0.0
            }
        return {
            "total_battles_completed": int(np.sum(n_battles_total)),
            "mean_battles_per_episode": float(np.mean(n_battles_total)),
            "min_battles_per_episode": int(np.min(n_battles_total)),
            "max_battles_per_episode": int(np.max(n_battles_total)),
            "std_battles_per_episode": float(np.std(n_battles_total))
        }


class PokemonRedStarter(MetricGroup):
    """
    Have some more specific metrics for Pokemon Red.
    """
    NAME = "PokemonRedStarter"
    REQUIRED_PARSER = PokemonRedStateParser
    def start(self):
        self.starters_chosen = []
        super().start()

    def reset(self, first = False):
        if not first:
            self.starters_chosen.append(self.current_starter)
        self.current_starter = None
        """ The starter Pokémon chosen in the current episode. """

    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        if self.current_starter is not None:
            return
        if recent_frames is not None:
            all_frames = recent_frames
        else:
            all_frames = np.array([current_frame])
        for frame in all_frames:
            chose_charmander = self.state_parser.named_region_matches_multi_target(frame, "dialogue_box_middle", "picked_charmander")
            if chose_charmander:
                self.current_starter = "charmander"
                return
            chose_bulbasaur = self.state_parser.named_region_matches_multi_target(frame, "dialogue_box_middle", "picked_bulbasaur")
            if chose_bulbasaur:
                self.current_starter = "bulbasaur"
                return
            chose_squirtle = self.state_parser.named_region_matches_multi_target(frame, "dialogue_box_middle", "picked_squirtle")
            if chose_squirtle:
                self.current_starter = "squirtle"
                return
        return
    
    def report(self) -> dict:
        """
        Reports the current starter Pokémon chosen in the episode.

        Returns:
            dict: A dictionary containing the current starter Pokémon.
        """
        return {
            "current_starter": self.current_starter
        }
    
    def close(self):
        self.reset()
        starter_choices = {
            "charmander": 0,
            "bulbasaur": 0,
            "squirtle": 0,
            None: 0
        }
        for choice in self.starters_chosen:
            starter_choices[choice] += 1
        self.starter_choices = starter_choices

    def report_final(self):
        """
        Reports the total number of times each starter Pokémon was chosen across all episodes.
        """
        return self.starter_choices
            

    

class CorePokemonTracker(StateTracker):
    """
    StateTracker for core Pokémon metrics.
    """
    def start(self):
        super().start()
        self.metric_classes.extend([CorePokemonMetrics])


class PokemonRedStarterTracker(CorePokemonTracker):
    """
    Example StateTracker that tracks the starter Pokémon chosen in Pokémon Red.
    """
    def start(self):
        super().start()
        self.metric_classes.extend([PokemonRedStarter])