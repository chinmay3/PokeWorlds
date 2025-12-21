from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import MetricGroup, StateTracker
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser, AgentState, MemoryBasedPokemonRedStateParser, PokemonRedStateParser
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
        But battles_completed or n_battles_total should not be used for anything requiring precise per-episode battle counts.    
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
            "agent_state": self.current_state,
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
    NAME = "pokemon_red_starter"
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
        starter_choices["None"] = starter_choices.pop(None)
        self.starter_choices = starter_choices

    def report_final(self):
        """
        Reports the total number of times each starter Pokémon was chosen across all episodes.
        """
        return self.starter_choices
            

class PokemonRedLocation(MetricGroup):
    """
    Reads from memory states to determine the player's current location in Pokemon Red.
    """
    NAME = "pokemon_red_location"
    REQUIRED_PARSER = MemoryBasedPokemonRedStateParser

    def start(self):
        self.total_n_walk_steps = []
        self.total_n_of_unique_locations = []
        super().start()

    def reset(self, first = False):
        if not first:
            self.total_n_of_unique_locations.append(len(self.unique_locations))
            self.total_n_walk_steps.append(self.n_walk_steps)
        else:
            self.direction = None
            self.current_local_location = None
            self.current_global_location = None
            self.has_moved = False
            self.n_walk_steps = 0
            self.unique_locations = set()


    def step(self, current_frame: np.ndarray, recent_frames: Optional[np.ndarray]):
        self.state_parser: MemoryBasedPokemonRedStateParser
        self.direction = self.state_parser.get_facing_direction()
        current_local_position = self.state_parser.get_local_coords()
        current_global_position = self.state_parser.get_global_coords()
        x, y, map_number = current_local_position
        map_name = self.state_parser.get_map_name(map_number)
        if self.current_local_location is None:
            self.current_local_location = (x, y, map_name)
            self.current_global_location = current_global_position
            self.unique_locations.add(map_name)
        else:
            if self.current_local_location[0] != x or self.current_local_location[1] != y or self.current_local_location[2] != map_name:
                self.has_moved = True
                if map_name == self.current_local_location[2]:
                    initial_coord = np.array(self.current_local_location[0:2])
                    new_coord = np.array(current_local_position[0:2])
                    manhattan_distance = np.sum(np.abs(initial_coord - new_coord))
                    self.n_walk_steps += manhattan_distance
                else:
                    # use global coords to estimate distance moved
                    initial_coord = np.array(self.current_global_location)
                    new_coord = np.array(current_global_position)
                    manhattan_distance = np.sum(np.abs(initial_coord - new_coord))
                    self.n_walk_steps += manhattan_distance
                    self.unique_locations.add(map_name)
            else:
                self.has_moved = False
            self.current_global_location = current_global_position
            self.current_local_location = (x, y, map_number)

    def report(self) -> dict:
        """
        Reports the current location metrics:
        - direction: The direction the player is facing
        - has_moved: Whether the player has moved since the last step
        - current_global_location: (x, y)
        - current_local_location: (x, y, map_name)
        - n_walk_steps: Number of walk steps taken in the current episode
        - unique_locations: List of unique locations visited in the current episode
        - n_of_unique_locations: Number of unique locations visited in the current episode

        Returns:
            dict: A dictionary containing the current location metrics.
        """
        return {
            "direction": self.direction,
            "has_moved": self.has_moved,
            "current_global_location": self.current_global_location,
            "current_local_location": self.current_local_location,
            "n_walk_steps": self.n_walk_steps,
            "unique_locations": list(self.unique_locations),
            "n_of_unique_locations": len(self.unique_locations)
        }
    
    def report_final(self):
        return {
            "mean_n_walk_steps_per_episode": float(np.mean(self.total_n_walk_steps)),
            "mean_n_unique_locations_per_episode": float(np.mean(self.total_n_of_unique_locations)),
            "std_n_walk_steps_per_episode": float(np.std(self.total_n_walk_steps)),
            "std_n_unique_locations_per_episode": float(np.std(self.total_n_of_unique_locations)),
            "max_n_walk_steps_per_episode": int(np.max(self.total_n_walk_steps)),
            "max_n_unique_locations_per_episode": int(np.max(self.total_n_of_unique_locations)),
        }
    
    def close(self):
        pass
    

class CorePokemonTracker(StateTracker):
    """
    StateTracker for core Pokémon metrics.
    """
    def start(self):
        super().start()
        self.metric_classes.extend([CorePokemonMetrics])

    def step(self, *args, **kwargs):
        """
        Calls on super().step(), but then modifies the 
        """
        super().step(*args, **kwargs)
        state = self.episode_metrics["pokemon_core"]["agent_state"]
        # if agent_state is in FREE ROAM, draw the grid, otherwise do not
        if state == AgentState.FREE_ROAM:
            screen = self.episode_metrics["core"]["current_frame"]
            screen = self.state_parser.draw_grid_overlay(current_frame=screen, grid_skip=20)
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