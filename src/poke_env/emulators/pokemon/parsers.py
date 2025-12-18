"""
Pokemon specific game state parser implementations for both PokemonRed and PokemonCrystal.
While this code base started from: Borrowing heavily from https://github.com/PWhiddy/PokemonRedExperiments/ (v2) and was initially read from memory states https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py, this is no longer the case as we have moved to visual based state parsing.
This decision was primarily made to facilitate easier extension to other games and rom hacks in the future, as well as to avoid reliance on specific memory addresses which may vary between different versions of the game.

However, the code base supports reading from memory addresses to extract game state information, which can be useful for incorporating domain knowledge into reward structures or other aspects of the environment. See the MemoryBasedPokemonRedStateParser class for examples of how to read game state information from memory addresses.
"""

from poke_env.utils import log_warn, log_info, log_error, load_parameters, verify_parameters
from poke_env.emulators.emulator import StateParser, NamedScreenRegion

from typing import Set, List, Type, Dict, Optional, Tuple
import os
from abc import ABC, abstractmethod
from enum import Enum

from pyboy import PyBoy

import json
import numpy as np
from bidict import bidict


class AgentState(Enum):
    """
    Enum representing different agent states in the Pokemon game.
    1. FREE_ROAM: The agent is freely roaming the game world.
    2. IN_DIALOGUE: The agent is currently in a dialogue state. (including reading signs, talking to NPCs, etc.)
    3. IN_MENU: The agent is currently in a menu state. (including PC, Name Entry, Pokedex, etc.)
    4. IN_BATTLE: The agent is currently in a battle state.
    """
    FREE_ROAM = 0
    IN_DIALOGUE = 1
    IN_MENU = 2
    IN_BATTLE = 3

class PokemonStateParser(StateParser, ABC):
    """
    Base class for Pokemon game state parsers. Uses visual screen regions to parse game state.
    Defines common named screen regions and methods for determining game states such as being in battle, menu, or dialogue.

    Can be used to determine the exact AgentState
    """
    COMMON_REGIONS = [
                ("dialogue_bottom_right", 153, 135, 10, 10), 
                ("menu_top_right", 152, 1, 6, 6),
                ("pc_top_left", 0, 0, 6, 6),
                ("battle_enemy_hp_text", 15, 17, 10, 5), 
                ("battle_player_hp_text", 80, 73, 10, 5), 
                ("dialogue_choice_bottom_right", 153, 87, 6, 6),
                ("name_entity_top_left", 0, 32, 6, 6),
                ("player_card_middle", 56, 70, 6, 6),
                ("map_bottom_right", 140, 130, 10, 10),
    ]
    """ List of common named screen regions for Pokemon games. """

    def __init__(self, variant: str, pyboy: PyBoy, parameters: dict, additional_named_screen_region_details: List[Tuple[str, int, int, int, int]] = []):
        """
        Initializes the PokemonStateParser.
        Args:
            variant (str): The variant of the Pokemon game.
            pyboy (PyBoy): The PyBoy emulator instance.
            parameters (dict): Configuration parameters for the emulator.
            additional_named_screen_region_details (List[Tuple[str, int, int, int, int]]): Parameters associated with additional named screen regions to include.
        """
        verify_parameters(parameters)
        additional_named_screen_region_details.extend(self.COMMON_REGIONS)
        self.variant = variant
        if f"{variant}_rom_data_path" not in parameters:
            log_error(f"ROM data path not found for variant: {variant}. Add {variant}_rom_data_path to the config files. See configs/pokemon_red_vars.yaml for an example", parameters)
        self.rom_data_path = parameters[f"{variant}_rom_data_path"]
        """ Path to the ROM data directory for the specific Pokemon variant."""
        captures_dir = self.rom_data_path + "/captures/"
        named_screen_regions = []
        for region_name, x, y, w, h in additional_named_screen_region_details:
            region = NamedScreenRegion(region_name, x, y, w, h, parameters=parameters, target_path=os.path.join(captures_dir, region_name))
            named_screen_regions.append(region)
        super().__init__(pyboy, parameters, named_screen_regions)

    @abstractmethod
    def is_in_pokedex(self, current_screen: np.ndarray) -> bool:
        """
        Determines if the Pokedex is currently open.
        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.

        Returns:
            bool: True if the Pokedex is open, False otherwise.
        """
        raise NotImplementedError

    def is_in_battle(self, current_screen: np.ndarray) -> bool:
        """
        Determines if the player is currently in a battle by checking for battle HP text regions.

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.

        Returns:
            bool: True if in battle, False otherwise.
        """
        enemy_hp_match = self.named_region_matches_target(current_screen, "battle_enemy_hp_text")
        player_hp_match = self.named_region_matches_target(current_screen, "battle_player_hp_text")
        return enemy_hp_match or player_hp_match

    def is_in_menu(self, current_screen: np.ndarray, trust_previous: bool = False) -> bool:
        """
        Determines if any form of menu (or choice dialogue) is currently open by checking a variety of screen regions.

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.
            trust_previous (bool): If True, trusts that checks for other states like is_in_battle have been done and can be skipped.

        Returns:
            bool: True if the menu is open, False otherwise.
        """
        any_match_regions = ["menu_top_right", "dialogue_choice_bottom_right", "pc_top_left", 
                             "name_entity_top_left", "player_card_middle", "map_bottom_right", 
                             "pokemon_list_hp_text" # This one is defined in each subclass as the position varies slightly between games
                             ]
        if not trust_previous:
            if self.is_in_battle(current_screen):
                return False
        if self.is_in_pokedex(current_screen):
            return True
        for region_name in any_match_regions:
            if self.named_region_matches_target(current_screen, region_name):
                return True
        return False

    def is_in_dialogue(self, current_screen: np.ndarray, trust_previous: bool = False) -> bool:
        """
        Determines if the player is currently in a dialogue state or reading text from a sign, interacting with an object etc.
        Essentially anything that causes text to appear at the bottom of the screen that isn't a battle, pc or menu.

        Args: 
            current_screen (np.ndarray): The current screen frame from the emulator.
            trust_previous (bool): If True, trusts that checks for other states like is_in_battle have been done and can be skipped.

        Returns:
            bool: True if in dialogue, False otherwise.
        """
        if trust_previous:
            return self.named_region_matches_target(current_screen, "dialogue_bottom_right")
        if self.is_in_battle(current_screen):
            return False
        elif self.is_in_menu(current_screen):
            return False
        else:
            return self.named_region_matches_target(current_screen, "dialogue_bottom_right")
        
    def get_agent_state(self, current_screen: np.ndarray) -> AgentState:
        """
        Determines the current agent state based on the screen.

        Uses trust_previous to optimize checks.

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.

        Returns:
            AgentState: The current agent state.
        """
        if self.is_in_battle(current_screen):
            return AgentState.IN_BATTLE
        elif self.is_in_menu(current_screen, trust_previous=True):
            return AgentState.IN_MENU
        elif self.is_in_dialogue(current_screen, trust_previous=True):
            return AgentState.IN_DIALOGUE
        else:
            return AgentState.FREE_ROAM

class BasePokemonRedStateParser(PokemonStateParser, ABC):
    """
    Game state parser for all PokemonRed-based games.
    """
    _REGIONS = [
                ("pokedex_top_left", 7, 6, 12, 6),
                ("pokedex_info_mid_left", 6, 71, 6, 6),
                ("pokemon_list_hp_text", 32, 9, 10, 5),
                ("pokemon_stats_hp_text", 88, 24, 10, 5)
            ]

    def __init__(self, pyboy: PyBoy, variant: str, parameters: dict):
        super().__init__(variant=variant, pyboy=pyboy, parameters=parameters, additional_named_screen_region_details=self._REGIONS)

    def is_in_pokedex(self, current_screen: np.ndarray) -> bool:
        return self.named_region_matches_target(current_screen, "pokedex_top_left") or self.named_region_matches_target(current_screen, "pokedex_info_mid_left")
    
    def __repr__(self):
        return f"<PokemonRedParser(variant={self.variant})>"


class BasePokemonCrystalStateParser(PokemonStateParser, ABC):
    """
    Game state parser for all PokemonCrystal-based games.

    TODO: The map screenshot for crystal assumes a Jhoto map. Must do a similar process for Kanto. To add Kanto we should add another named screen region called map_bottom_right_kanto with same boundary as player_card_middle and then recapture it.
    Without this fix, the is_in_menu check may fail when in Kanto as the map_bottom_right region will not match.
    """
    _REGIONS = [
        ("pokemon_list_hp_text", 87, 16, 10, 5),
        ("pokedex_seen_text", 3, 88, 5, 5),
        ("pokedex_info_height_text", 69, 57, 5, 5),
        ("pokegear_top_left", 0, 0, 6, 6),
        ("pokemon_stats_lvl_text", 113, 0, 5, 5),
        ("bag_text", 18, 0, 6, 6),
    ]

    def __init__(self, pyboy: PyBoy, variant: str, parameters: dict):
        super().__init__(variant=variant, pyboy=pyboy, parameters=parameters, additional_named_screen_region_details=self._REGIONS)


    def is_in_bag(self, current_screen: np.ndarray) -> bool:
        """
        Determines if the Bag is currently open.
        """
        return self.named_region_matches_target(current_screen, "bag_text")
    
    def is_in_pokegear(self, current_screen: np.ndarray) -> bool:
        """
        Determines if the Pokegear is currently open.
        """
        return self.named_region_matches_target(current_screen, "pokegear_top_left")

    def is_in_pokedex(self, current_screen):
        return self.named_region_matches_target(current_screen, "pokedex_seen_text") or self.named_region_matches_target(current_screen, "pokedex_info_height_text")
    
    def is_in_menu(self, current_screen: np.ndarray, trust_previous: bool = False) -> bool:
        # This technically mistakenly also flags when someone calls you on the pokegear, but that's probably fine for now. 
        # Could change by adding special region for pokegear_call_top_left and overriding is_in_menu and is_in_dialogue.
        result = super().is_in_menu(current_screen, trust_previous=trust_previous)
        if result:
            return True
        if self.is_in_bag(current_screen):
            return True
        if self.is_in_pokegear(current_screen):
            return True
        # Finally, when transitioning to PC screens, maps etc, the screen goes white. Catch that here.
        #print(f"Checking for white screen... Pixel stats: {np.min(current_screen)}, {np.max(current_screen)}, {np.mean(current_screen)}") # I get 248, 248, 248.0
        # The following doesn't catch all white screens (e.g town maps), but does catch some important ones like PC screens.
        if np.mean(current_screen) > 245 and np.min(current_screen) > 245:
            return True
        elif np.mean(current_screen) > 210: # screen coming down from full white
            return True
        else:
            return False

    def __repr__(self):
        return f"<PokemonCrystalParser(variant={self.variant})>"


class PokemonRedStateParser(BasePokemonRedStateParser):
    def __init__(self, pyboy, parameters):
        super().__init__(pyboy, variant="pokemon_red", parameters=parameters)

class PokemonBrownStateParser(BasePokemonRedStateParser):
    def __init__(self, pyboy, parameters):
        super().__init__(pyboy, variant="pokemon_brown", parameters=parameters)


class PokemonCrystalStateParser(BasePokemonCrystalStateParser):
    def __init__(self, pyboy, parameters):
        super().__init__(pyboy, variant="pokemon_crystal", parameters=parameters)




"""
The below code shows how to add domain information into the game state parser and read from memory addresses to get descriptive state information. 

This is not actually used in any of the current environments, but is left here to show that if you want to bake in more domain knowledge and create explicit reward schedules etc., you can read the information required to do so in this class. 
"""
class MemoryBasedPokemonRedStateParser(BasePokemonRedStateParser):
    """
    Game state parser for Pokemon Red. Uses memory addresses to parse game state.
    Can be used to reproduce https://github.com/PWhiddy/PokemonRedExperiments/ (v2) and facilitates reward engineering based on memory states.
    """
    _PAD = 20
    _GLOBAL_MAP_SHAPE = (444 + _PAD * 2, 436 + _PAD * 2)
    _MAP_ROW_OFFSET = _PAD
    _MAP_COL_OFFSET = _PAD

    def __init__(self, pyboy, parameters):
        """
        Initializes the Pokemon Red game state parser.

        Args: 
            pyboy: An instance of the PyBoy emulator.
            parameters: A dictionary of parameters for configuration.
        """
        super().__init__(pyboy, parameters=parameters)
        events_location = parameters["pokemon_red_rom_data_path"] + "/events.json"
        with open(events_location) as f:
            event_slots = json.load(f)
        event_slots = event_slots
        event_names = {v: k for k, v in event_slots.items() if not v[0].isdigit()}
        beat_opponent_events = bidict()
        def _pop(d, keys):
            for key in keys:
                if key in d:
                    d.pop(key, None)        
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Beat "):
                beat_opponent_events[name.replace("Beat ", "")] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.defeated_opponent_events = beat_opponent_events
        """Events related to beating specific opponents. E.g. Beat Brock"""
        tms_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got Tm"):
                tms_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.tms_obtained_events = tms_obtained_events
        """Events related to obtaining specific TMs. E.g. Got Tm01"""
        hm_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got Hm"):
                hm_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.hm_obtained_events = hm_obtained_events
        """Events related to obtaining specific HMs. E.g. Got Hm01"""
        passed_badge_check_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Passed ") and "badge" in name:
                passed_badge_check_events[name.replace("Passed ", "").replace(" Check", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.passed_badge_check_events = passed_badge_check_events
        """Events related to passing badge checks. E.g. Passed Boulder badge check. These will only be relevant to enter Victory Road."""
        self.key_items_obtained_events = bidict()
        """Events related to obtaining key items. E.g. Got Bicycle"""
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got "):
                self.key_items_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.map_events = {"Cinnabar Gym": bidict(), "Victory Road": bidict(), "Silph Co": bidict(), "Seafoam Islands": bidict()}
        """Events related to specific map events like unlocking gates or moving boulders."""
        for name, slot in event_names.items():
            if name.startswith("Cinnabar Gym Gate") and name.endswith("Unlocked"):
                self.map_events["Cinnabar Gym"][name] = slot
                pop_queue.append(name)
            elif name.startswith("Victory Road") and "Boulder On" in name:
                self.map_events["Victory Road"][name] = slot
                pop_queue.append(name)
            elif name.startswith("Silph Co") and "Unlocked" in name:
                self.map_events["Silph Co"][name] = slot
                pop_queue.append(name)
            elif name.startswith("Seafoam"):
                self.map_events["Seafoam Islands"][name] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.cutscene_events = bidict()
        """ Flags for cutscene based events (I think, lol). """

        cutscenes = ["Event 001", "Daisy Walking", "Pokemon Tower Rival On Left", "Seel Fan Boast", "Pikachu Fan Boast", "Lab Handing Over Fossil Mon", "Route22 Rival Wants Battle"] # my best guess, need to verify, Silph Co Receptionist At Desk? Autowalks?
        pop_queue = []
        for name, slot in event_names.items():
            if name in cutscenes:
                self.cutscene_events[name] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.special_events = bidict(event_names)
        """ All other events not categorized elsewhere."""


        MAP_PATH = parameters["pokemon_red_rom_data_path"] + "/map_data.json"
        with open(MAP_PATH) as map_data:
            MAP_DATA = json.load(map_data)["regions"]
        self._MAP_DATA = {int(e["id"]): e for e in MAP_DATA}
        
    def local_to_global(self, r: int, c: int, map_n: int) -> tuple[int, int]:
        """
        Converts local map coordinates to global map coordinates.
        Args:
            r (int): Local row coordinate.
            c (int): Local column coordinate.
            map_n (int): Map identifier.
        Returns:
            (int, int): Global (row, column) coordinates.
        """
        try:
            (
                map_x,
                map_y,
            ) = self._MAP_DATA[map_n]["coordinates"]
            gy = r + map_y + self._MAP_ROW_OFFSET
            gx = c + map_x + self._MAP_COL_OFFSET
            if 0 <= gy < self._GLOBAL_MAP_SHAPE[0] and 0 <= gx < self._GLOBAL_MAP_SHAPE[1]:
                return gy, gx
            print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
            return self._GLOBAL_MAP_SHAPE[0] // 2, self._GLOBAL_MAP_SHAPE[1] // 2
        except KeyError:
            print(f"Map id {map_n} not found in map_data.json.")
            return self._GLOBAL_MAP_SHAPE[0] // 2, self._GLOBAL_MAP_SHAPE[1] // 2        

    def get_opponents_defeated(self) -> Set[str]:
        """
        Returns a set of all defeated opponents. This function isn't actually used in any current environments, but is left here to show how to read game state information.
        Similar functions can be created to read obtained TMs, HMs, key items, passed badge checks, etc.

        Returns:
            Set[str]: A set of names of defeated opponents.       
        """
        return self.get_raised_flags(self.defeated_opponent_events)
    
    def get_local_coords(self) -> tuple[int, int, int]:
        """
        Gets the local game coordinates (x, y, map number).
        Returns:
            (int, int, int): Tuple containing (x, y, map number).
        """
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def get_global_coords(self):
        """
        Gets the global coordinates of the player.
        Returns:
            (int, int): Tuple containing (global y, global x) coordinates.
        """
        x_pos, y_pos, map_n = self.get_local_coords()
        return self.local_to_global(y_pos, x_pos, map_n)

    def get_badges(self) -> np.array:
        """
        Gets the player's badges as a binary array.
        Returns:
            np.array: Array of 8 binary values representing whether the player has obtained each of the badges.
        """
        # or  self.bit_count(self.read_m(0xD356))
        return np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8)
