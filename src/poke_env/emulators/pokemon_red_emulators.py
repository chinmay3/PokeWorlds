"""
Pokemon Red specific emulator and game state parser implementations.
While this code base started from: Borrowing heavily from https://github.com/PWhiddy/PokemonRedExperiments/ (v2) and was initially read from memory states https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py, this is no longer the case as we have moved to visual based state parsing.
This decision was primarily made to facilitate easier extension to other games and rom hacks in the future, as well as to avoid reliance on specific memory addresses which may vary between different versions of the game.

However, the code base supports reading from memory addresses to extract game state information, which can be useful for incorporating domain knowledge into reward structures or other aspects of the environment. Examples of how to do this are provided in the PokemonRedGameStateParser class.

See the MemoryBasedPokemonRedGameStateParser class for examples of how to read game state information from memory addresses.
"""



from poke_env.utils import log_warn, log_info, log_error, load_parameters
from poke_env.emulators.emulator import Emulator, NamedScreenRegion
from poke_env.emulators.pokemon_emulator import PokemonGameStateParser
from abc import ABC
from typing import Set
import os

import json
import numpy as np
from bidict import bidict


class BasePokemonRedGameStateParser(PokemonGameStateParser, ABC):
    """
    Game state parser for all PokemonRed-based games. Uses visual screen regions to parse game state.
    """
    # dialogue_bottom_left (4, 140, 10, 10)
    _REGIONS = [("dialogue_bottom_right", 151, 135, 10, 10), 
                ("menu_top_right", 152, 1, 6, 6),
                ]

    def __init__(self, pyboy, variant, parameters):
        """
        Initializes the Pokemon Red game state parser.

        Args: 
            pyboy: An instance of the PyBoy emulator.
            parameters: A dictionary of parameters for configuration.
        """
        captures_dir = parameters[f"{variant}_rom_data_path"] + "/captures/"
        regions = []
        for region_name, x, y, w, h in self._REGIONS:
            region = NamedScreenRegion(region_name, x, y, w, h, parameters=parameters, target_path=os.path.join(captures_dir, region_name))
            regions.append(region)
        super().__init__(variant=variant, pyboy=pyboy, parameters=parameters, named_screen_regions=regions)
    
    def get_screen_top_left(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Returns the ndarray capture of the top left of the screen. The PC menu has a pokeball here when its open
        This can be used to assess state of menu or dialogue
        Args:
            current_frame (np.ndarray): The current frame from the emulator.

        Returns:
            np.ndarray: The captured top right section of the screen.
        """
        return self.capture_square_centered(current_frame, center_x=4, center_y=2, box_size=10)    
    
    def parse_all(self):
        pass

    def parse_step(self):
        #centered = self.get_screen_top_left(self.get_current_frame())
        #centered = self.draw_square_centered(self.get_current_frame(), center_x=4, center_y=2, box_size=10, thickness=2)
        #import matplotlib.pyplot as plt
        #plt.imshow(centered)
        #plt.show()
        #np.save("pc_top_left.npy", centered)
        current_frame = self.get_current_frame()

    def is_in_dialogue(self) -> bool:
        return False
    
    def is_menu_open(self) -> bool:
        return False
    
    def is_in_battle(self) -> bool:
        return False
    
    def is_pokedex_open(self) -> bool:
        return False
    


class PokemonRedGameStateParser(BasePokemonRedGameStateParser):
    def __init__(self, pyboy, parameters):
        super().__init__(pyboy, variant="pokemon_red", parameters=parameters)

    
# TODO: What is this adding here.     
class BasicPokemonRedEmulator(Emulator):
    def __init__(self, parameters: dict = None, init_state=None, headless: bool = False, max_steps: int = None, save_video: bool = None, session_name: str = None, instance_id: str = None):
        parameters = load_parameters(parameters)
        if init_state is None:
            init_state = parameters["pokemon_red_rom_data_path"] + "/states/default.state"
        gb_path = parameters["pokemon_red_rom_data_path"] + "/PokemonRed.gb"
        game_state_parser_class = PokemonRedGameStateParser
        super().__init__(gb_path, game_state_parser_class, init_state, parameters, headless, max_steps, save_video, session_name, instance_id)
    
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        return "pokemon_red"


"""
The below code shows how to add domain information into the game state parser and read from memory addresses to get descriptive state information. 

This is not actually used in any of the current environments, but is left here to show that if you want to bake in more domain knowledge and create explicit reward schedules etc., you can read the information required to do so in this class. 
"""
class MemoryBasedPokemonRedGameStateParser(BasePokemonRedGameStateParser):
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
