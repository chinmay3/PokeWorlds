from poke_env.utils import log_warn, log_info, log_error, load_parameters, verify_parameters
from poke_env.emulators.emulator import Emulator, GameStateParser, NamedScreenRegion

from typing import Set, List, Type, Dict, Optional
import os
from abc import ABC, abstractmethod
from pyboy import PyBoy

import json
import numpy as np
from bidict import bidict

VARIANTS = {
    "pokemon_red": "pokemon_red", 
    "pokemon_brown": "pokemon_red",
    "pokemon_crystal": "pokemon_crystal",
    "pokemon_fools_gold": "pokemon_crystal",
    "pokemon_prism": "pokemon_crystal",
    "pokemon_quarantine_crystal": "pokemon_crystal",
    "pokemon_starbeasts": "pokemon_crystal",
}
""" Mapping of variant names to base game types."""


class PokemonController:
    # TODO: Must implement the following
    # Autobattler (simple, first check if there is an attacking move in any pokemons slots. If so and not active, switch into it. Then spam that attack)
    # This is meant to be used with nerf_opponents to allow simulation without fear of battles getting in the way. 
    # A* pathfinding towards a given coordinate. 
    # Menu: Open Items, Open Pokemon, 
    # Open Map, Exit Map
    # Throw Ball
    # Show inventory 
    # Use Specific Item (e.g. Antidote etc) (String Based) You must establish the mapping to game id and get that done
    # Use Item on Pokemon 
    # Run from Battle
    # Show other pokemon info
    # Check Pokemon Info
    # Switch to Pokemon
    # Maybe set up an OCR on the frame to catch some amount of string mapping (i.e. catch cant use that )
        # Then you can have OCR on a sign
    # Simple mappings of interact with NPC or sign. 
    # Move in direction (for a specified number of steps)
    # Try to Buy x Items
    # 
    pass
    # TODO: See if you can find a way to draw the local coordinate system on your map. To help the VLM specify pathfinding coordinates. 
"""
Action Spaces: 
Open World, No Window open:
    Move:
        Move(direction, steps): either returns success message or failure and early exit (could be wild pokemon, could be trainer, could be obstacle). 
        Move(x, y): Try to move towards this coordinate using A* algorithm. 
    Interact: Basically press A
    Inventory:
        Search(Item Name): return [not an item or the count of item in bag]
        List items: List all items in bag
        Use(Item Name):
            For each item, perhaps have an argument input (i.e. Use Potion on [Pokemon Name])
    Pokemon:
        List: List all Pokemon
        Check(Pokemon Name)
        Check_all
        SwitchFirst(Pokemon): Switches a new pokemon to the first slot in the party
    Fly: (If there is a fly pokemon in inventory)


Battle:
    Fight:
        Attack(name): either says attack not found or uses the attack
    Inventory:
        List
        Search(Item Name)
        Use(Item Name)
    Throw Ball(Ball Kind)
    Pokemon:
        List
        Check(Pokemon Name)
        Switch(Pokemon Name)
    Run

Conversation: (Hard?)
    Continue

Menu / Options: (HARD)
    Select specific option

Manual:
    All controls (other than Enter)    
"""


class PokemonGameStateParser(GameStateParser, ABC):
    """
    Reads from memory addresses to form the state
    """
    def __init__(self, variant: str, pyboy: PyBoy, parameters: dict, named_screen_regions: Optional[list[NamedScreenRegion]] = None):
        """
        Initializes the PokemonGameStateParser.
        Args:
            variant (str): The variant of the Pokemon game (e.g., "pokemon_red", "pokemon_crystal", "pokemon_quarantine_crystal").
            pyboy (PyBoy): The PyBoy emulator instance.
            parameters (dict): Configuration parameters for the emulator.
            named_screen_regions (Optional[list[NamedScreenRegion]]): List of named screen regions to monitor.
        """
        self.variant = variant
        """ The variant of the Pokemon game. """
        verify_parameters(parameters)
        if self.variant not in VARIANTS:
            log_error(f"Variant {self.variant} not recognized. Available variants: {list(VARIANTS.keys())}", parameters)
        super().__init__(pyboy, parameters, named_screen_regions)

    @abstractmethod
    def is_in_dialogue(self) -> bool:
        """
        Determines if the player is currently in a dialogue state or reading text from a sign, interacting with an object etc.
        Essentially anything that causes text to appear at the bottom of the screen that isn't a battle, pc or menu.

        Returns:
            bool: True if in dialogue, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def is_menu_open(self) -> bool:
        """
        Determines if the menu is currently open.

        Returns:
            bool: True if the menu is open, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def is_in_battle(self) -> bool:
        """
        Determines if the player is currently in a battle.

        Returns:
            bool: True if in battle, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def is_pokedex_open(self) -> bool:
        """
        Determines if the Pokedex is currently open.
        Returns:
            bool: True if the Pokedex is open, False otherwise.
        """
        # for pokemon Red, is at last entry if the screen does not change on a down input. Same for Crystal
        raise NotImplementedError

    def is_bag_open(self) -> bool:
        """
        Determines if the bag is currently open.
        Returns:
            bool: True if the bag is open, False otherwise.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"PokemonGameStateParser(variant={self.variant}, base={VARIANTS[self.variant]})"