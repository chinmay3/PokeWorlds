"""
Documentation for the ![PokéWorlds](https://dhananjayashok.github.io/PokeWorlds/assets/logo_tilt.png) package. 

### Navigating the Documentation
This page and some of the nested links will look a bit empty, but rest assured, the documentation is there. You just need to look for it a little. 
There are two ways to navigate the documentation:
1. **Sidebar Navigation**: Use the sidebar on the left to explore different modules. Expand the sections to find detailed documentation. 
2. **Search Functionality**: Use the search bar at the top of the sidebar to quickly find specific classes, methods, or keywords within the documentation.

### Overview
The `emulators` subpackage does most of the heavy lifting in this project. It defines the most high level abstractions of the fundamental classes:
1. `Emulator`: The core class that handles the emulation of Pokémon games.
2. `StateParser`: Responsible for parsing the game state from the emulator when called. Is called by a `StateTracker`.
3. `StateTracker`: Keeps track of the game state over time, allowing for state comparisons and history tracking. Is called once per step in the `Emulator`. 

Briefly skim the documentation for each of these classes to understand their roles, the fundamental structure they impose and how they interact with each other.

In practice, unless you are implementing new games, you will not need to interact with these base classes directly. The `StateParser` and `StateTracker` have subclasses that implement Pokémon specific logic and provides some additional structure.
This is what you should familiarize yourself with most deeply if you wish to use this package as a black box API and not care about the internals. Search for the `pokemon` submodule and look at the parsers and trackers defined there. 

### Notable Imports
* `LowLevelActions`: Enum defining low level actions that can be performed in the emulator.
* `AVAILABLE_POKEMON_VARIANTS`: List of available Pokémon game variants supported by the package.
* `get_pokemon_emulator`: Factory function to get an emulator instance for a specified Pokémon game variant.

"""
from poke_worlds.emulators.emulator import LowLevelActions
from poke_worlds.emulators.pokemon import AVAILABLE_POKEMON_VARIANTS, get_pokemon_emulator