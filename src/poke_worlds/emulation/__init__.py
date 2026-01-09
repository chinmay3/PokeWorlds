"""
This submodule handles the emulation of Gameboy games. The core classes here are:
1. `StateParser`: Responsible for parsing the game state from the screen, and the memory states of the gameboy. Can be used to extract relevant information from the game state at each step (e.g. player location, current party, inventory items, etc).
2. `StateTracker`: Uses the `StateParser` and pre-registers specific state tracking logic that is called after every button press in the game. This class is responsible for maintaining the game state information at each step. It also tracks relevant metrics over playthroughs (e.g. number of battles won) and aggregates metrics over multiple playthroughs (across resets) if required. 
3. `Emulator`: The core class that handles the emulation of the game. It carries a `StateParser` and `StateTracker`. Every step runs a single button press, and ensures that the state tracker is properly updated.

Briefly skim the documentation for each of these classes to understand their roles, the fundamental structure they impose and how they interact with each other.

In practice, unless you are implementing new games, you will not need to interact with these base classes directly. Each have subclasses that implement Pokémon specific logic and provides some additional structure.
This is what you should familiarize yourself with most deeply if you wish to use this package as a black box API and not care about the internals. Go to the `pokemon` submodule and look at the classes defined there. 
"""
from poke_worlds.emulation.emulator import Emulator, LowLevelActions
from poke_worlds.emulation.tracker import StateTracker
from poke_worlds.emulation.parser import StateParser