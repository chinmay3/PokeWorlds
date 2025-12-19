"""
The classes here define the most high level abstractions of the fundamental classes used for emulation. 
1. `Emulator`: The core class that handles the emulation of Pokémon games.
2. `StateParser`: Responsible for parsing the game state from the emulator when called. Is called by a `StateTracker`.
3. `StateTracker`: Keeps track of the game state over time, allowing for state comparisons and history tracking. Is called once per step in the `Emulator`. 

Briefly skim the documentation for each of these classes to understand their roles, the fundamental structure they impose and how they interact with each other.

In practice, unless you are implementing new games, you will not need to interact with these base classes directly. Each have subclasses that implement Pokémon specific logic and provides some additional structure.
This is what you should familiarize yourself with most deeply if you wish to use this package as a black box API and not care about the internals. Go to the `pokemon` submodule and look at the parsers and trackers defined there. 
"""