"""
The classes here define the most high level abstractions of the fundamental classes used for control and environment interaction.
1. `HighLevelAction`: The base class for all high level actions that can be executed through the controller. Handles the logic of mapping high level actions to low level actions on the emulator and can be seamlessly converted to and from Gym action spaces.
2. `Controller`: The class that organizes and manages multiple `HighLevelAction` instances, providing a unified Gym action space and methods to execute a variety of high level actions on the emulator.
3. `Environment`: The class that implements the Gym API, combining an `Emulator` and a `Controller` to provide observations, rewards, and episode termination logic.

You can skim the abstract base class documentation to understand the structure they follow, but where you focus will depend on your goals:


"""
from poke_worlds.interface.action import HighLevelAction
from poke_worlds.interface.controller import Controller
from poke_worlds.interface.environment import Environment, History