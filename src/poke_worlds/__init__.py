"""
Documentation for the <img src="https://dhananjayashok.github.io/PokeWorlds/assets/logo_tilt.png" width="70px"> package. 

### Navigating the Documentation
This page and some of the nested links will look a bit empty, but rest assured, the documentation is there. You just need to look for it a little. 
There are two ways to navigate the documentation:
1. **Sidebar Navigation**: Use the sidebar on the left to explore different modules. Expand the sections to find detailed documentation. 
2. **Search Functionality**: Use the search bar at the top of the sidebar to quickly find specific classes, methods, or keywords within the documentation.

### Package Structure
* The `emulation` submodule is the "root" of this project. It handles GameBoy emulation, allows users to parse the game state and infer things about the game state, and tracks relevant metrics and game state information at every step of gameplay. Look here if you want to:
    - Parse the game state to extract relevant information (e.g. Player location, current party, inventory items).
    - Check for specific triggers or events in the game (e.g. Did the player open the menu? Did the player enter a battle?).
    - Track specific metrics or information over multiple steps of gameplay (e.g. Number of regions visited, number of battles won, etc).
    - Modify the emulator to change how the game is run (e.g. automatically skip dialogues and cutscenes, etc).
* The `interface` submodue is where we create the actions that agents can take in the game (e.g. `Seek(looking for prof oak, npc)` that tries to locate an NPC on the screen, determine which one is prof oak, move towards him and then interact with him, all in one command). The submodule also has the Gym API integration which allows us to train RL agents on top of the game. Look here if you want to:
    - Create new high level actions, which compose multiple button presses into a single action. These actions can use VLMs to help guide their behavior (e.g. using VLMs to determine which grid cell of the screen contains the target object to interact with).
    - Create new environments or test scenarios, with different observation spaces, reward functions, and termination conditions for the game. 
* The `execution` submodule contains the highest level APIs in this project. This allows users (or LMs) to spin up a VLM agent that tries to use the specified high level actions to execute a short, arbitrary task in a given game environment. Look here if you want to:
    - Use a (V)LM to construct plans for a high level goal in the game (e.g. Clear the first gym), analyze a game playthrough to determine its progress towards this goal and then adjust its strategy accordingly.
    - Prompt engineer VLM agents to better use the existing high level actions


### Notable API Imports

**Emulation Submodule:**
* `AVAILABLE_GAMES`: List of available game variants supported by the package.
* `get_emulator`: Factory function to get an emulator instance for a specified game variant.

**Interface Submodule:**
* `get_environment`: Factory function to get an environment instance for a specified game variant and environment variant.

**Execution Submodule:**
* `Executor`: Class that manages the execution of high-level tasks in the game environment using VLM agents.

"""
from poke_worlds.emulation.registry import AVAILABLE_GAMES, get_emulator
from poke_worlds.interface.registry import get_environment
from poke_worlds.execution.executor import Executor