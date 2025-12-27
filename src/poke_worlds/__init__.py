"""
Documentation for the <img src="https://dhananjayashok.github.io/PokeWorlds/assets/logo_tilt.png" width="70px"> package. 

### Navigating the Documentation
This page and some of the nested links will look a bit empty, but rest assured, the documentation is there. You just need to look for it a little. 
There are two ways to navigate the documentation:
1. **Sidebar Navigation**: Use the sidebar on the left to explore different modules. Expand the sections to find detailed documentation. 
2. **Search Functionality**: Use the search bar at the top of the sidebar to quickly find specific classes, methods, or keywords within the documentation.

### Package Structure
* The `interface` submodule contains the highest level APIs in this project, including the Gym-Style API implementations. If you wish to simply use this repository to train the best agents, look no further than the `get_pokemon_environment` method. See the `interface` submodule documentation for more details.
* The `emulation` submodule is the "root" or "core" of this project and handles all of the emulation logic. If you are interested in understanding how the code emulates the games, creates state spaces, implements the low level action controller etc., this is where you should start. See the `emulation` submodule documentation for more details.


### Notable Imports


**Emulation Submodule:**
* `get_pokemon_emulator`: Factory function to get an emulator instance for a specified Pokémon game variant.
* `AVAILABLE_POKEMON_VARIANTS`: List of available Pokémon game variants supported by the package.

**Interface Submodule:**
* `get_pokemon_environment`: Factory function to create a Pokémon environment with the specified game variant and parameters.
* `LowLevelController`: A controller that allows low-level interaction with the emulator.
* `LowLevelPlayController`: A controller that allows low-level play interactions, but not menu buttons with the emulator.
* `RandomPlayController`: A controller that samples random actions from fixed groups for gameplay. Is a minimal example of a higher-level controller.

"""
from poke_worlds.emulation.pokemon import AVAILABLE_POKEMON_VARIANTS, get_pokemon_emulator
from poke_worlds.interface.pokemon import get_pokemon_environment

from poke_worlds.interface.controller import LowLevelController, RandomPlayController, LowLevelPlayController
from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController
try:
    from poke_worlds.interface.pokemon.environments import PokemonHighLevelEnvironment
except ImportError:
    print("Could not import PokemonHighLevelEnvironment. Likely missing transformers or broken pip installation ")
    PokemonHighLevelEnvironment = None