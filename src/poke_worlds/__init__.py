"""
Documentation for the <img src="https://dhananjayashok.github.io/PokeWorlds/assets/logo_tilt.png" width="70px"> package. 

### Navigating the Documentation
This page and some of the nested links will look a bit empty, but rest assured, the documentation is there. You just need to look for it a little. 
There are two ways to navigate the documentation:
1. **Sidebar Navigation**: Use the sidebar on the left to explore different modules. Expand the sections to find detailed documentation. 
2. **Search Functionality**: Use the search bar at the top of the sidebar to quickly find specific classes, methods, or keywords within the documentation.

### Package Structure
* The `emulation` submodule is the "root" or "core" of this project and handles all of the emulation logic. If you are interested in understanding how the code emulates the games, creates state spaces, implements the low level action controller etc., this is where you should start. See the `emulation` submodule documentation for more details.


### Notable Imports


**Emulation Submodule:**
* `get_pokemon_emulator`: Factory function to get an emulator instance for a specified Pokémon game variant.
* `AVAILABLE_POKEMON_VARIANTS`: List of available Pokémon game variants supported by the package.
"""
from poke_worlds.emulation.pokemon import AVAILABLE_POKEMON_VARIANTS, get_pokemon_emulator