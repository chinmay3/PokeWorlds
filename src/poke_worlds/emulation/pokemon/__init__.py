from poke_worlds.emulation.pokemon.emulators import PokemonEmulator
from poke_worlds.utils import load_parameters, log_error, verify_parameters
from poke_worlds.emulation.emulator import Emulator
from poke_worlds.emulation.tracker import StateTracker
from poke_worlds.emulation.pokemon.parsers import MemoryBasedPokemonRedStateParser, PokemonBrownStateParser, PokemonCrystalStateParser, PokemonStarBeastsStateParser, PokemonPrismStateParser, PokemonFoolsGoldStateParser
from poke_worlds.emulation.pokemon.trackers import CorePokemonMetrics, CorePokemonTracker, PokemonRedStarterTracker
from typing import Optional, Union, Type

VARIANT_TO_GB_NAME = {
    "pokemon_red": "PokemonRed.gb",
    "pokemon_brown": "PokemonBrown.gb",
    "pokemon_starbeasts": "PokemonStarBeasts.gb",
    "pokemon_crystal": "PokemonCrystal.gbc",
    "pokemon_fools_gold": "PokemonFoolsGold.gbc",
    "pokemon_prism": "PokemonPrism.gbc",
}
""" Expected save name for each variant. Save the file to <project_root>/<variant_name>_rom_data/<gb_name>"""

_VARIANT_TO_BASE_MAP = {
    "pokemon_red": "pokemon_red", 
    "pokemon_brown": "pokemon_red",
    "pokemon_starbeasts": "pokemon_red",
    "pokemon_crystal": "pokemon_crystal",
    "pokemon_fools_gold": "pokemon_crystal",
    "pokemon_prism": "pokemon_crystal",
}
""" Mapping of variant names to base game types."""

_VARIANT_TO_STRONGEST_PARSER = {
    "pokemon_red": MemoryBasedPokemonRedStateParser,
    "pokemon_brown": PokemonBrownStateParser,
    "pokemon_crystal": PokemonCrystalStateParser, 
    "pokemon_starbeasts": PokemonStarBeastsStateParser,
    "pokemon_fools_gold": PokemonFoolsGoldStateParser,
    "pokemon_prism": PokemonPrismStateParser,
}
""" 
Mapping of variant names to their corresponding StateParser classes. 
Unless you have a very good reason, you should always use the STRONGEST possible parser for a given variant. 
The parser itself does not affect performance, as for it to perform a read / screen comparison operation, it must be called upon by the state tracker. 
This means there is never a reason to use a weaker parser. 
"""


_VARIANT_TO_TRACKER = {
    "pokemon_red": {
        "default": CorePokemonTracker,
        "starter_example": PokemonRedStarterTracker
        },
    "pokemon_brown": {
        "default": CorePokemonTracker
    },
    "pokemon_crystal": {
        "default": CorePokemonTracker
    },
    "pokemon_starbeasts": {
        "default": CorePokemonTracker
    },
    "pokemon_fools_gold": {
        "default": CorePokemonTracker
    },
    "pokemon_prism": {
        "default": CorePokemonTracker
    },
}
""" Mapping of variant names to their corresponding StateTracker classes. """


AVAILABLE_POKEMON_VARIANTS = list(VARIANT_TO_GB_NAME.keys())
""" List of available Pokemon game variants. """


def infer_variant(variant: str, parameters: dict) -> str:
    """ 
    Try to infer the variant given a variant name 
    Args:
        variant (str): The variant name to infer.
        parameters (dict): Additional parameters for logging.

    Returns:
        str: The inferred variant name.
    """
    verify_parameters(parameters)
    variant = variant.strip().lower()
    if variant in _VARIANT_TO_BASE_MAP:
        return variant
    options = _VARIANT_TO_BASE_MAP.keys()
    for option in options:
        if variant == option.lower().replace(" ", "_").replace("-", "_"):
            return option
        if variant == option.strip("pokemon_"):
            return option
    log_error(f"Could not infer variant from '{variant}'. Available variants are: {options}", parameters)


def get_pokemon_emulator(game_variant: str, *, parameters: Optional[dict] = None, init_state: str = None, state_tracker_class: Union[str, Type[StateTracker]] = "default", **kwargs) -> Emulator:
    """ 
    Factory method to get a Pokemon emulator instance based on the specified variant.
    Args:
        game_variant (str): The variant of the Pokemon game (e.g., "pokemon_red", "pokemon_crystal").
        parameters (dict, optional): Additional parameters for emulator configuration.
        init_state_name (str, optional): Name of the initial state file to load (without path).
        state_tracker_class (Union[str, Type[StateTracker]]): The variant of the state tracker to use.
        **kwargs: Additional keyword arguments to pass to the Emulator constructor (e.g. headless)
    Returns:
        Emulator: An instance of the Emulator class configured for the specified variant.
    """
    parameters = load_parameters(parameters)
    game_variant = infer_variant(game_variant, parameters=parameters)
    gb_path = parameters[f"{game_variant}_rom_data_path"] + "/" + VARIANT_TO_GB_NAME[game_variant]
    if init_state is not None:
        if not init_state.endswith(".state"):
            init_state = init_state + ".state"
        init_state = parameters[f"{game_variant}_rom_data_path"] + "/states/" + init_state
    else:
        init_state = parameters[f"{game_variant}_rom_data_path"] + "/states/default.state"
    state_parser_class = _VARIANT_TO_STRONGEST_PARSER[game_variant]
    if state_parser_class is None:
        log_error(f"StateParser for variant '{game_variant}' is not yet implemented.", parameters)
    if state_tracker_class is None:
        log_error(f"state_tracker_class cannot be None", parameters)
    if isinstance(state_tracker_class, str):
        state_tracker_class = _VARIANT_TO_TRACKER[game_variant][state_tracker_class]
    emulator = PokemonEmulator(name=game_variant, gb_path=gb_path,
                        state_parser_class=state_parser_class, state_tracker_class=state_tracker_class,
                        init_state=init_state, parameters=parameters, **kwargs)
    return emulator