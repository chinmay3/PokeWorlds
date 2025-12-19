from poke_worlds.utils import load_parameters, log_error, verify_parameters
from poke_worlds.emulation.emulator import Emulator, LowLevelActions
from poke_worlds.emulation.pokemon.parsers import PokemonStateParser, PokemonRedStateParser, PokemonBrownStateParser, PokemonCrystalStateParser, PokemonStarBeastsStateParser, PokemonPrismStateParser, PokemonFoolsGoldStateParser
from poke_worlds.emulation.pokemon.trackers import CorePokemonMetrics, CorePokemonTracker, PokemonRedStarterTracker
from typing import Optional

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

_VARIANT_TO_PARSER = {
    "pokemon_red": PokemonRedStateParser,
    "pokemon_brown": PokemonBrownStateParser,
    "pokemon_crystal": PokemonCrystalStateParser, 
    "pokemon_starbeasts": PokemonStarBeastsStateParser,
    "pokemon_fools_gold": PokemonFoolsGoldStateParser,  # To be implemented
    "pokemon_prism": PokemonPrismStateParser,  # To be implemented
}
""" Mapping of variant names to their corresponding StateParser classes."""


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


def _infer_variant(variant: str, parameters: dict) -> str:
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


class PokemonEmulator(Emulator):
    """
    Almost the exact same as Emulator, but forces the agent to not mess with the menu options cursor.
    """
    def __init__(self, *args, **kwargs):
        if "state_parser_class" not in kwargs:
            if len(args) < 3:
                log_error("Malformed initialization of Emulator")
            else:
                state_parser_class = args[2]
        else:
            state_parser_class = kwargs["state_parser_class"]
        if not issubclass(state_parser_class, PokemonStateParser):
            log_error("state_parser_class must be a subclass of PokemonStateParser")
        super().__init__(*args, **kwargs)

    def step(self, action):
        rets = super().step(action)
        if self.state_parser.is_hovering_over_options_in_menu(self.get_current_frame()):
            # force the agent to click the up button to get off the options
            self.run_action_on_emulator(LowLevelActions.PRESS_ARROW_UP)
        return rets



def get_pokemon_emulator(game_variant: str, parameters: Optional[dict] = None, init_state_name: str = None, tracker_variant: str = "default", **kwargs) -> Emulator:
    """ 
    Factory method to get a Pokemon emulator instance based on the specified variant.
    Args:
        game_variant (str): The variant of the Pokemon game (e.g., "pokemon_red", "pokemon_crystal").
        parameters (dict, optional): Additional parameters for emulator configuration.
        init_state_name (str, optional): Name of the initial state file to load (without path).
        tracker_variant (str): The variant of the state tracker to use.
        **kwargs: Additional keyword arguments to pass to the Emulator constructor (e.g. headless)
    Returns:
        Emulator: An instance of the Emulator class configured for the specified variant.
    """
    parameters = load_parameters(parameters)
    game_variant = _infer_variant(game_variant, parameters=parameters)
    gb_path = parameters[f"{game_variant}_rom_data_path"] + "/" + VARIANT_TO_GB_NAME[game_variant]
    init_state = None
    if init_state_name is not None:
        if not init_state_name.endswith(".state"):
            init_state_name = init_state_name + ".state"
        init_state = parameters[f"{game_variant}_rom_data_path"] + "/states/" + init_state_name
    else:
        init_state = parameters[f"{game_variant}_rom_data_path"] + "/states/default.state"
    state_parser_class = _VARIANT_TO_PARSER[game_variant]
    if state_parser_class is None:
        log_error(f"StateParser for variant '{game_variant}' is not yet implemented.", parameters)
    state_tracker_class = _VARIANT_TO_TRACKER[game_variant][tracker_variant]
    if state_tracker_class is None:
        log_error(f"StateTracker for variant '{game_variant}' is not yet implemented.", parameters)
    emulator = PokemonEmulator(name=game_variant, gb_path=gb_path,
                        state_parser_class=state_parser_class, state_tracker_class=state_tracker_class,
                        init_state=init_state, parameters=parameters, **kwargs)
    return emulator