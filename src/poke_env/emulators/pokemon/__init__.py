from poke_env.utils import load_parameters, log_error, verify_parameters
from poke_env.emulators.emulator import Emulator
from poke_env.emulators.pokemon.parsers import PokemonRedGameStateParser, PokemonBrownGameStateParser
from typing import Optional

VARIANT_TO_GB_NAME = {
    "pokemon_red": "PokemonRed.gb",
    "pokemon_brown": "PokemonBrown.gb",
    "pokemon_crystal": "PokemonCrystal.gbc",
    "pokemon_fools_gold": "PokemonFoolsGold.gbc",
    "pokemon_prism": "PokemonPrism.gbc",
    "pokemon_quarantine_crystal": "PokemonQuarantineCrystal.gbc",
    "pokemon_starbeasts": "PokemonStarBeasts.gbc",
}
""" Expected save name for each variant. Save the file to <project_root>/<variant_name>_rom_data/<gb_name>"""

_VARIANT_TO_BASE_MAP = {
    "pokemon_red": "pokemon_red", 
    "pokemon_brown": "pokemon_red",
    "pokemon_crystal": "pokemon_crystal",
    "pokemon_fools_gold": "pokemon_crystal",
    "pokemon_prism": "pokemon_crystal",
    "pokemon_quarantine_crystal": "pokemon_crystal",
    "pokemon_starbeasts": "pokemon_crystal",
}
""" Mapping of variant names to base game types."""

_VARIANT_TO_PARSER = {
    "pokemon_red": PokemonRedGameStateParser,
    "pokemon_brown": PokemonBrownGameStateParser,
    "pokemon_crystal": None,  # To be implemented
    "pokemon_fools_gold": None,  # To be implemented
    "pokemon_prism": None,  # To be implemented
    "pokemon_quarantine_crystal": None,  # To be implemented
    "pokemon_starbeasts": None,  # To be implemented
}
""" Mapping of variant names to their corresponding GameStateParser classes."""


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


def get_pokemon_emulator(variant: str, parameters: Optional[dict] = None, init_state_name: str = None, **kwargs) -> Emulator:
    """ 
    Factory method to get a Pokemon emulator instance based on the specified variant.
    Args:
        variant (str): The variant of the Pokemon game (e.g., "pokemon_red", "pokemon_crystal").
        parameters (dict, optional): Additional parameters for emulator configuration.
        init_state_name (str, optional): Name of the initial state file to load (without path).
        **kwargs: Additional keyword arguments to pass to the Emulator constructor (e.g. headless)
    Returns:
        Emulator: An instance of the Emulator class configured for the specified variant.
    """
    parameters = load_parameters(parameters)
    variant = _infer_variant(variant, parameters=parameters)
    gb_path = parameters[f"{variant}_rom_data_path"] + "/" + VARIANT_TO_GB_NAME[variant]
    init_state = None
    if init_state_name is not None:
        if not init_state_name.endswith(".state"):
            init_state_name = init_state_name + ".state"
        init_state = parameters[f"{variant}_rom_data_path"] + "/states/" + init_state_name
    else:
        init_state = parameters[f"{variant}_rom_data_path"] + "/states/default.state"
    game_state_parser_class = _VARIANT_TO_PARSER[variant]
    if game_state_parser_class is None:
        log_error(f"GameStateParser for variant '{variant}' is not yet implemented.", parameters)
    emulator = Emulator(name=variant, gb_path=gb_path, game_state_parser_class=game_state_parser_class, 
                        init_state=init_state, parameters=parameters, **kwargs)
    return emulator