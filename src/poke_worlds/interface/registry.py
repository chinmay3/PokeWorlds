"""
Keeps a record of:
- Available `Controller`s for each game, with string identifiers. There is NO default controller for any game.
- Available `Environment`s for each game, with string identifiers and a default environment for each game.

Provides methods to access these. 
"""
from typing import Dict, Type, Optional, Union
from poke_worlds.utils import log_error, load_parameters, log_warn
from poke_worlds.emulation.registry import get_emulator, AVAILABLE_GAMES, infer_game
from poke_worlds.interface.controller import Controller, _ALWAYS_VALID_CONTROLLERS
from poke_worlds.interface.environment import Environment
from poke_worlds.interface.pokemon.environments import PokemonEnvironment, PokemonRedChooseCharmanderFastEnv, PokemonOCREnvironment
from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController

_project_parameters = load_parameters()

AVAILABLE_ENVIRONMENTS: Dict[str, Dict[str, Type[Environment]]] = {
    "pokemon_red": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "charmander_enthusiast": PokemonRedChooseCharmanderFastEnv,
    },
    "pokemon_brown": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
    },

    "pokemon_starbeasts": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,            
    },

    "pokemon_crystal": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
    },

    "pokemon_prism": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
    },

    "pokemon_fools_gold": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
    },
}

AVAILABLE_CONTROLLERS: Dict[str, Dict[str, Type[Controller]]] = {
    "pokemon_red": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_brown": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_crystal": {
        "state_wise": PokemonStateWiseController,
    },

    "pokemon_starbeasts": {
        "state_wise": PokemonStateWiseController,
    },

    "pokemon_prism": {
        "state_wise": PokemonStateWiseController,
    },

    "pokemon_fools_gold": {
        "state_wise": PokemonStateWiseController,
    },
}

for game in AVAILABLE_GAMES:
    if game not in AVAILABLE_ENVIRONMENTS:
        log_warn(f"No environments registered for game variant '{game}'. Will error out if you try to get an environment for this game variant." , _project_parameters)
    if game not in AVAILABLE_CONTROLLERS:
        log_warn(f"No controllers registered for game variant '{game}'. Will error out if you try to get a controller for this game variant." , _project_parameters)
    else:
        for valid_controller_key in _ALWAYS_VALID_CONTROLLERS:
            if valid_controller_key in AVAILABLE_CONTROLLERS[game]:
                log_error(f"Controller key '{valid_controller_key}' for game variant '{game}' is reserved for always valid controllers. Do not add a controller with this key in the registry.", _project_parameters)
            AVAILABLE_CONTROLLERS[game][valid_controller_key] = _ALWAYS_VALID_CONTROLLERS[valid_controller_key]


def get_controller(game: str, *, controller_variant: Union[str, Type[Controller], Controller], parameters: dict=None) -> Type[Controller]:
    """
    Factory function to get a Controller class for the specified game variant and controller variant.

    Args:
        game (str): The variant of the game to emulate.
        controller_variant (Union[str, Type[Controller], Controller]): The variant of the controller to create or the Controller class itself or an instance of Controller.
        parameters (dict, optional): Additional parameters for error logging.

    Returns:
        Controller: An instance of the requested Controller class.
    """
    parameters = load_parameters(parameters)
    game = infer_game(game, parameters)
    if game not in AVAILABLE_CONTROLLERS:
        log_error(f"No controllers registered for game variant '{game}'.", parameters)
    available_controllers = AVAILABLE_CONTROLLERS[game]
    if isinstance(controller_variant, str):
        if controller_variant not in available_controllers:
            log_error(f"Unsupported controller variant '{controller_variant}' for game variant '{game}'. Available variants are: {list(available_controllers.keys())}", parameters)
        return available_controllers[controller_variant](parameters=parameters)
    available_variants = [cls for cls in available_controllers.values()]
    is_eqs = [controller_variant == cls for cls in available_variants]
    is_inst = [isinstance(controller_variant, cls) for cls in available_variants]
    if not any(is_eqs) and not any(is_inst):
        log_error(f"The provided controller_variant is not a registered Controller class for game variant '{game}'. Available variants are: {list(available_controllers.keys())}", parameters)
    else:
        if any(is_inst):
            return controller_variant
        else: # then it must be a class that matches one of the available variants
            return controller_variant(parameters=parameters)
        
def get_environment(game: str, *, environment_variant: str, controller_variant: Union[str, Type[Controller], Controller], parameters: dict=None, **emulator_kwargs) -> Environment:
    """
    Factory function to get an Environment instance for the specified game variant and environment variant.

    Args:
        game (str): The variant of the game to emulate.
        environment_variant (str): The variant of the environment to create.
        controller_variant (Union[str, Type[Controller], Controller]): The variant of the controller to create or the Controller class itself or an instance of Controller.
        parameters (dict, optional): Additional parameters for the environment and error logging. Must come from `load_parameters`.
        **emulator_kwargs: Additional keyword arguments passed to the `get_emulator` method.

    Returns:
        Environment: An instance of the requested Environment class.
    """
    parameters = load_parameters(parameters)
    game = infer_game(game, parameters)
    if game not in AVAILABLE_ENVIRONMENTS:
        log_error(f"No environments registered for game variant '{game}'.", parameters)
    available_environments = AVAILABLE_ENVIRONMENTS[game]
    if environment_variant not in available_environments:
        log_error(f"Unsupported environment variant '{environment_variant}' for game variant '{game}'. Available variants are: {list(available_environments.keys())}. Make sure to add your environment to the registry if you think this is a mistake.", parameters)
    controller = get_controller(game, controller_variant=controller_variant, parameters=parameters)
    environment_class = available_environments[environment_variant]
    emulator_kwargs["game"] = game
    emulator_kwargs = environment_class.override_emulator_kwargs(emulator_kwargs)
    emulator = get_emulator(parameters=parameters, **emulator_kwargs)
    return environment_class(emulator=emulator, controller=controller, parameters=parameters)