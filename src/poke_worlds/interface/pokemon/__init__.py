from typing import Type, Union
from poke_worlds.utils import load_parameters, log_info, log_error, log_warn
from poke_worlds.emulation.pokemon import get_pokemon_emulator, infer_variant
from poke_worlds.interface.pokemon.environments import PokemonEnvironment, PokemonRedChooseCharmanderFastEnv
# check if transformers is installed, and only then import the expensive environments
try:
    import transformers  # noqa: F401
except ImportError:
    log_warn("Transformers library not found. Expensive Pokemon environments will not be available.")
    from poke_worlds.interface.pokemon.environments import PokemonEnvironment as PokemonHighLevelEnvironment  # I don't know how else to sort this out 
else:
    from poke_worlds.interface.pokemon.expensive_environments import PokemonHighLevelEnvironment
from poke_worlds.interface.controller import Controller, LowLevelController

from typing import Optional


_VARIANT_TO_ENVIRONMENT = {
    "pokemon_red": {
        "default": PokemonEnvironment,
        "charmander_enthusiast": PokemonRedChooseCharmanderFastEnv,
        "high_level": PokemonHighLevelEnvironment
    },
    "pokemon_brown": {
        "default": PokemonEnvironment
    },
    "pokemon_crystal": {
        "default": PokemonEnvironment
    },
    "pokemon_starbeasts": {
        "default": PokemonEnvironment
    },
    "pokemon_fools_gold": {
        "default": PokemonEnvironment
    },
}


def get_pokemon_environment(game_variant: str, *, parameters: dict = None, controller: Optional[Controller]=None, environment_variant: Union[str, Type[PokemonEnvironment]]="default", environment_kwargs: dict={}, **kwargs) -> PokemonEnvironment:
    """
    Factory function to create a PokemonEnvironment with the specified game variant and parameters.

    Args:
        game_variant (str): The variant of the Pokemon game to emulate.
        parameters (dict, optional): Additional parameters for the environment.
        controller (Controller, optional): The controller to use for the environment. If None, a LowLevelController is used.
        environment_variant (str, optional): The variant of the environment to create.
        environment_kwargs (dict, optional): Additional keyword arguments for the environment constructor.
        **kwargs: Additional keyword arguments passed to the `get_pokemon_emulator` function.

    Returns:
        PokemonEnvironment: An instance of the requested PokemonEnvironment.
    """
    parameters = load_parameters(parameters)
    game_variant = infer_variant(game_variant, parameters) 
    if game_variant not in _VARIANT_TO_ENVIRONMENT:
        log_error(f"Unsupported game variant '{game_variant}'. Available variants are: {list(_VARIANT_TO_ENVIRONMENT.keys())}", parameters)
    environment_class = None
    if isinstance(environment_variant, str):
        variant_map = _VARIANT_TO_ENVIRONMENT[game_variant]
        if environment_variant not in variant_map:
            log_error(f"Unsupported environment variant '{environment_variant}' for game variant '{game_variant}'. Available variants are: {list(variant_map.keys())}", parameters)
        environment_class = variant_map[environment_variant]
    else:
        if not issubclass(environment_variant, PokemonEnvironment):
            log_error(f"The provided environment_variant is not a subclass of PokemonEnvironment.", parameters)
        environment_class = environment_variant
    kwargs = environment_class.override_emulator_kwargs(kwargs)
    emulator = get_pokemon_emulator(game_variant=game_variant, parameters=parameters, **kwargs)
    if controller is None:
        controller = LowLevelController(parameters=parameters)
    environment = environment_class(emulator=emulator, controller=controller, 
                                                   parameters=parameters, **environment_kwargs)
    return environment
    