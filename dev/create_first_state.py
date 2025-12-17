from poke_env.emulators.emulator import Emulator
from poke_env import AVAILABLE_POKEMON_VARIANTS
from poke_env.emulators.pokemon import VARIANT_TO_GB_NAME
from poke_env.utils import load_parameters, log_error
import click

@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), help="Variant of the Pokemon game to create the first state for.")
@click.option("--state_name", default="default", type=str, help="Name of the state to create, e.g., default")
def create_first_state(variant: str, state_name: str):
    """ Creates the first state for a given GameBoy ROM file. """
    parameters = load_parameters()
    if f"{variant}_rom_data_path" not in parameters:
        log_error(f"ROM data path not found for variant: {variant}. Add {variant}_rom_data_path to the config files. See configs/pokemon_red_vars.yaml for an example", parameters)
    gb_path = parameters[f"{variant}_rom_data_path"] + "/" + VARIANT_TO_GB_NAME[variant]
    state_path = parameters[f"{variant}_rom_data_path"] + f"/states/{state_name}"
    Emulator.create_first_state(gb_path, state_path)

if __name__ == "__main__":
    create_first_state()