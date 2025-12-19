from poke_worlds import get_pokemon_emulator, AVAILABLE_POKEMON_VARIANTS
import os
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--sav_file", type=str, default=None, help="Path to save the .sav file")
@click.option("--state_name", type=str, default="tmp", help="Name of the state")
def main(variant, sav_file, state_name):
    env = get_pokemon_emulator(game_variant=variant, headless=True)
    if not hasattr(env.state_parser, "rom_data_path"):
        raise ValueError("Parsers without rom_data_path are not supported for save state creation.")
    state_dir = env.state_parser.rom_data_path + "/states/"
    os.makedirs(state_dir, exist_ok=True)
    if not state_name.endswith(".state"):
        state_name = state_name + ".state"
    state_file = os.path.join(state_dir, state_name)
    env._sav_to_state(sav_file=sav_file, state_file=state_file)

if __name__ == "__main__":
    main()