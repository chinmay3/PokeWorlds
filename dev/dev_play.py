from poke_worlds import get_pokemon_emulator, AVAILABLE_POKEMON_VARIANTS
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--init_state", type=str, default=None, help="Name of the initial state file")
def main(variant, init_state):
    env = get_pokemon_emulator(game_variant=variant, init_state_name=init_state, headless=False)
    env._dev_play()

if __name__ == "__main__":
    main()