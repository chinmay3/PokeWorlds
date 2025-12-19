from poke_worlds import get_pokemon_emulator, AVAILABLE_POKEMON_VARIANTS
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--init_state", type=str, default=None, help="Name of the initial state file")
@click.option("--play_mode", type=click.Choice(["human", "random"]), default="human", help="Play mode: 'human' for manual play, 'random' for random actions.")
@click.option("--headless", type=bool, default=None, help="Whether to run the emulator in headless mode (no GUI).")
@click.option("--save_video", type=bool, default=None, help="Whether to save a video of the gameplay. If not specified, uses default from config.")
def main(variant, init_state, play_mode, headless, save_video):
    if play_mode == "human":
        env = get_pokemon_emulator(game_variant=variant, init_state_name=init_state, headless=False, save_video=save_video)
        env.human_play()
    else:
        if headless != False:
            headless = True
        env = get_pokemon_emulator(game_variant=variant, init_state_name=init_state, headless=headless, save_video=save_video)
        env.random_play()

if __name__ == "__main__":
    main()