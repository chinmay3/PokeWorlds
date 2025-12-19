from poke_worlds import get_pokemon_emulator, AVAILABLE_POKEMON_VARIANTS
from poke_worlds.interface.controller import RandomController
import numpy as np
from tqdm import tqdm
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--init_state", type=str, default=None, help="Name of the initial state file")
@click.option("--play_mode", type=click.Choice(["human", "random"]), default="human", help="Play mode: 'human' for manual play, 'random' for random actions.")
@click.option("--headless", type=bool, default=None, help="Whether to run the emulator in headless mode (no GUI).")
@click.option("--save_video", type=bool, default=None, help="Whether to save a video of the gameplay. If not specified, uses default from config.")
def main(variant, init_state, play_mode, headless, save_video):
    if play_mode == "human":
        emulator = get_pokemon_emulator(game_variant=variant, init_state_name=init_state, headless=False, save_video=save_video)
        emulator.human_play()
    else:
        if headless != False:
            headless = True
        emulator = get_pokemon_emulator(game_variant=variant, init_state_name=init_state, headless=headless, save_video=save_video)
        #emulator.random_play() # This plays with low level actions. See this to understand how the emulator works.
        controller = RandomController()
        controller.assign_emulator(emulator)
        steps = 0
        max_steps = 500
        pbar = tqdm(total=max_steps)
        while steps < max_steps:
            valid_actions = controller.get_valid_actions()
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            reports, success = controller.execute_action(action)
            steps += 1
            pbar.update(1)
        pbar.close()
        emulator.close()




if __name__ == "__main__":
    main()