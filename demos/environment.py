from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, LowLevelController, RandomPlayController, LowLevelPlayController
from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController
from poke_worlds.interface.pokemon.environments import PokemonHighLevelEnvironment
from tqdm import tqdm
import matplotlib.pyplot as plt
import click


@click.command()
@click.option("--play_mode", type=click.Choice(["human", "random", "restricted_random", "grouped_random"]), default="random", help="Play mode: 'random' for random actions.")
@click.option("--environment_variant", type=str, default="charmander_enthusiast", help="The environment variant to use.")
@click.option("--render", type=bool, default=False, help="Whether to render the environment with PyGame.")
@click.option("--save_video", type=bool, default=None, help="Whether to save a video of the gameplay. If not specified, uses default from config.")
def main(play_mode, environment_variant, render, save_video):
    if play_mode == "random":
        controller = LowLevelController()
    elif play_mode == "restricted_random":
        controller = LowLevelPlayController()
    elif play_mode == "grouped_random":
        controller = RandomPlayController()
    else:
        controller = LowLevelPlayController()
    if play_mode != "human":
        environment = get_pokemon_environment(game_variant="pokemon_red", controller=controller, save_video=save_video,
                                    environment_variant=environment_variant, max_steps=500, headless=True)
        steps = 0
        max_steps = 500
        pbar = tqdm(total=max_steps)
        rewards = []
        while steps < max_steps:
            action = environment.action_space.sample()
            observation, reward, terminated, truncated, info = environment.step(action)
            rewards.append(reward)
            if render:
                environment.render()
            if terminated or truncated:
                break
            steps += 1
            pbar.update(1)
        pbar.close()    
        environment.close()
        if render:
            # Plot rewards over time
            plt.plot(rewards)
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.title("Rewards over Time in Charmander Enthusiast Environment")
            plt.show()
    else:
        environment = get_pokemon_environment(game_variant="pokemon_red", environment_variant=environment_variant,
                                              controller=PokemonStateWiseController(), save_video=save_video,
                                              max_steps=500, headless=True, init_state="starter")
        environment.human_step_play()

if __name__ == "__main__":
    main()