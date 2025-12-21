from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, LowLevelController, RandomPlayController, LowLevelPlayController
from tqdm import tqdm
import click
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("The environment demo uses matplotlib to plot rewards over time. Please run 'pip install matplotlib' to install it.")


@click.command()
@click.option("--play_mode", type=click.Choice(["random", "restricted_random", "grouped_random"]), default="random", help="Play mode: 'random' for random actions.")
@click.option("--render", type=bool, default=False, help="Whether to render the environment with PyGame.")
@click.option("--save_video", type=bool, default=None, help="Whether to save a video of the gameplay. If not specified, uses default from config.")
def main(play_mode, render, save_video):
    if play_mode == "random":
        controller = LowLevelController()
    elif play_mode == "restricted_random":
        controller = LowLevelPlayController()
    elif play_mode == "grouped_random":
        controller = RandomPlayController()
    else:
        raise ValueError(f"Unknown play mode: {play_mode}")
    environment = get_pokemon_environment(game_variant="pokemon_red", controller=controller, save_video=save_video,
                                        environment_variant="charmander_enthusiast", max_steps=500, headless=True)    
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
    # Plot rewards over time
    plt.plot(rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Rewards over Time in Charmander Enthusiast Environment")
    plt.show()

if __name__ == "__main__":
    main()