from poke_env import BasicPokemonRedEmulator

if __name__ == "__main__":
    env = BasicPokemonRedEmulator(parameters=None, headless=False)
    env._human_step_play()