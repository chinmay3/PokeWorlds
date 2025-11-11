from poke_env import PokemonRedEmulator

if __name__ == "__main__":
    env = PokemonRedEmulator(parameters=None)
    env.reset()
    while True:
        env.pyboy.tick(1, True)
        env.render()
        truncated = env.step_count >= env.max_steps - 1
        if truncated:
            break
    env.close()