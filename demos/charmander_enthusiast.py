from poke_worlds import get_pokemon_environment, LowLevelPlayController
import gymnasium as gym
from gymnasium.spaces import Discrete, OneOf
import numpy as np

num_cpu = 4  # Number of processes to use
batch_size = 64
num_epochs = 10
render = False # Whether to render the environment at test time



class OneOfToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate total actions across all sub-spaces
        # Example: OneOf([Discrete(2), Discrete(3)]) -> total 5
        self.sub_spaces = env.action_space.spaces
        self.total_actions = sum(s.n for s in self.sub_spaces)
        self.action_space = Discrete(self.total_actions)

    def action(self, action):
        # Map the single integer back to (choice, sub_action)
        offset = 0
        for i, space in enumerate(self.sub_spaces):
            if action < offset + space.n:
                return (i, action - offset)
            offset += space.n
        return (0, 0) # Fallback


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        original_env = get_pokemon_environment(game_variant="pokemon_red", controller=LowLevelPlayController(),
                                                environment_variant="charmander_enthusiast", max_steps=500, headless=True)
        original_env.seed(seed + rank) # Doesn't matter here, its deterministic
        ind_env = OneOfToDiscreteWrapper(original_env)
        return ind_env
    return _init


from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList


from wandb.integration.sb3 import WandbCallback
import wandb


if __name__ == "__main__":
    callbacks = []
    run = wandb.init(
    project="PokeWorlds",
    name="charmander_enthusiast",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
    )
    callbacks.append(WandbCallback())

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Instantiate the agent
    model = DQN("MultiInputPolicy", env, verbose=1, gamma=0.999, batch_size=batch_size, n_epochs=num_epochs)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True, callback=CallbackList(callbacks))
    # Save the agent
    model.save("charmander_enthusiast_agent")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = PPO.load("charmander_enthusiast_agent", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Reward: {mean_reward} +/- {std_reward}")

    # Enjoy trained agent
    if render:
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render()