from time import time
import os
import json
from typing import Type

import numpy as np
import gymnasium as gym
from ale_py import ALEInterface
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.evaluation import evaluate_policy

from callback import TrainingMonitorCallback
from custom_logger import setup_logging

import logging

setup_logging(log_dir="logs", filename_prefix="train_eval_log")

def timed(func):
    def wrapper(*args, **kwargs):
        t_begin = time()
        result = func(*args, **kwargs)
        t_total = time() - t_begin
        hours, rem = divmod(t_total, 3600)
        minutes, seconds = divmod(rem, 60)
        logging.info(f"Total running time: {int(hours)}:{int(minutes)}:{int(seconds)}")
        return result
    return wrapper

# In case we use the RAM version of the game, apply some wrappers
def vec_env_wrapper(env: gym.Env) -> gym.Env:
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    return env

def make_env_with_framestack(env_name: str, n_envs: int, seed: int, framestack: int):
    if env_name.find("-ram-") == -1: # If we use the game's screen then use atari env
        vec_env = make_atari_env(
            env_name,
            n_envs=n_envs,
            seed=seed,
            env_kwargs={},
            wrapper_kwargs={"terminal_on_life_loss": True, "frame_skip": 1} # frame_skip=1 or it messes with our episode count
        )
    else:
        vec_env = make_vec_env(ENV_NAME, n_envs=n_envs, seed=seed, wrapper_class=vec_env_wrapper) # Fire reset and episodic life
    vec_env = VecFrameStack(vec_env, n_stack=framestack)
    return vec_env


def train_model(
    env_name: str,
    model_name: str,
    framestack: int,
    num_envs: int,
    total_timesteps: int,
    seed: int,
    policy: str,
    log_freq: int,
    verbose: int = 1,
    progress_bar: bool = True,
    device: str = "auto"
):
    """
    Trains the model and returns the model with results
    """
    vec_env = make_env_with_framestack(env_name, n_envs=num_envs, seed=seed, framestack=framestack)

    if model_name == "PPO":
        model = PPO(
            policy,
            vec_env,
            verbose=verbose,
            seed=seed,
            device=device
        )
    elif model_name == "DQN":
        model = DQN(
            policy,
            vec_env,
            verbose=verbose,
            seed=seed,
            device=device,
            buffer_size=DQN_BUFFER_SIZE
        )
    elif model_name == "A2C":
        model = A2C(
            policy,
            vec_env,
            verbose=verbose,
            seed=seed,
            device=device
        )
    else:
        raise ValueError(f"model_name must be be one of {MODEL_NAMES}, got {model_name}")

    callback = TrainingMonitorCallback(num_envs, log_freq) # Callback to monitor training progress
    t_begin = time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=progress_bar)
    t_training = time() - t_begin
        
    episode_rewards = callback.episode_rewards
    sample_efficiency_data = callback.sample_efficiency_data
    
    result_dict = {
        "params": {
            "model_name": model_name,
            "env_name": env_name,
            "num_envs": num_envs,
            "framestack": framestack,
            "total_timesteps": total_timesteps,
            "seed": seed,
            "policy": policy,
            "log_freq": log_freq
        },
        "results": {
            "t_training": t_training,
            "train_episode_rewards": episode_rewards,
            "train_sample_efficiency_data": sample_efficiency_data
        }
    }
    return model, result_dict

def eval_model(
    env_name: str,
    model_class: Type[PPO | DQN | A2C],
    model_path: str,
    framestack: int,
    eval_episodes: int,
    seed: int
):
    """
    Evaluates the model and returns the results
    """
    # Only 1 env for evaluation
    eval_env = make_env_with_framestack(env_name, n_envs=1, seed=seed+1000, framestack=framestack)
    model = model_class.load(model_path, env=eval_env) # Needs to reload the env because num_envs has changed
    
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        return_episode_rewards=True
    )
        
    eval_results = {
        "eval_rewards": episode_rewards,
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_episode_lengths": episode_lengths,
        "eval_mean_episode_lengths": float(np.mean(episode_lengths)),
        "eval_std_episode_lengths": float(np.std(episode_lengths)),
    }
    return eval_results


@timed
def main():
    total_configs = len(MODEL_NAMES) * len(FRAMESTACKS) * len(SEEDS)
    total_done = 0
    logging.info(f"Configurations to run: {total_configs}")
    
    for model_name in MODEL_NAMES:
        for framestack in FRAMESTACKS:
            for seed in SEEDS:
                
                model_class = PPO if model_name == "PPO" else DQN if model_name == "DQN" else A2C
                
                prefix = f"[{total_done + 1}/{total_configs}]"
                logging.info(f"{prefix} {model_name=} {framestack=} {seed=}")
                
                logging.info(f"{prefix} Training...")
                model, results = train_model(
                    ENV_NAME,
                    model_name,
                    framestack,
                    NUM_ENVS,
                    TOTAL_TRAIN_TIMESTEPS,
                    seed,
                    POLICY,
                    LOG_FREQ,
                    verbose=0,
                    progress_bar=True,
                    device="auto"
                )
                
                save_dir = f"{SAVE_DIR}/{model_name}/framestack_{framestack}/seed_{seed}"
                os.makedirs(save_dir, exist_ok=True)
                
                model_path = f"{save_dir}/model"
                model.save(model_path)
                
                logging.info(f"{prefix} Evaluating...")
                eval_results = eval_model(
                    ENV_NAME,
                    model_class,
                    model_path,
                    framestack,
                    TOTAL_EVAL_EPISODES,
                    seed
                )
                
                results["results"]["eval_rewards"] = eval_results["eval_rewards"]
                results["results"]["eval_mean_reward"] = eval_results["eval_mean_reward"]
                results["results"]["eval_std_reward"] = eval_results["eval_std_reward"]
                results["results"]["eval_episode_lengths"] = eval_results["eval_episode_lengths"]
                results["results"]["eval_mean_episode_lengths"] = eval_results["eval_mean_episode_lengths"]
                results["results"]["eval_std_episode_lengths"] = eval_results["eval_std_episode_lengths"]
                
                with open(f"{save_dir}/results.json", "w") as f:
                    json.dump(results, f, indent=4)
                logging.info(f"{prefix} Saved results to {save_dir}")
                
                total_done += 1        


NUM_ENVS = 4
ENV_NAME = "Breakout-v4"
POLICY = "CnnPolicy" # MLP for Ram, else CNN
TOTAL_TRAIN_TIMESTEPS = 1_000_000
TOTAL_EVAL_EPISODES = 250
LOG_FREQ = 10_000
SAVE_DIR = "results"
DQN_BUFFER_SIZE = 50_000 # Reduced from 1_000_000 to avoid out of memory errors

# Combinations
MODEL_NAMES = ["PPO", "DQN", "A2C"]
FRAMESTACKS = [1, 2, 4, 8]
SEEDS = [2042, 2091, 2112, 2205, 2288, 2296, 2473, 2654, 2761, 2987] # For the eval, seed+1000 will be used


# Test params for "quick" run (few hours)
# TOTAL_TRAIN_TIMESTEPS = 40_000
# TOTAL_EVAL_EPISODES = 100
# LOG_FREQ = 2_000
# MODEL_NAMES = ["PPO", "DQN", "A2C"]
# FRAMESTACKS = [1, 2, 4]


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e
