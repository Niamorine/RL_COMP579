
import gymnasium as gym
from ale_py import ALEInterface
from stable_baselines3 import PPO, DQN, A2C

from train_eval import make_env_with_framestack

def demo(env_name: str, model_name: str, model_path: str, framestack: int):
    
    model_class = PPO if model_name == "PPO" else DQN if model_name == "DQN" else A2C
    
    vec_env = make_env_with_framestack(env_name, n_envs=1, seed=42, framestack=framestack)
    model = model_class.load(model_path, env=vec_env)

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        

MODEL = "DQN"
ENV_NAME = "Breakout-v4"
MODEL_PATH = "results_10million/DQN/framestack_8/seed_2042/model"
FRAMESTACK = 8

if __name__ == "__main__":
    demo(ENV_NAME, MODEL, MODEL_PATH, FRAMESTACK)