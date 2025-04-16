import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_train_episode_rewards(all_results_dict: dict, save_fig: bool):
    
    episode_rewards_dict = {}
    
    # Getting required data
    for model_name, framestacks in all_results_dict.items():
        for framestack, seeds in framestacks.items():
            key = f"{model_name}_framestack_{framestack}"
            episode_rewards_dict[key] = []
            for seed, results in seeds.items():
                episode_rewards = results["train_episode_rewards"]
                episode_rewards_dict[key].append(episode_rewards)
    
    mean_std_dict = {}

    for key, rewards_list in episode_rewards_dict.items():
        # rewards_array = np.array(rewards_list)
        min_len = min(len(r) for r in rewards_list)  # Ensure same length
        rewards_array = np.array([r[:min_len] for r in rewards_list])  # Trim to shortest
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        mean_std_dict[key] = {"mean": mean_rewards, "std": std_rewards}
    
    
    plt.figure(figsize=FIG_SIZE)
    for key, data in mean_std_dict.items():
        episodes = np.arange(len(data["mean"]))
        plt.plot(episodes, data["mean"], label=key)
        plt.fill_between(episodes, data["mean"] - data["std"], data["mean"] + data["std"], alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Training Reward")
    plt.title("Training Reward per Episode (Mean ± Std Dev over Seeds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{RESULTS_ROOT}/training_episode_rewards.png")
    plt.show()


def plot_sample_efficiency(all_results_dict: dict, save_fig: bool):
    sample_efficiency_dict = {}

    # Getting required data
    for model_name, framestacks in all_results_dict.items():
        for framestack, seeds in framestacks.items():
            key = f"{model_name}_framestack_{framestack}"
            sample_efficiency_dict[key] = {
                "timesteps": None,
                "rewards": []
            }
            for seed, results in seeds.items():
                se_data = results["train_sample_efficiency_data"]
                timesteps = se_data["timesteps"]
                rewards = se_data["mean_rewards"]
                
                # Store timesteps once (assumes all runs use same steps)
                if sample_efficiency_dict[key]["timesteps"] is None:
                    sample_efficiency_dict[key]["timesteps"] = timesteps
                
                sample_efficiency_dict[key]["rewards"].append(rewards)

    mean_std_dict = {}
    for key, data in sample_efficiency_dict.items():
        rewards_array = np.array(data["rewards"])  # shape (num_seeds, num_eval_points)
        min_len = min(len(r) for r in rewards_array)
        rewards_array = np.array([r[:min_len] for r in rewards_array])
        timesteps = data["timesteps"][:min_len]

        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        mean_std_dict[key] = {
            "timesteps": timesteps,
            "mean": mean_rewards,
            "std": std_rewards
        }


    plt.figure(figsize=FIG_SIZE)
    for key, data in mean_std_dict.items():
        timesteps = data["timesteps"]
        plt.plot(timesteps, data["mean"], label=key)
        plt.fill_between(timesteps,
                         np.array(data["mean"]) - np.array(data["std"]),
                         np.array(data["mean"]) + np.array(data["std"]),
                         alpha=0.2)
    plt.xlabel("Environment Steps")
    plt.ylabel("Evaluation Reward")
    plt.title("Sample Efficiency (Mean ± Std Dev over Seeds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{RESULTS_ROOT}/training_sample_efficiency.png")
    plt.show()


def plot_evaluation_summary(all_results_dict: dict, save_fig: bool):
    eval_means = []
    eval_stds = []
    labels = []

    # Getting required data
    for model_name, framestacks in all_results_dict.items():
        for framestack, seeds in framestacks.items():
            rewards = []
            for seed, results in seeds.items():
                if "eval_mean_reward" in results:
                    rewards.append(results["eval_mean_reward"])
            if rewards:
                mean = np.mean(rewards)
                std = np.std(rewards)
                eval_means.append(mean)
                eval_stds.append(std)
                labels.append(f"{model_name}_fs{framestack}")


    x = np.arange(len(labels))
    plt.figure(figsize=FIG_SIZE)
    plt.bar(x, eval_means, yerr=eval_stds, capsize=5, alpha=0.8)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Evaluation Performance (Mean ± Std Dev over Seeds)")
    plt.grid(axis="y")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{RESULTS_ROOT}/evaluation_rewards_summary.png")
    plt.show()


def load_all_results(results_root: str = "results") -> dict:
    """
    Loads all results into a single dict respecting directory structure
    """
    all_results_dict = {}

    for model_name in os.listdir(results_root):
        model_path = os.path.join(results_root, model_name)
        if not os.path.isdir(model_path):
            continue
        all_results_dict[model_name] = {}

        for framestack_dir in os.listdir(model_path):
            if not framestack_dir.startswith("framestack_"):
                continue
            framestack = int(framestack_dir.split("_")[1])
            framestack_path = os.path.join(model_path, framestack_dir)
            all_results_dict[model_name][framestack] = {}

            for seed_dir in os.listdir(framestack_path):
                if not seed_dir.startswith("seed_"):
                    continue
                seed = int(seed_dir.split("_")[1])
                result_file = os.path.join(framestack_path, seed_dir, "results.json")
                if not os.path.exists(result_file):
                    continue
                with open(result_file, "r") as f:
                    results = json.load(f)
                    all_results_dict[model_name][framestack][seed] = results["results"]

    return all_results_dict


RESULTS_ROOT = "results_test"
FIG_SIZE = (16, 8)
SAVE_FIG = True

if __name__ == "__main__":
    all_results_dict = load_all_results(RESULTS_ROOT)

    
    plot_train_episode_rewards(all_results_dict, SAVE_FIG)
    
    plot_sample_efficiency(all_results_dict, SAVE_FIG)
    
    plot_evaluation_summary(all_results_dict, SAVE_FIG)
    
    