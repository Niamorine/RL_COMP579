
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMonitorCallback(BaseCallback):
    """
    Logs episode rewards during training and calculates
    mean/std reward over intervals for sample efficiency analysis.
    """
    def __init__(self, num_envs: int, log_freq: int, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Data for Individual Episodes
        self.episode_rewards = []
        self.episode_end_timesteps = []
        
        # Rewards in the current episode
        self.current_rewards = np.zeros(num_envs)
        
        # Data for Sample Efficiency (Mean Reward over Interval)
        self.sample_efficiency_data = {'timesteps': [], 'mean_rewards': [], 'std_rewards': []}
        self.rewards_since_last_log = []
        self.last_log_timestep = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"]

        # Check for finished episodes in any parallel environment
        dones = self.locals["dones"]
        current_total_timesteps = self.num_timesteps # Total timesteps seen so far

        for i, done in enumerate(dones):
            if done:
                # Episode finished in environment i
                finished_episode_reward = self.current_rewards[i]

                # Log individual episode data
                self.episode_rewards.append(finished_episode_reward)
                self.episode_end_timesteps.append(current_total_timesteps)

                # Add to rewards collected since last sample efficiency log
                self.rewards_since_last_log.append(finished_episode_reward)

                # Reset the reward accumulator for this environment
                self.current_rewards[i] = 0.0

        # Check if it's time to log sample efficiency data
        if current_total_timesteps - self.last_log_timestep >= self.log_freq:
            if len(self.rewards_since_last_log) > 0:
                # Calculate stats over the episodes finished in this interval
                mean_interval_reward = np.mean(self.rewards_since_last_log)
                std_interval_reward = np.std(self.rewards_since_last_log)

                # Store for sample efficiency plot
                self.sample_efficiency_data['timesteps'].append(current_total_timesteps)
                self.sample_efficiency_data['mean_rewards'].append(mean_interval_reward)
                self.sample_efficiency_data['std_rewards'].append(std_interval_reward)

                # Reset for the next interval
                self.rewards_since_last_log = []
                self.last_log_timestep = current_total_timesteps

        return True