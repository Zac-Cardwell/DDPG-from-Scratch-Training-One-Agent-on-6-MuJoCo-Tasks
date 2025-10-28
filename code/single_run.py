# systemd-inhibit --what=idle --why="RL training" python single_run.py

import torch
import numpy as np
import os
import gymnasium as gym
import sys
sys.path.append(os.path.abspath("/mnt/Linux_4TB_HDD/CodeProjects/Machine_Learning/Paper_Recreations/common"))
import util, graphing

import ddpg
import train_agent

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Overwriting existing videos.*",
    category=UserWarning,
)



def save_exp(config, aloss, closs, train, test, steps, save_path):

    summary = {
        "actor_loss": {
            "mean": float(np.mean(aloss)),
            "min": float(np.min(aloss)),
            "max": float(np.max(aloss)),
            "final": float(aloss[-1])
        },
        "critic_loss": {
            "mean": float(np.mean(closs)),
            "min": float(np.min(closs)),
            "max": float(np.max(closs)),
            "final": float(closs[-1])
        },
        "train_reward": {
            "mean": float(np.mean(train)),
            "max": float(np.max(train)),
            "final": float(train[-1])
        },
        "test_reward": {
            "mean": float(np.mean(test)),
            "max": float(np.max(test)),
            "final": float(test[-1])
        },
        "mean_steps": float(np.mean(steps)),
        "max_steps": int(np.max(steps))
    }

    # Metadata (full arrays for mean & std)
    metadata = {
        "actor_loss": aloss,
        "critic_loss": closs,
        "train_reward": train,
        "test_reward": test,
        "steps": steps
    }

    # Save config, summary, and metadata
    util.save_metadata_json(config, save_path, "config.json")
    util.save_metadata_json(summary, save_path, "summary.json")
    util.save_metadata_pkl(metadata, save_path, "aggregated_metadata.pkl")



if __name__ == "__main__":
    config = util.load_config("/mnt/Linux_4TB_HDD/CodeProjects/Machine_Learning/Paper_Recreations/ddpg/config.json")
    util.set_seed(config["seeds"][0])

    save_path = None
    if config["save_model"]:
        save_path, plots_dir = util.setup_experiment_dir("/mnt/Linux_4TB_HDD/CodeProjects/Machine_Learning/Paper_Recreations/ddpg", config["name"], config["env"])
        video_dir = os.path.join(save_path, "videos")
        os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(config["env"])

    print("Training on device:", device)
    print(f"Training on env: {config['env']} | Observation Space: {env.observation_space.shape[0]} | Action Space: {env.action_space.shape[0]} | Max action: {float(env.action_space.high[0])} \n")

    agent = ddpg.DDPG(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dim=config["hidden_dim"],
    max_action=float(env.action_space.high[0]),
    actor_lr=config["actor_lr"],
    critic_lr=config["critic_lr"],
    gamma=config["gamma"],
    tau=config["tau"],
    noise_std=config["expl_noise"],
    noise_decay=config["noise_decay"],
    min_expl_noise=config["min_expl_noise"],
    weight_decay=config["weight_decay"],
    normalize=config["normalize"],
    device=device
    )
    replay_buffer = ddpg.ReplayBuffer(config["buffer_size"], config["batch_size"])

    if config["load_model"]:
        agent.load(config["load_model"])

    alosses, closses, train_rewards, test_rewards, steps_taken, target_qs, current_qs, targets = train_agent.train_agent(
        agent=agent,
        env=env,
        replay_buffer=replay_buffer,
        num_episodes=config["num_episodes"],
        warmup_steps=config["warmup_steps"],
        max_step=config["max_steps_per_episode"],
        test_freq=config["test_freq"],
        test_episodes=config["test_episodes"],
        reward_scale=config["reward_scale"],
        seed=config["seeds"][0],
        save_path=save_path
    )

    if config["save_model"]:
        save_exp(config, alosses, closses, train_rewards, test_rewards, steps_taken, save_path)

        save_file = os.path.join(plots_dir, f"actor_loss.png")
        graphing.plot_loss(alosses, loss2=None, title='Actor Loss',
                   label1='Actor Loss', label2='', file_path=save_file)
        
        save_file = os.path.join(plots_dir, f"q_values.png")
        graphing.plot_loss(target_qs, loss2=current_qs, title='Target vs Current q',
                   label1='Target q', label2='Current q', file_path=save_file)
        
        save_file = os.path.join(plots_dir, f"critic_loss.png")
        graphing.plot_loss(closses, loss2=None, title='Critic Loss',
                   label1='Critic Loss', label2='', file_path=save_file)
        
        save_file = os.path.join(plots_dir, f"training_rewards.png")
        graphing.plot_rewards(train_rewards, rewards2=None, title='Total and Average Reward per Episode',
                      label1='Training Reward', label2='Testing Reward', file_path=save_file)
        
        save_file = os.path.join(plots_dir, f"testing_rewards.png")
        graphing.plot_rewards(test_rewards, rewards2=None, title='Total and Average Reward per Episode',
                      label1='Testing Reward', label2='Testing Reward', file_path=save_file)
        
        save_file = os.path.join(plots_dir, f"steps_taken.png")
        graphing.plot_steps(steps_taken, title='Steps Taken per Episode', label='Steps taken', file_path=save_file)

    env.close()