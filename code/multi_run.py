# systemd-inhibit --what=idle --why="RL training" python multi_run.py

import json
import torch
import random
import numpy as np
import os
import gymnasium as gym
import sys

sys.path.append(os.path.abspath("../common"))
import util, graphing

import ddpg
import train_agent

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Overwriting existing videos.*",
    category=UserWarning,
)


def save_single_exp(aloss, closs, train, test, steps, save_path):

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

    # Save summary, and metadata
    util.save_metadata_json(summary, save_path, "summary.json")
    util.save_metadata_pkl(metadata, save_path, "aggregated_metadata.pkl")




def save_exp(config, 
             mean_aloss, std_aloss,
             mean_closs, std_closs,
             mean_train, std_train,
             mean_test, std_test,
             mean_steps, std_steps,
             save_path):

    summary = {
        "actor_loss": {
            "mean_of_means": float(np.mean(mean_aloss)),
            "min_mean": float(np.min(mean_aloss)),
            "max_mean": float(np.max(mean_aloss)),
            "final_mean": float(mean_aloss[-1]),
            "mean_std": float(np.mean(std_aloss))
        },
        "critic_loss": {
            "mean_of_means": float(np.mean(mean_closs)),
            "min_mean": float(np.min(mean_closs)),
            "max_mean": float(np.max(mean_closs)),
            "final_mean": float(mean_closs[-1]),
            "mean_std": float(np.mean(std_closs))
        },
        "train_reward": {
            "mean_of_means": float(np.mean(mean_train)),
            "max_mean": float(np.max(mean_train)),
            "final_mean": float(mean_train[-1]),
            "mean_std": float(np.mean(std_train))
        },
        "test_reward": {
            "mean_of_means": float(np.mean(mean_test)),
            "max_mean": float(np.max(mean_test)),
            "final_mean": float(mean_test[-1]),
            "mean_std": float(np.mean(std_test))
        },
        "mean_steps": float(np.mean(mean_steps)),
        "max_steps": int(np.max(mean_steps)),
        "mean_std_steps": float(np.mean(std_steps))
    }

    # Metadata (full arrays for mean & std)
    metadata = {
        "actor_loss": {"mean": mean_aloss.tolist(), "std": std_aloss.tolist()},
        "critic_loss": {"mean": mean_closs.tolist(), "std": std_closs.tolist()},
        "train_reward": {"mean": mean_train.tolist(), "std": std_train.tolist()},
        "test_reward": {"mean": mean_test.tolist(), "std": std_test.tolist()},
        "steps": {"mean": mean_steps.tolist(), "std": std_steps.tolist()}
    }

    # Save config, summary, and metadata
    util.save_metadata_json(config, save_path, "config.json")
    util.save_metadata_json(summary, save_path, "summary.json")
    util.save_metadata_pkl(metadata, save_path, "aggregated_metadata.pkl")


if __name__ == "__main__":
    config = util.load_config("config.json")
    
    all_train_rewards = []
    all_test_rewards = []
    all_actor_losses = []
    all_critic_losses = []
    all_steps_taken = []
    all_target_qs = []
    all_current_qs = []
    all_targets = []

    main_save_path = None
    if config["save_model"]:
        main_save_path, main_plots_dir = util.setup_experiment_dir("ddpg", config["name"], config["env"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(config["env"])

    print("Training on device:", device)
    print(f"Training on env: {config['env']} | Observation Space: {env.observation_space.shape[0]} | Action Space: {env.action_space.shape[0]} | Max action: {float(env.action_space.high[0])} \n")

    for seed in config["seeds"]:
        print(f"\n Running seed: {seed}\n")
        util.set_seed(seed)

        seed_save_path = None
        if config["save_model"]:
            seed_save_path, seed_plots_dir = util.setup_experiment_dir(main_save_path, f"seed_{seed}")
            video_dir = os.path.join(seed_save_path, "videos")
            os.makedirs(video_dir, exist_ok=True)

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
            seed=seed,
            save_path=seed_save_path
        )

        save_single_exp(alosses, closses, train_rewards, test_rewards, steps_taken, seed_save_path)

        all_train_rewards.append(train_rewards)
        all_test_rewards.append(test_rewards)
        all_actor_losses.append(alosses)
        all_critic_losses.append(closses)
        all_steps_taken.append(steps_taken)
        all_target_qs.append(target_qs)
        all_current_qs.append(current_qs)
        all_targets.append(targets)

        del agent, replay_buffer
        torch.cuda.empty_cache()
        env.close()


    mean_train, std_train = util.aggregate_metric(all_train_rewards)
    mean_test, std_test = util.aggregate_metric(all_test_rewards)
    mean_aloss, std_aloss = util.aggregate_metric(all_actor_losses)
    mean_closs, std_closs = util.aggregate_metric(all_critic_losses)
    mean_steps, std_steps = util.aggregate_metric(all_steps_taken)
    mean_target_q, std_target_q = util.aggregate_metric(all_target_qs)
    mean_current_q, std_current_q = util.aggregate_metric(all_current_qs)
    mean_targets, std_targets = util.aggregate_metric(all_targets)


    if config["save_model"]:
        save_exp(config, 
             mean_aloss, std_aloss,
             mean_closs, std_closs,
             mean_train, std_train,
             mean_test, std_test,
             mean_steps, std_steps,
             main_save_path)

        graphing.plot_mean_std(mean_aloss, std_aloss, title="Actor Loss (mean ± std)",
                       label="Actor Loss", file_path=f"{main_plots_dir}/actor_loss.png")

        graphing.plot_mean_std(mean_closs, std_closs, title="Critic Loss (mean ± std)",
                            label="Critic Loss", file_path=f"{main_plots_dir}/critic_loss.png")

        graphing.plot_mean_std(mean_train, std_train, title="Training Reward (mean ± std)",
                            label="Train Reward", file_path=f"{main_plots_dir}/train_reward.png")

        graphing.plot_mean_std(mean_test, std_test, title="Test Reward (mean ± std)",
                            label="Test Reward", file_path=f"{main_plots_dir}/test_reward.png")

        graphing.plot_mean_std(mean_steps, std_steps, title="Steps per Episode (mean ± std)",
                            label="Steps", file_path=f"{main_plots_dir}/steps_taken.png")

        graphing.plot_mean_std(mean_target_q, std_target_q, title="Target Q (mean ± std)",
                            label="Target Q", file_path=f"{main_plots_dir}/target_q.png")

        graphing.plot_mean_std(mean_current_q, std_current_q, title="Current Q (mean ± std)",
                            label="Current Q", file_path=f"{main_plots_dir}/current_q.png")
        
        graphing.plot_mean_std(mean_targets, std_targets, title="Targets (mean ± std)",
                            label="Targets", file_path=f"{main_plots_dir}/targets.png")
        
        graphing.plot_loss(mean_target_q, loss2=mean_current_q, title='Target vs Current q', label1='Target q', label2='Current q', file_path=f"{main_plots_dir}/q_compared.png")
