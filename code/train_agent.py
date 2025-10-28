import numpy as np
import time
import os
import gymnasium as gym
import torch


def test_policy(agent, env, test_episodes=10, max_step=200, seed=42, deterministic=True):
    episode_rewards = []
    episode_actions = []
    episode_states = []
    agent.eval()
    for i in range(test_episodes):
        base_seed = seed + 10_000
        state, _ = env.reset(seed=base_seed + i)
        agent.reset_noise()
        step = 0
        episode_reward = 0
        actions = []
        states = []

        while step < max_step:
            action = agent.select_action(state, deterministic=deterministic)
            actions.append(action)
            states.append(state)
            next_state, reward, done, truncated, info = env.step(action)
            done_flag = done or truncated
            episode_reward += reward
            state = next_state
            step += 1
            if done_flag:
                break

        episode_rewards.append(episode_reward)
        episode_actions.append(np.mean(actions))
        episode_states.append(np.mean(states, axis=0))
    
    agent.train()
    return np.mean(episode_rewards), np.mean(episode_actions), np.mean(episode_states)


def record_run(agent, env_name, save_dir, max_steps=500, name_prefix="best_model"):
    os.environ["MUJOCO_GL"] = "egl"  

    eval_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=save_dir,
        episode_trigger=lambda ep: True,
        name_prefix=name_prefix
    )

    state, _ = eval_env.reset()
    agent.eval()
    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
        action = agent.select_action(state_tensor, deterministic=True)
        state, reward, done, truncated, info = eval_env.step(action)
        if done or truncated:
            break
    eval_env.close()
    agent.train()


def train_agent(agent, env, replay_buffer, num_episodes=1000, warmup_steps=1000, max_step = 500, test_freq=100, test_episodes=10, reward_scale=1,seed=42, save_path=None):
    test_rewards = []
    test_actions = []
    test_states = []

    train_rewards = []
    train_actions = []
    train_states = []

    alosses = []
    closses = []
    target_qs = []
    current_qs = []
    time_elapsed = []
    steps_taken = []
    targets = []

    best_test_avg = -np.inf
    best_episode = 0
    total_steps = 0

    for episode in range(num_episodes):

        state, _ = env.reset(seed=seed + episode)
        agent.reset_noise()
        agent.train()

        step = 0
        episode_reward = 0
        episode_aloss = 0
        episode_closs = 0
        episode_target_q = 0
        episode_current_q = 0
        episode_target = 0

        episode_actions = []
        episode_states = []
        
        start = time.time()
        while step < max_step:
            
            action = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)
            done_flag = done or truncated

            scaled_reward = reward * reward_scale
            replay_buffer.add([state, action, scaled_reward, next_state, done_flag])

            state = next_state
            episode_reward += reward
            episode_actions.append(action)
            episode_states.append(state)

            if replay_buffer.size() > warmup_steps and replay_buffer.size() > replay_buffer.batch_size:
                aloss, closs, target_q, current_q, target = agent.update(replay_buffer)
                agent.exp_decay_noise(total_steps)
                episode_aloss += aloss
                episode_closs += closs
                episode_target_q += target_q
                episode_current_q += current_q
                episode_target += target
        
            step += 1
            total_steps += 1

            if done_flag:
                break

        end = time.time()
        #agent.exp_decay_noise(episode)

        alosses.append(episode_aloss/ step)
        closses.append(episode_closs/ step)
        target_qs.append(episode_target_q / step)
        current_qs.append(episode_current_q / step)
        train_rewards.append(episode_reward)
        train_actions.append(np.mean(episode_actions))
        train_states .append(np.mean(episode_states, axis=0))
        time_elapsed.append((end - start) / 60)
        steps_taken.append(step)
        targets.append(episode_target/ step)

        if episode % test_freq == 0:
            test_reward, test_action, test_state = test_policy(agent, env, test_episodes, max_step, seed, deterministic=True)
            test_rewards.append(test_reward)
            test_actions.append(test_action)
            test_states.append(test_state)

            if test_rewards[-1] > best_test_avg:
                best_test_avg = test_rewards[-1]
                best_episode = episode

                if save_path:
                    save_file = os.path.join(save_path, f"best_model")
                    agent.save(save_file)
                    record_run(agent, env.spec.id, save_path, max_steps=max_step, name_prefix=f"best_model_seed_{seed}")

        if episode % 500 == 0 or episode == num_episodes-1:  
            avg_train_reward = np.mean(train_rewards[-500:]) if len(train_rewards) >= 500 else train_rewards[-1]
            avg_test_reward = np.mean(test_rewards[-(500//test_freq):]) if len(test_rewards) >= (500//test_freq) else test_rewards[-1]
            avg_steps = np.mean(steps_taken[-500:]) if len(steps_taken) >= 500 else steps_taken[-1]
            avg_time = np.mean(time_elapsed[-500:]) if len(time_elapsed) >= 500 else time_elapsed[-1]
            avg_aloss = np.mean(alosses[-500:]) if len(alosses) >= 500 else alosses[-1]
            avg_closs = np.mean(closses[-500:]) if len(closses) >= 500 else closses[-1]
            avg_target_q = np.mean(target_qs[-500:]) if len(target_qs) >= 500 else target_qs[-1]
            avg_current_q = np.mean(current_qs[-500:]) if len(current_qs) >= 500 else current_qs[-1]
            avg_target = np.mean(targets[-500:]) if len(targets) >= 500 else targets[-1]
            avg_test_actions = float(np.mean(test_actions[-500:])) if len(test_actions) >= 500 else float(np.mean(test_actions[-1]))
            avg_train_actions = float(np.mean(train_actions[-500:])) if len(train_actions) >= 500 else float(np.mean(train_actions[-1]))
            avg_test_states = float(np.mean(test_states[-500:])) if len(test_states) >= 500 else float(np.mean(test_states[-1]))
            avg_train_states = float(np.mean(train_states[-500:])) if len(train_states) >= 500 else float(np.mean(train_states[-1]))


            print(f"Episode {episode} | Avg Train Reward (last 500): {avg_train_reward:.2f} | "
                f"Avg Test Reward (last 500): {avg_test_reward} | This Train Reward: {train_rewards[-1]}\n"
                f"Avg Actor Loss: {avg_aloss:.2f} | Avg Critic Loss: {avg_closs:.2f} | "
                f"Avg Steps Taken: {avg_steps:.2f} | Avg Time per Ep: {avg_time:.2f}m\n"
                f"Avg Target q: {avg_target_q:.2f} | Avg Current q: {avg_current_q:.2f} | Avg Target: {avg_target:.2f}\n"
                f"Avg Train Actions: {avg_train_actions:.2f} | Avg Test Actions: {avg_test_actions:.2f} | Avg Train States: {avg_train_states:.2f} | Avg Test States: {avg_test_states:.2f}\n"
                )
            
            video_dir = os.path.join(save_path, "videos")
            record_run(agent, env.spec.id, video_dir, max_steps=max_step, name_prefix=f"agent_ep_{episode}_seed_{seed}")
            
    print(f"Best test rewards {best_test_avg} achieved at episode: {best_episode}")

    return alosses, closses, train_rewards, test_rewards, steps_taken, target_qs, current_qs, targets