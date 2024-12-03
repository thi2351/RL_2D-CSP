import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from Env import CuttingStockEnv
from A2C_Agent import A2CAgent

def train_cutting_stock():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment and agent parameters
    seed = 42
    env = CuttingStockEnv(seed=seed, num_sheet=10, max_w=100, max_h=100, min_w=50, min_h=50, max_rec_type=5, max_rect_per_type=5, device=device)
    hidden_dim = 64
    action_dim = env.num_sheet
    input_dim = env.num_sheet * env.max_w * env.max_h + env.max_rec_type * 3
    agent = A2CAgent(device, input_dim, action_dim, hidden_dim, lr=1e-3, gamma=0.99, seed=seed)

    # Training parameters
    episodes = 100  # Increase for better results
    batch_size = 16
    max_steps_per_episode = 500  # Limit to avoid infinite loops

    rewards_per_episode = []

    with tqdm(total=episodes, desc="Training Episodes", unit="episode") as pbar:
        for episode in range(episodes):
            state = env.reset()
            trajectory = []
            total_reward = 0
            done = False
            steps = 0

            with tqdm(total=max_steps_per_episode, desc=f"Episode {episode + 1}", unit="step", leave=False) as inner_pbar:
                while not done and steps < max_steps_per_episode:
                    # Flatten state
                    flat_state = torch.cat([
                        state["sheet"].flatten().float().to(device),
                        torch.cat([
                            torch.cat((rect["size"].float().to(device), torch.tensor([rect["remaining"]], dtype=torch.float32).to(device)))
                            for rect in state["rect"]
                        ])
                    ])

                    # Select action
                    action = agent.select_action(flat_state, env)

                    # Perform action in the environment
                    next_state, reward, done, _ = env.step(action)

                    # Flatten next state
                    flat_next_state = torch.cat([
                        next_state["sheet"].flatten().float().to(device),
                        torch.cat([
                            torch.cat((rect["size"].float().to(device), torch.tensor([rect["remaining"]], dtype=torch.float32).to(device)))
                            for rect in next_state["rect"]
                        ])
                    ])

                    # Store transition in trajectory
                    trajectory.append((flat_state, action["stock_index"], reward, flat_next_state, done))
                    total_reward += reward
                    steps += 1

                    # Update agent if trajectory reaches batch size
                    if len(trajectory) >= batch_size:
                        agent.update(zip(*trajectory))
                        trajectory = []

                    state = next_state

                    inner_pbar.update(1)

            # Render the environment after each episode
            print(f"Rendering results for Episode {episode + 1}")
            env.render(mode="human")

            # Update remaining transitions after the episode ends
            if len(trajectory) > 0:
                agent.update(zip(*trajectory))

            rewards_per_episode.append(total_reward)

            pbar.update(1)
            pbar.set_postfix({"Episode": episode + 1, "Total Reward": total_reward})

    return rewards_per_episode

def plot_training_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    rewards = train_cutting_stock()
    plot_training_rewards(rewards)
