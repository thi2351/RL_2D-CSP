import torch
import torch.nn as nn
import torch.optim as optim
from Env import CuttingStockEnv
class ActorNetWork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorNetWork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.model(x)


class CriticNetWork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNetWork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class A2CAgent:
    def __init__(self, device, input_dim, action_dim, hidden_dim, lr, gamma, seed):
        self.device = device
        self.actor = ActorNetWork(input_dim, hidden_dim, action_dim).to(self.device)
        self.critic = CriticNetWork(input_dim, hidden_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        torch.manual_seed(seed)

    def select_action(self, state, env):
        with torch.no_grad():
            action_logits = self.actor(state.unsqueeze(0))  # Add batch dimension
            action_probs = torch.softmax(action_logits, dim=-1).squeeze(0)
            action = torch.argmax(action_probs).item()


        rectangle_size = env._rectangles[0]["size"]  # Select first rectangle for simplicity
        stock_index = env.choose_best_sheet(rectangle_size)  # Use tensor directly


        if stock_index == -1:  # If no sheet can fit, open a new one
            stock_index = torch.randint(0, env.num_sheet, (1,)).item()

        position = torch.tensor([
            torch.randint(0, env.max_w - rectangle_size[0].item() + 1, (1,)).item(),
            torch.randint(0, env.max_h - rectangle_size[1].item() + 1, (1,)).item()
        ], dtype=torch.int32)


        return {
            "stock_index": stock_index,
            "size": rectangle_size,
            "position": position
        }

    def update(self, trajectory):
        states, actions, rewards, next_states, dones = trajectory
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.stack(next_states).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute values
        values = self.critic(states_tensor).squeeze()
        next_values = self.critic(next_states_tensor).squeeze()
        targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)

        # Critic loss
        critic_loss = nn.MSELoss()(values, targets)

        # Actor loss
        action_logits = self.actor(states_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs[range(len(actions_tensor)), actions_tensor])
        advantages = (targets - values).detach()
        actor_loss = -(log_probs * advantages).mean()

        # Update networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

def test_a2c_agent():
    # Thiết lập môi trường
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    env = CuttingStockEnv(seed=seed, num_sheet=10, max_w=100, max_h=100, min_w=50, min_h=50, max_rec_type=5, max_rect_per_type=5, device=device)
    
    # Khởi tạo agent
    input_dim = env.num_sheet * env.max_w * env.max_h + env.max_rec_type * 3
    action_dim = env.num_sheet
    hidden_dim = 64
    lr = 1e-3
    gamma = 0.99
    agent = A2CAgent(device, input_dim, action_dim, hidden_dim, lr, gamma, seed)

    # Kiểm tra chọn hành động
    state = env.reset()
    flat_state = torch.cat([
        state["sheet"].flatten().float().to(device),
        torch.cat([
            torch.cat((rect["size"].clone().detach().to(torch.float32).to(device), 
                       torch.tensor([rect["remaining"]], dtype=torch.float32).to(device)))
            for rect in state["rect"]
        ])
    ])
    
    action = agent.select_action(flat_state, env)
    print("Action selected:", action)

    # Xác minh hành động hợp lệ
    assert "stock_index" in action, "Action must contain 'stock_index'."
    assert "size" in action, "Action must contain 'size'."
    assert "position" in action, "Action must contain 'position'."

    # Tạo một trajectory đơn giản để kiểm tra cập nhật
    trajectory = []
    for _ in range(5):  # Giả lập 5 bước
        next_state, reward, done, _ = env.step(action)
        flat_next_state = torch.cat([
            next_state["sheet"].flatten().float().to(device),
            torch.cat([
                torch.cat((rect["size"].clone().detach().to(torch.float32).to(device), 
                           torch.tensor([rect["remaining"]], dtype=torch.float32).to(device)))
                for rect in next_state["rect"]
            ])
        ])

        trajectory.append((flat_state, action["stock_index"], reward, flat_next_state, done))
        if done:
            break
        state = next_state
        flat_state = flat_next_state
        action = agent.select_action(flat_state, env)

    # Kiểm tra cập nhật
    agent.update(zip(*trajectory))
    print("Update complete. No errors encountered.")

if __name__ == "__main__":
    test_a2c_agent()
