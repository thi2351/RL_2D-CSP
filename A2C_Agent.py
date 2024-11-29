import torch
from A2C import A2CNetwork
class A2CAgent:
    def __init__(self, input_dim, action_dim, hidden_dim=128, lr=1e-4, gamma=0.99, device='cpu'):
        self.device = device
        self.model = A2CNetwork(input_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state, env):
        # Kiểm tra và chuyển đổi state thành tensor
        state_tensor = (
            state.clone().detach().unsqueeze(0).to(self.device)
            if isinstance(state, torch.Tensor)
            else torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        action_index = torch.multinomial(action_probs, 1).item()
        
        # Map action index to action components using env
        stock_index = action_index % env.num_sheet
        size = [
            torch.randint(1, env.max_w, (1,)).item(),
            torch.randint(1, env.max_h, (1,)).item()
        ]
        position = [
            torch.randint(0, env.max_w, (1,)).item(),
            torch.randint(0, env.max_h, (1,)).item()
        ]
        
        # Trả về dict
        return {
            "stock_index": stock_index,
            "size": size,
            "position": position
        }


    def update(self, trajectory):
        # Trajectory unpacking
        states, actions, rewards, next_states, dones = trajectory

        # Kiểm tra và xử lý định dạng của states
        if isinstance(states[0], torch.Tensor):
            states_tensor = torch.stack(states).to(self.device)
        else:
            raise TypeError(f"Expected states to be a list of tensors, but got {type(states[0])}")

        # Tiếp tục xử lý actions và các giá trị khác như trước
        if isinstance(actions[0], dict):
            processed_actions = [
                [action["stock_index"]] + action["size"] + action["position"]
                for action in actions
            ]
            actions_tensor = torch.tensor(processed_actions, dtype=torch.float32).to(self.device)
        elif isinstance(actions[0], torch.Tensor):
            actions_tensor = actions
        else:
            raise TypeError(f"Expected actions to be a list of dict or tensor, but got {type(actions[0])}")

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.stack(next_states).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Tính toán hành động và giá trị
        action_probs, values = self.model(states_tensor)
        _, next_values = self.model(next_states_tensor)

        # Tính toán advantages
        advantages = rewards_tensor + self.gamma * next_values.squeeze() * (1 - dones_tensor) - values.squeeze()
        actor_loss = -torch.log(action_probs[range(len(actions_tensor)), actions_tensor[:, 0].long()]) * advantages.detach()
        critic_loss = advantages.pow(2)

        # Tổng hợp loss
        loss = actor_loss.mean() + critic_loss.mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

