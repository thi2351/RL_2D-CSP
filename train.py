import numpy as np
import torch
from Env import CuttingStockEnv
from A2C_Agent import A2CAgent

def main():
    # Thiết lập thiết bị (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Khởi tạo môi trường
    env = CuttingStockEnv()
    
    # Khởi tạo agent với action_dim cố định
    action_dim = env.action_space["position"].shape[0] + env.action_space["size"].shape[0] + 1
    hidden_dim = 64
    agent = None  # Sẽ khởi tạo agent sau khi tính được input_dim
    
    episodes = 100
    batch_size = 2  # Đặt batch size là 2
    
    for episode in range(episodes):
        # Reset môi trường và lấy trạng thái ban đầu
        state = env.reset()

        # Tính toán `input_dim` động dựa trên trạng thái thực tế
        sheet_dim = state["sheet"].shape
        rect_count = len(state["rect"])  # Số lượng hình chữ nhật
        rect_attr_dim = 3  # Mỗi hình chữ nhật có 2 kích thước + 1 số lượng còn lại
        input_dim = sheet_dim[0] * sheet_dim[1] * sheet_dim[2] + rect_count * rect_attr_dim

        # Khởi tạo agent nếu chưa có
        if agent is None:
            agent = A2CAgent(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim, device=device)

        done = False
        trajectory = []  # Trajectory tạm thời
        total_reward = 0
        
        while not done:
            # Chuyển trạng thái thành mảng 1 chiều
            flat_state = torch.tuple(torch.cat([
                torch.tensor(state["sheet"].flatten(), dtype=torch.float32).to(device),
                torch.cat([
                    torch.cat((torch.tensor(rect["size"], dtype=torch.float32).to(device), 
                            torch.tensor([rect["remaining"]], dtype=torch.float32).to(device)))
                    for rect in state["rect"]
                ])
            ]))

            # Chọn hành động
            action = agent.select_action(flat_state, env)

            # Thực hiện bước trong môi trường
            next_state, reward, done, _ = env.step(action)

            # Chuyển trạng thái kế tiếp thành mảng 1 chiều
            flat_next_state = torch.cat([
                torch.tensor(next_state["sheet"].flatten(), dtype=torch.float32).to(device),
                torch.cat([
                    torch.cat((torch.tensor(rect["size"], dtype=torch.float32).to(device), 
                               torch.tensor([rect["remaining"]], dtype=torch.float32).to(device)))
                    for rect in next_state["rect"]
                ])
            ])

            # Thêm vào trajectory
            trajectory.append((flat_state, action, reward, flat_next_state, done))

            # Nếu đạt đủ batch size, cập nhật agent
            if len(trajectory) >= batch_size:
                states, actions, rewards, next_states, dones = zip(*trajectory)
                
                # Xử lý actions: Trích xuất từ dict
                processed_actions = [
                    [a["stock_index"]] + a["size"] + a["position"] for a in actions
                ]
                actions_tensor = torch.tensor(processed_actions, dtype=torch.float32).to(device)

                agent.update((
                    torch.stack(states).to(device),
                    actions_tensor,
                    torch.tensor(rewards, dtype=torch.float32).to(device),
                    torch.stack(next_states).to(device),
                    torch.tensor(dones, dtype=torch.float32).to(device)
                ))
                trajectory = []  # Reset trajectory

            state = next_state
            total_reward += reward

        # Xử lý phần còn lại của trajectory (nếu có)
        if len(trajectory) > 0:
            states, actions, rewards, next_states, dones = zip(*trajectory)
            processed_actions = [
                [a["stock_index"]] + a["size"] + a["position"] for a in actions
            ]
            actions_tensor = torch.tensor(processed_actions, dtype=torch.float32).to(device)

            agent.update((
                torch.stack(states).to(device),
                actions_tensor,
                torch.tensor(rewards, dtype=torch.float32).to(device),
                torch.stack(next_states).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device)
            ))

        # Log tiến độ
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Hiển thị trạng thái mỗi 50 tập
        if (episode + 1) % 50 == 0:
            env.render()

if __name__ == '__main__':
    main()
