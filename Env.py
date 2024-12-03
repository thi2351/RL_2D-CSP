import torch
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
class CuttingStockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, seed, num_sheet=100, max_w=100, max_h=100, min_w=50, min_h=50, max_rec_type=20, max_rect_per_type=25, device="cpu"):
        super(CuttingStockEnv, self).__init__()
        self.min_w, self.min_h, self.max_w, self.max_h = min_w, min_h, max_w, max_h
        self.num_sheet, self.max_rec_type, self.max_rect_per_type = num_sheet, max_rec_type, max_rect_per_type
        self.device = torch.device(device)
        self.seed = seed

        # Observation and Action Spaces
        self.observation_space = spaces.Dict({
            "sheet": spaces.Box(-2, max_rec_type, shape=(num_sheet, max_w, max_h), dtype=int),
            "rect": spaces.Tuple([
                spaces.Dict({
                    "size": spaces.MultiDiscrete([max_w, max_h]),
                    "remaining": spaces.Discrete(max_rect_per_type)
                }) for _ in range(max_rec_type)
            ])
        })
        self.action_space = spaces.Dict({
            "stock_index": spaces.Discrete(num_sheet),
            "size": spaces.Box(low=np.array([1, 1], dtype=np.int32), high=np.array([max_w, max_h], dtype=np.int32), dtype=np.int32),
            "position": spaces.Box(low=np.array([0, 0], dtype=np.int32), high=np.array([max_w - 1, max_h - 1], dtype=np.int32), dtype=np.int32),
        })


        self._stocks = []
        self._rectangles = []
        self.reset()

    def reset(self):
        torch.manual_seed(self.seed)  # Sử dụng self.seed để đảm bảo tính ngẫu nhiên cố định
        
        self._stocks = [torch.full((self.max_w, self.max_h), fill_value=-2, device=self.device, dtype=torch.int32) for _ in range(self.num_sheet)]
        self._rectangles = [
            {"size": torch.randint(self.min_w, self.max_w + 1, (2,), device=self.device), 
             "remaining": torch.randint(1, self.max_rect_per_type + 1, (1,), device=self.device).item()}
            for _ in range(self.max_rec_type)
        ]
        return self._get_observation()

    def _get_observation(self):
        return {"sheet": torch.stack(self._stocks), "rect": self._rectangles}

    def step(self, action):
        stock_idx, rect_size, position = action["stock_index"], action["size"], action["position"]
        if self._can_place_(stock_idx, rect_size, position):
            self._place_rectangle(stock_idx, rect_size, position)
            reward = self._calculate_reward()
            done = all(rect["remaining"] == 0 for rect in self._rectangles)
            return self._get_observation(), reward, done, {}
        else:
            return self._get_observation(), -1, False, {}

    def _can_place_(self, stock_idx, rect, position):
        x, y, w, h = position[0], position[1], rect[0], rect[1]
        return (
            x + w <= self.max_w and 
            y + h <= self.max_h and 
            torch.all(self._stocks[stock_idx][x:x + w, y:y + h] == -2)
        )

    def _place_rectangle(self, stock_idx, rect, position):
        x, y, w, h = position[0], position[1], rect[0], rect[1]
        self._stocks[stock_idx][x:x + w, y:y + h] = 1
        for r in self._rectangles:
            rect_tensor = rect.clone().detach()  # Sử dụng clone và detach thay vì torch.tensor
            if torch.equal(r["size"], rect_tensor) and r["remaining"] > 0:
                r["remaining"] -= 1
                break

    def _calculate_reward(self):
        total_used_area = sum(torch.sum(stock == 1).item() for stock in self._stocks)
        total_unused_sheets = len([stock for stock in self._stocks if torch.all(stock == -2)])
        return total_used_area - total_unused_sheets * 10  # Ưu tiên sử dụng ít tấm hơn


    def render(self, mode="human"):
        """Visualize the cutting stock sheets and placements."""
        fig, axs = plt.subplots(1, len(self._stocks), figsize=(15, 5))
        if len(self._stocks) == 1:
            axs = [axs]
        elif len(self._stocks) > 1:
            axs = axs.ravel()

        for idx, stock in enumerate(self._stocks):
            ax = axs[idx]
            ax.set_title(f"Sheet {idx + 1}")
            ax.set_xlim(0, self.max_w)
            ax.set_ylim(0, self.max_h)
            ax.set_aspect("equal")

            # Vẽ từng ô của tấm
            for x in range(self.max_w):
                for y in range(self.max_h):
                    if stock[x, y] == -1:
                        ax.add_patch(patches.Rectangle((x, y), 1, 1, color="white", edgecolor="gray"))
                    elif stock[x, y] == 1:
                        ax.add_patch(patches.Rectangle((x, y), 1, 1, color="blue", edgecolor="gray"))
                    elif stock[x, y] == -2:
                        ax.add_patch(patches.Rectangle((x, y), 1, 1, color="red", edgecolor="gray"))

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            data = torch.frombuffer(fig.canvas.tostring_rgb(), dtype=torch.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data.cpu().numpy()

    def choose_best_sheet(self, rectangle_size):
        best_sheet_idx = -1
        min_waste = float('inf')
        rect_w, rect_h = rectangle_size

        for i, sheet in enumerate(self._stocks):
            positions = []
            for x in range(sheet.shape[0] - rect_w + 1):
                for y in range(sheet.shape[1] - rect_h + 1):
                    if torch.all(sheet[x:x+rect_w, y:y+rect_h] == -2):
                        positions.append((x, y))

            if positions:
                waste = torch.sum(sheet == -2).item() - (rect_w * rect_h)
                if waste < min_waste:
                    min_waste = waste
                    best_sheet_idx = i

        return best_sheet_idx

def test_cutting_stock_env():
    # Thiết lập môi trường
    env = CuttingStockEnv(
        seed=42,
        num_sheet=5,
        max_w=10,
        max_h=10,
        min_w=3,
        min_h=3,
        max_rec_type=5,
        max_rect_per_type=10,
        device="cpu"
    )
    obs = env.reset()

    # Kiểm tra cấu trúc quan sát
    assert "sheet" in obs, "Observation should contain 'sheet'."
    assert "rect" in obs, "Observation should contain 'rect'."
    assert len(obs["sheet"].shape) == 3, "Sheets should have 3 dimensions (num_sheet, width, height)."

    # Kiểm tra _can_place_ với hành động hợp lệ
    rect = torch.tensor([3, 3], dtype=torch.int32)
    position = torch.tensor([0, 0], dtype=torch.int32)
    can_place = env._can_place_(0, rect, position)
    assert can_place, "Should be able to place the rectangle at this position."

    # Kiểm tra _can_place_ với hành động không hợp lệ
    rect = torch.tensor([11, 11], dtype=torch.int32)
    can_place = env._can_place_(0, rect, position)
    assert not can_place, "Should not be able to place the rectangle outside the bounds."

    # Kiểm tra _place_rectangle
    rect = torch.tensor([3, 3], dtype=torch.int32)
    position = torch.tensor([0, 0], dtype=torch.int32)
    env._place_rectangle(0, rect, position)
    assert torch.all(env._stocks[0][0:3, 0:3] == 1), "Rectangle should be placed on the sheet."

    # Kiểm tra choose_best_sheet
    rectangle_size = torch.tensor([3, 3], dtype=torch.int32)
    best_sheet = env.choose_best_sheet(rectangle_size)
    assert best_sheet != -1, "Should find a valid sheet for the rectangle."

    # Kiểm tra step với hành động hợp lệ
    action = {
        "stock_index": 0,
        "size": torch.tensor([3, 3], dtype=torch.int32),
        "position": torch.tensor([4, 4], dtype=torch.int32)
    }
    obs, reward, done, info = env.step(action)
    assert reward > 0 or reward == -1, "Reward should be valid."
    assert not done, "The environment should not be done after one step."

    # Kiểm tra step với hành động không hợp lệ
    action = {
        "stock_index": 0,
        "size": torch.tensor([11, 11], dtype=torch.int32),
        "position": torch.tensor([0, 0], dtype=torch.int32)
    }
    obs, reward, done, info = env.step(action)
    assert reward == -1, "Reward should be -1 for invalid action."
    assert not done, "The environment should not be done after an invalid step."

    # Kiểm tra render
    try:
        env.render(mode="human")
        print("Render passed.")
    except Exception as e:
        assert False, f"Render failed: {e}"

    print("All tests passed.")

if __name__ == "__main__":
    test_cutting_stock_env()
