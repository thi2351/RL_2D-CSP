import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import patches

class CuttingStockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, num_sheet=100, max_w=100, max_h=100, min_w=50, min_h=50, max_rec_type=20, max_rect_per_type=25):
        super(CuttingStockEnv, self).__init__()
        self.min_w, self.min_h, self.max_w, self.max_h = min_w, min_h, max_w, max_h
        self.num_sheet, self.max_rec_type, self.max_rect_per_type = num_sheet, max_rec_type, max_rect_per_type

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
            "size": spaces.Box(np.array([1, 1]), np.array([max_w, max_h]), dtype=int),
            "position": spaces.Box(np.array([0, 0]), np.array([max_w - 1, max_h - 1]), dtype=int),
        })

        self._stocks = []
        self._rectangles = []
        self.reset()

    def reset(self, seed=None):
        np.random.seed(seed)
        self._stocks = [np.full((self.max_w, self.max_h), fill_value=-2) for _ in range(self.num_sheet)]
        self._rectangles = [
            {"size": np.array([np.random.randint(1, self.max_w), np.random.randint(1, self.max_h)]), 
             "remaining": np.random.randint(1, self.max_rect_per_type)}
            for _ in range(np.random.randint(1, self.max_rec_type))
        ]
        return self._get_observation()

    def _get_observation(self):
        return {"sheet": np.array(self._stocks), "rect": self._rectangles}

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
        return x + w <= self.max_w and y + h <= self.max_h and np.all(self._stocks[stock_idx][x:x + w, y:y + h] == -2)

    def _place_rectangle(self, stock_idx, rect, position):
        x, y, w, h = position[0], position[1], rect[0], rect[1]
        self._stocks[stock_idx][x:x + w, y:y + h] = 1
        for r in self._rectangles:
            if np.array_equal(r["size"], rect) and r["remaining"] > 0:
                r["remaining"] -= 1
                break

    def _calculate_reward(self):
        total_used_area = 0
        total_area = 0
        used_sheets = 0

        for stock in self._stocks:
            sheet_area = np.sum(stock != -2)
            used_area = np.sum(stock == 1)
            if used_area > 0:
                used_sheets += 1
                total_used_area += used_area
                total_area += sheet_area

        return -total_used_area / total_area if total_area > 0 else -1
    
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
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data
