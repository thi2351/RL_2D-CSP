import gym 
import numpy as np 
from gym import spaces 

class CuttingStockEnv(gym.Env):
    def __init__(self, sheet_size, rect):
        super(CuttingStockEnv, self).__init__()
        self.sheet_size = sheet_size # witdh and height
        self.rect = rect # list of rectangle (witdh, height)
        self.grid = np.zeros(self.sheet_size)
        
        # Define remaining rectangles
        self.remaining_rect = self.rect.copy()

        # Define action and observation
        self.valid_action = self._get_valid_action_()
        self.action_space = spaces.Discrete(len(self.valid_action))
        observation_space = spaces.Dict({
                            "grid": spaces.Box(low=0, high=1, shape=sheet_size, dtype=np.int8),
                            "remaining_rects": spaces.Box(low=0, high=max(sheet_size), shape=(len(self.rect), 2), dtype=np.int32),
                            "used_area": spaces.Box(low=0, high=sheet_size[0] * sheet_size[1], shape=(), dtype=np.int32)
                            })

    def _get_valid_action_(self):
        valid_action = []
        for rect  in self.remaining_rect:
            for x in range(self.sheet_size[0]):
                for y in range(self.sheet_size[1]):
                    if self._can_place_([x, y]):
                        valid_action.append(x, y, rect)
        return valid_action
    
    def _can_place_(self, x, y,rect):
        if x + rect[0] > self.sheet_size[0] or y + rect[1] > self.sheet_size[1]:
            return False
        return np.all(self.grid[x:x+rect[0], y:y+rect[1] == 0])
    
    def reset(self):
        self.grid = np.zeros(self.sheet_size, dtype=np.int8)
        self.remaining_rect = self.rect.copy()
        self.used_area = 0
        self.valid_action = self._get_valid_action_()
        return self._get_observation()
    
    def _get_observation(self):
        padded_rects = np.array(
            self.remaining_rect + [[0, 0]] * (len(self.rect) - len(self.remaining_rect))
        )
        return {
            "grid": self.grid,
            "remaining_rects": padded_rects,
            "used_area": self.used_area
        }