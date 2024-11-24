from A2C import A2C
import torch
import torch.nn as nn 
from Env import CuttingStockEnv 



class A2C_Agent():
    def __init__(self, actor_critic: A2C, value_loss_coef, entropy_coef, lr, episode, max_grad_norm):
        