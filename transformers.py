import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        