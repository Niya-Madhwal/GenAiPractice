import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class ModelTraining(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=2):
        super().__init__()