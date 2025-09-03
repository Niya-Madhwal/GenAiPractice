import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils as data
import math
import copy

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads ==0,"d_model must be divisible by num_heads "
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def scaled_product_attention(self, Q, K, V, mask=None):
        attention_score = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            attention_score= attention_score.masked_fill(mask==0, -1e9)
        attention_probes= torch.softmax(attention_score, dim=-1)
        output= torch.matmul(attention_probes, V)
        return output
    def split_head(self, x):
        batch_size, _, seq_length, d_k= x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k= x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)

        output = self.scaled_product_attention(Q, K, V, mask)
        output = self.combine_heads(output)
        output = self.W_o(output)
        return output