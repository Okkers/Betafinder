import math 
import numpy as np 
from typing import Dict, List, Tuple, Union 

import torch 
import torch.nn as nn
import torch.nn.functional as F

''' 
@misc{vaswani2023attention,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
'''

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(Multi_Head_Attention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, 
                                           k: torch.Tensor, 
                                           v: torch.Tensor, 
                                           mask: torch.Tensor = None):
        
        qk = torch.matmul(q, torch.transpose(k, -2, -1)) / torch.sqrt(self.d_k)
        if mask is not None: 
            qk = qk.masked_fill(mask == 0, -math.inf)

        soft_qk = torch.softmax(qk, dim=-1)
        output = torch.matmul(soft_qk, v)

        return output
    
    def split_heads(self, x: torch.Tensor):

        batch_size, seq_length, _ = x.size()
        return torch.transpose(x.view(batch_size, seq_length, self.n_heads, self.d_k), 1,2)
    
    def combine_heads(self, x: torch.Tensor):

        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, q: torch.Tensor, 
                      k: torch.Tensor, 
                      v: torch.Tensor, 
                      mask: torch.Tensor = None):
        
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        attn_out = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_out))
        return output
        
class Position_Feed_Forward(nn.Module):
    def __init__(self, d_model: int, d_ff):
        super(Position_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
class Positional_Encoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super(Positional_Encoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        res = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype = torch.float()).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        res[:,0::2] = torch.sin(position * div_term)
        res[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("res", res.unsqueeze(0))

    def forward(self, x):
        return x + self.res[:,:x.size(1)]
        
class Encoder_layer(nn.Module):
    def __init__(self, d_model: int, 
                       num_heads: int, 
                       d_ff: int, 
                       dropout: int):
        
        super(Encoder_layer, self).__init__()
        self.attn = Multi_Head_Attention(d_model, num_heads)
        self.fc = Position_Feed_Forward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout 

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        attn = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.fc(x)
        out = self.norm2(x + self.dropout(ff))

        return out

class Decoder_Layer(nn.Module):
    def __init__(self, d_model: int, 
                       num_heads: int, 
                       d_ff: int, 
                       dropout: int):
        
        super(Decoder_Layer, self).__init__()
        self.masked_attn = Multi_Head_Attention(d_model, num_heads)
        self.attn = Multi_Head_Attention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.fc = Position_Feed_Forward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                      enc_out: torch.Tensor, 
                      src_mask: torch.Tensor, 
                      tgt_mask: torch.Tensor):
        
        mask_attn = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(mask_attn))
        attn = self.attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn))
        ff = self.fc(x)
        x = self.norm3(x + self.dropout(ff))
        return x