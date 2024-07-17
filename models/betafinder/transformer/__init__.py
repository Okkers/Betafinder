import math 
import numpy as np 
from typing import Dict, List, Tuple, Union 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from sub_modules import Encoder_layer, Decoder_Layer, Positional_Encoding

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

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 max_seq_len: int, 
                 d_model: int = 512, 
                 num_heads: int = 8, 
                 num_layers: int = 6, 
                 d_ff: int = 2048, 
                 dropout: int = 0.1):
        super(Transformer, self).__init__()
        self.Enc_embedding = nn.Embedding(src_vocab_size, d_model)
        self.Dec_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.Positional_Encoding = Positional_Encoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList([Encoder_layer(d_model, num_heads, d_ff, dropout)] for _ in range(num_layers))
        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, d_ff, dropout)] for _ in range(num_layers))

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt: torch.Tensor, src: torch.Tensor):

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        
        src_mask, tgt_mask = self.generate_mask(tgt, src)
        src_embed = self.dropout(self.Positional_Encoding(self.Enc_embedding(src)))
        tgt_embed = self.dropout(self.Positional_Encoding(self.Enc_embedding(tgt)))

        enc_output = src_embed
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embed 
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output