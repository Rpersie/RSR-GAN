#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src sent len, dec hid dim]
        
        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hid dim, src sent len]
        
        # context = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # context = [batch size, 1, dec hid dim]
        
        energy = torch.bmm(v, energy).squeeze(1)
        # energy = [batch size, src len]
        
        return F.softmax(energy, dim=1)

