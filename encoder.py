#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[2]:


class Encoder(nn.Module):
    def __init__(self, input_size, num_gru, enc_hid_dim,
                 dec_hid_dim, dropout_rate, use_pooling=False):
        super().__init__()
        self.input_size = input_size
        self.num_gru = num_gru
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_rate = dropout_rate
        self.use_pooling = use_pooling
        
        self.rnn_stack = nn.ModuleList()
        for i in range(num_gru):
            _input_size = input_size if i == 0 else enc_hid_dim * 2
            self.rnn_stack.append(self.biGru(input_size=_input_size, 
                                  hidden_size=enc_hid_dim, dropout_rate=dropout_rate))
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.pool =  nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        
    def forward(self, input):
        for gru in self.rnn_stack:
            output, h_n = self.layerBlock(gru, input)
            input = output        
        
        init_hidden_decoder = self.fc(torch.cat((h_n[-2, :, : ], h_n[-1, :, : ]), 
                                                dim=1))
        
        return output, h_n , init_hidden_decoder
    
    
    def biGru(self, input_size, hidden_size, dropout_rate):
        return nn.GRU(input_size=input_size, hidden_size=hidden_size, bias=True, 
                      bidirectional=True, batch_first=True, dropout=dropout_rate)
    
        
    def layerBlock(self, gru, input):
        # input = [batch_size, seq_len, input_size]
        
        output, h_n = gru(input)
        # output = [batch_size, seq_len, enc_hid_dim * num_directions]
        # h_n = [num_layers * num_directions, batch_size, enc_hid_dim]
        
        batch_norm = nn.BatchNorm1d(num_features=output.size(2))
        # batch_norm input -> (N,C,L), where C is num_features. 
        
        output = batch_norm(output.permute(0, 2, 1)).permute(0, 2, 1)
        # first permute to match batch_norm input convention 
        # then second permute to contruct original shape.
        # output = [batch_size, seq_len, enc_hid_dim * num_directions]
        
        output = F.leaky_relu(output)
        
        if self.use_pooling:
            raise NotImplementedError('Implement pooling option for first 3 layer.')
            """
            reminder = output.size(0) % h_n.size(0)
            h_n = h_n.repeat(math.floor(output.size(0) / h_n.size(0)), 1, 1)
            if not reminder == 0:
                zeros = torch.zeros(output.size(0) % h_n.size(0), h_n.size(1), h_n.size(2))
                h_n = torch.cat((h_n, zeros), dim=0)
            merge_output = torch.cat((output, h_n), dim=2)
            merge_output = merge_output.permute(1, 0, 2)
            merge_output = merge_output.unsqueeze(1)
            merge_output = pool(merge_output)
            merge_output = merge_output.squeeze(1)
            """
        
        return output, h_n