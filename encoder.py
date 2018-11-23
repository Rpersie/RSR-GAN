#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Encoder(nn.Module):
    def __init__(self, seq_len, input_size, enc_hid_dim, dec_hid_dim, dropout_rate):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_rate = dropout_rate
        
        self._input_size = (enc_hid_dim * 2) + enc_hid_dim
        
        self.bi_gru1 = nn.GRU(input_size=input_size, 
                                              hidden_size=enc_hid_dim, bidirectional=True)
        self.bi_gru2 = nn.GRU(input_size=self._input_size, 
                              hidden_size=enc_hid_dim, bidirectional=True)
        self.bi_gru3 = nn.GRU(input_size=self._input_size, 
                              hidden_size=enc_hid_dim, bidirectional=True)
        
        self.bi_gru4 = nn.GRU(input_size=self._input_size, 
                              hidden_size=enc_hid_dim, bidirectional=True)
        self.bi_gru5 = nn.GRU(input_size=self._input_size, 
                              hidden_size=enc_hid_dim, bidirectional=True)
        self.bi_gru6 = nn.GRU(input_size=self._input_size, 
                              hidden_size=enc_hid_dim, bidirectional=True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.pool =  nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
    def forward(self, input):
        bi_gru1_out = self.layerBlock(self.bi_gru1, input, self.pool)
        bn1 = nn.BatchNorm1d(num_features=bi_gru1_out.size(1))
        bi_gru1_out = self.dropout(F.leaky_relu(bn1(bi_gru1_out)).permute(1, 0, 2))
        
        bi_gru2_out = self.layerBlock(self.bi_gru2, bi_gru1_out, self.pool)
        bn2 = nn.BatchNorm1d(num_features=bi_gru2_out.size(1))
        bi_gru2_out = self.dropout(F.leaky_relu(bn2(bi_gru2_out)).permute(1, 0, 2))
        
        bi_gru3_out = self.layerBlock(self.bi_gru3, bi_gru2_out, self.pool)
        bn3 = nn.BatchNorm1d(num_features=bi_gru3_out.size(1))
        bi_gru3_out = self.dropout(F.leaky_relu(bn3(bi_gru3_out)).permute(1, 0, 2))
        
        bi_gru4_out = self.layerBlock(self.bi_gru4, bi_gru3_out)
        bn4 = nn.BatchNorm1d(num_features=bi_gru4_out.size(1))
        bi_gru4_out = self.dropout(F.leaky_relu(bn4(bi_gru4_out)).permute(1, 0, 2))
        
        bi_gru5_out = self.layerBlock(self.bi_gru5, bi_gru4_out)
        bn5 = nn.BatchNorm1d(num_features=bi_gru5_out.size(1))
        bi_gru5_out = self.dropout(F.leaky_relu(bn5(bi_gru5_out)).permute(1, 0, 2))
        
        bi_gru6_out, bi_gru6_h_n = self.bi_gru6(bi_gru5_out)
        bn6_out = nn.BatchNorm1d(bi_gru6_out.size(0))
        bn6_h_n = nn.BatchNorm1d(bi_gru6_h_n.size(0))
        bi_gru6_out = F.leaky_relu(bn6_out(bi_gru6_out.permute(1, 0, 2))).permute(1, 0, 2)
        bi_gru6_h_n = F.leaky_relu(bn6_h_n(bi_gru6_h_n.permute(1, 0, 2))).permute(1, 0, 2)
        
        init_hidden_decoder = self.fc(torch.cat((bi_gru6_h_n[-2, :, : ], bi_gru6_h_n[-1, :, : ]), 
                                                dim=1))
        
        return bi_gru6_out, bi_gru6_h_n, init_hidden_decoder
        
        
    def layerBlock(self, gru, input, pool=None):
        output, h_n = gru(input)
        reminder = output.size(0) % h_n.size(0)
        h_n = h_n.repeat(math.floor(output.size(0) / h_n.size(0)), 1, 1)
        if not reminder == 0:
            zeros = torch.zeros(output.size(0) % h_n.size(0), h_n.size(1), h_n.size(2))
            h_n = torch.cat((h_n, zeros), dim=0)
        merge_output = torch.cat((output, h_n), dim=2)
        merge_output = merge_output.permute(1, 0, 2)
        
        if pool:
            merge_output = merge_output.unsqueeze(1)
            merge_output = pool(merge_output)
            merge_output = merge_output.squeeze(1)
        
        return merge_output

