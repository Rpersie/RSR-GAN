#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout_rate, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_rate = dropout_rate
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        # input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        context = torch.bmm(a, encoder_outputs)
        # context = [batch size, 1, enc hid dim * 2]
        
        context = context.permute(1, 0, 2)
        # context = [1, batch size, enc hid dim * 2]
        
        gru_input = torch.cat((embedded, context), dim=2)
        # gru_input = [1, batch size, (enc hid dim * 2) + emb dim]
        
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        
        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0) #[batch size, emb dim]
        output = output.squeeze(0) #[sent len, batch size, dec hid dim * n directions]??????????
        context = context.squeeze(0) #[batch size, enc hid dim * 2]
        
        output = self.out(torch.cat((output, context, embedded), dim=1))
        # output = [batch size, output dim]
        
        return output, hidden.squeeze(0)

