import concurrent.futures
import threading
import torch.nn as nn
import torch.nn.functional as F
import torch


class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, attention_r2, d_model, d_ff=None, dropout=0.1, activation="relu",enc_in=2):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.attention_r2 = attention_r2
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.enc_in=enc_in
    #@staticmethod
    #def smart_dtype(self,x):


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        X1 = self.attention(inputs_embeds=x, attn_mask=None)
        if hasattr(X1, "last_hidden_state"):
            X1 = X1.last_hidden_state

        # Forward pass through second attention (reversed direction)
        #X2 = self.attention_r(inputs_embeds=x.flip(dims=[1]), attn_mask=None)
        #if hasattr(X2, "last_hidden_state"):
        #    X2 = X2.last_hidden_state
        #X2 = X2.flip(dims=[1])

        # Combine both directions
        new_x = X1 #+ X2
       
        attn = 1
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y=self.norm2(x + y)

        if attn_mask is not None:
            y, attn = self.attention_r2(y[:,:attn_mask.shape[-1],:], y[:,:attn_mask.shape[-1],:], y[:,:attn_mask.shape[-1],:], attn_mask= attn_mask, tau=tau, delta=delta)  
        else:
            y, attn = self.attention_r2(y,y,y, attn_mask= attn_mask, tau=tau, delta=delta) 
        return y, attn



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None,  tau=None, delta=None):

        orig_B = x.size(0)
        orig_L = x.size(1) if x.ndim > 1 else 1  # sequence length (if 1D, set L=1)
        
        x = x.to(dtype=torch.float16).contiguous() 

        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x,attn_mask=attn_mask,  tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x[:orig_B,:orig_L, ...], attns



