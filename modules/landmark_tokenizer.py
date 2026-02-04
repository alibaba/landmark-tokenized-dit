import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.attention import Attention

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # or 'linear'
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,l = q.shape
        q = q.permute(0,2,1)   # b,l_q,c
        w_ = torch.bmm(q,k)     # b,l_q,l_k   
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        #v = v.reshape(b,c,h*w)  #l, c, l_k
        w_ = w_.permute(0,2,1)   # b,l_k,l_q (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,l_q (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        #h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_num=32, max_channels=512):
        super(ResBlock, self).__init__()
        skip = max(1, max_channels // out_channels - 1)
        self.block = nn.Sequential(
            nn.GroupNorm(group_num, in_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=skip, dilation=skip),
            nn.GroupNorm(group_num, out_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.conv_short = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        hidden_states = self.block(x)
        if hidden_states.shape != x.shape:
            x = self.conv_short(x)
        x = x + hidden_states
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        mid_channels=[128, 512],
        out_channels=3072,
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.attn1 = AttnBlock(mid_channels[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.attn2 = AttnBlock(mid_channels[1])
        self.conv_out = nn.Conv1d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # or 'linear'
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):

        x = self.conv_in(x)
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.attn1(x)
        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        # x = x + self.attn(x)
        x = self.attn2(x)
        x = self.conv_out(x)

        return x

class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels=3072, 
        mid_channels=[512, 128], 
        out_channels=2,
        ):
        super(Decoder, self).__init__()

        self.conv_in = nn.Conv1d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.attn1 = AttnBlock(mid_channels[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.attn2 = AttnBlock(mid_channels[1])
        self.conv_out = nn.Conv1d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # or 'linear'
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_in(x)
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.attn1(x)
        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        x = self.attn2(x)
        x = self.conv_out(x)

        return x

class VectorQuantizer(nn.Module):
    def __init__(self, nb_code, code_dim, is_train=False):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        self.reset_count = 0
        self.usage = torch.zeros((self.nb_code, 1))
        self.is_train = is_train
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))
    
    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out
    
    def init_codebook(self, x):
        if torch.all(self.codebook == 0):
            out = self._tile(x)
            self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        if self.is_train:
          self.init = True

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # [nb_code, code_dim]
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[torch.randperm(out.shape[0])[:self.nb_code]]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        self.usage = self.usage.to(usage.device)
        if self.reset_count >= 40:      # reset codebook every 40 steps for stability
            self.reset_count = 0
            usage = (usage + self.usage >= 1.0).float()
        else:
            self.reset_count += 1
            self.usage = (usage + self.usage >= 1.0).float()
            usage = torch.ones_like(self.usage, device=x.device)
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-6)))
            
        return perplexity

    def preprocess(self, x):
        # [bs, c, n] -> [bs * n, c]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)  
        return x

    def forward(self, x, return_vq=False):
        bs, c, n = x.shape 

        # Preprocess
        x = self.preprocess(x)
        assert x.shape[-1] == self.code_dim

        # Init codebook if not inited
        if not self.init and self.is_train:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.is_train:
            perplexity = self.update_codebook(x, code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        if return_vq:
            return x_d.view(bs, n, c).contiguous(), commit_loss

        # Postprocess
        x_d = x_d.view(bs, n, c).permute(0, 2, 1).contiguous()

        if self.is_train:
            return x_d, commit_loss, perplexity
        else:
            return x_d, commit_loss


class LM_VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super(LM_VQVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
    
    
    def encdec_slice_frames(self, x, encdec, return_vq):
        x_output = encdec(x)

        if encdec == self.encoder and self.vq is not None and not self.vq.is_train:
            x_output, loss = self.vq(x_output, return_vq=return_vq)
            return x_output, loss
        elif encdec == self.encoder and self.vq is not None and self.vq.is_train:
            x_output, loss, preplexity = self.vq(x_output)
            return x_output, loss, preplexity
        else:
            return x_output, None, None

    
    def forward(self, x, return_vq=False):
        x = x.permute(0, 2, 1)  #B, 2, N
        if not self.vq.is_train:
            x, loss = self.encdec_slice_frames(x, encdec=self.encoder, return_vq=return_vq)
        else:
            x, loss, perplexity = self.encdec_slice_frames(x, encdec=self.encoder, return_vq=return_vq)
        if return_vq:
            return x, loss
        x, _, _ = self.encdec_slice_frames(x, encdec=self.decoder, return_vq=return_vq)
        x = x.permute(0, 2, 1) #B, N, 2
        if self.vq.is_train:
            return x, loss, perplexity
        return x, loss