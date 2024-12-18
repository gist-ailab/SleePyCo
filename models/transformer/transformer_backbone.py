# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from typing import List
from .signal_backbone import SignalBackBone
from .transformer import AutoEncoderViT
from functools import partial


class transformer_backbone(nn.Module):
    def __init__(self, fs: int = 100, second: int = 30, time_window: int = 5, time_step: float = 0.5,
                 encoder_embed_dim = 256, encoder_heads: int = 6, encoder_depths: int = 8,
                 decoder_embed_dim: int = 128, decoder_heads: int = 4, decoder_depths: int = 8,
                 projection_hidden: List = [1024, 512], temperature=0.01, use_sig_backbone=True, input_size=500):
        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step
        self.use_sig_backbone = use_sig_backbone

        self.num_patches, _ = frame_size(fs=fs, second=second, time_window=time_window, time_step=time_step)
        self.frame_backbone = SignalBackBone(fs=self.fs, window=self.time_window)
        self.input_size = self.frame_backbone.feature_num
        if not use_sig_backbone:
            self.input_size = input_size
        self.autoencoder = AutoEncoderViT(input_size=self.input_size,
                                                encoder_embed_dim=encoder_embed_dim, num_patches=self.num_patches,
                                                encoder_heads=encoder_heads, encoder_depths=encoder_depths,
                                                decoder_embed_dim=decoder_embed_dim, decoder_heads=decoder_heads,
                                                decoder_depths=decoder_depths)

        projection_hidden = [encoder_embed_dim] + projection_hidden
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            if i != len(projection_hidden) - 2:
                projectors.append(nn.Linear(h1, h2))
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
            else:
                projectors.append(nn.Linear(h1, h2))
        self.projectors = nn.Sequential(*projectors)
        self.projectors_bn = nn.BatchNorm1d(projection_hidden[-1], affine=False)
        self.norm_pix_loss = False

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.make_frame(x)
        if self.use_sig_backbone:
            x = self.frame_backbone(x)

        # Masked Prediction
        latent1, pred1 = self.autoencoder(x)
        latent2, pred2 = self.autoencoder(x)
        o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()
        recon_loss1 = self.forward_mae_loss(x, pred1, torch.ones(pred1.size))
        recon_loss2 = self.forward_mae_loss(x, pred2, np.ones(pred1.size))
        recon_loss = recon_loss1 + recon_loss2

        # Contrastive Learning
        o1, o2 = self.projectors(o1), self.projectors(o2)
        contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)
        print(contrastive_loss)
        return recon_loss, contrastive_loss, (labels, logits)

    def forward_latent(self, x: torch.Tensor):
        x = self.make_frame(x)
        if self.use_sig_backbone:
            x = self.frame_backbone(x)
        latent = self.autoencoder.forward_encoder(x)
        latent_o = latent[:, :1, :].squeeze()
        return latent_o
    
    def forward_predict(self, x: torch.Tensor):
        x = self.make_frame(x)
        if self.use_sig_backbone:
            x = self.frame_backbone(x)
        
        latent, pred = self.autoencoder(x)
        latent_o = latent[:, :1, :].squeeze()
        return latent, pred

    def forward_mae_loss(self,
                         real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        if self.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.e-6) ** .5

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1)
        return frame

def frame_size(fs, second, time_window, time_step):
    x = np.random.randn(1, fs * second)
    size = fs * second
    step = int(time_step * fs)
    window = int(time_window * fs)
    frame = []
    for i in range(0, size, step):
        start_idx, end_idx = i, i + window
        sample = x[..., start_idx: end_idx]
        if sample.shape[-1] == window:
            frame.append(sample)
    frame = np.stack(frame, axis=1)
    return frame.shape[1], frame.shape[2]



if __name__ == '__main__':
    x0 = torch.randn((50, 3000))
    m0 = transformer_backbone(fs=100, second=30, time_window=5, time_step=0.5,
                  encoder_embed_dim=256, encoder_depths=6, encoder_heads=8,
                  decoder_embed_dim=128, decoder_heads=4, decoder_depths=8,
                  projection_hidden=[1024, 512], use_sig_backbone=True)
    m1 = transformer_backbone(fs=100, second=30, time_window=5, time_step=0.5,
                  encoder_embed_dim=256, encoder_depths=6, encoder_heads=8,
                  decoder_embed_dim=128, decoder_heads=4, decoder_depths=8,
                  projection_hidden=[1024, 512], use_sig_backbone=False)
    latent, pred = m1.forward_predict(x0)
    print(f"{latent.shape}, {pred.shape}")