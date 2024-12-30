# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from models.transformer.signal_backbone import SignalBackBone
from models.transformer.transformer import AutoEncoderViT
from models.base_model import BaseModel

# TODO: there is no masking implemented yet!!!

"""
Transformer backbone Model : Model taken from https://github.com/dlcjfgmlnasa/NeuroNet/tree/main
"""
class TransformerBackbone(BaseModel):

    SUPPORTED_MODES = ['pretrain_mp', 'pretrain']  # support Contrastive Learning and Masked Prediction

    def __init__(self, mode: str, conf: dict):
        super().__init__(mode) # check if mode supported and set self.mode
        self.fs, self.second = conf["fs"], conf["second"]
        self.time_window = conf["time_window"]
        self.time_step = conf["time_step"]
        self.use_sig_backbone = conf["use_sig_backbone"]

        self.num_patches, _ = frame_size(fs=self.fs, second=self.second, time_window=self.time_window,
                                         time_step=self.time_step)
        self.frame_backbone = SignalBackBone(fs=self.fs, window=self.time_window)
        self.input_size = self.frame_backbone.feature_num
        if not self.use_sig_backbone:
            self.input_size = conf["input_size"]
        self.autoencoder = AutoEncoderViT(input_size=self.input_size,
                                          encoder_embed_dim=conf["encoder_embed_dim"], num_patches=self.num_patches,
                                          encoder_heads=conf["encoder_heads"], encoder_depths=conf["encoder_depths"],
                                          decoder_embed_dim=conf["decoder_embed_dim"],
                                          decoder_heads=conf["decoder_heads"],
                                          decoder_depths=conf["decoder_depths"])

        projection_hidden = [conf["encoder_embed_dim"]] + conf["projection_hidden"]
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

    def forward(self, x: torch.Tensor, squeeze: bool = True):
        """
        Gets a single eeg epoch or batched as input. Performs a forward pass. Supports different modes:
        1. if mode == pretrain_mp: use masked autoencoder and return reconstructed sequence/or reconstr latent
        2. if mode == pretrain: in contrastive learning, only use normal autoencoder without masking and return latent (so it can be used in train_crl.py)
        3. all other: assume we want classification - only return latent (so classifier can be used on top)
        """
        if squeeze:
            x = torch.squeeze(x, 1)
        x = self.make_frame(x)
        if self.use_sig_backbone:
            x = self.frame_backbone(x)

        if self.mode == 'pretrain_mp':
            latent, pred = self.autoencoder(x) # TODO: shape?
            return [pred]
        else:
            # includes mode == pretrain and all other
            latent = self.autoencoder.forward_encoder(x)
            latent_o = latent[:, :1, :].squeeze() # TODO: shape?
            return [latent_o]

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
    mode = 'pretrain_mp'
    conf = {
        "name": "Transformer",
        "fs": 100,
        "second": 30,
        "time_window": 5,
        "time_step": 0.5,
        "encoder_embed_dim": 256,
        "encoder_heads": 8,
        "encoder_depths": 6,
        "decoder_embed_dim": 128,
        "decoder_heads": 4,
        "decoder_depths": 8,
        "projection_hidden": [1024, 512],
        "use_sig_backbone": False,
        "input_size": 500
    }
    x0 = torch.randn((50, 3000)) # Transformer needs input in this form without middle dim ex (50, 1, 3000) but this is the case for CNN!!
    m1 = TransformerBackbone(mode, conf)
    pred = m1.forward(x0)
    print(f"Pred: {type(pred)}, len={len(pred)}")
    print(f"Pred-shape: {pred[0].shape}") # with sig backbone: (50, 51, 1472), without (50, 51, 500)
    # TODO: weird outputshape! -> 51 is probably num of frames/patches made from the 30s epoch! 500 is 5s-window in 10ms parts