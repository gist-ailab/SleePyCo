# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn

from loss import LOSS_MAP
from models.transformer.signal_backbone import SignalBackBone
from models.transformer.transformer import AutoEncoderViT
from models.base_model import BaseModel

"""
Transformer backbone Model: Model adapted from Neuronet (https://github.com/dlcjfgmlnasa/NeuroNet/tree/main)
"""
class TransformerBackbone(BaseModel):

    SUPPORTED_MODES = ['pretrain_mp', 'pretrain']  # support Contrastive Learning and Masked Prediction
    INTERNAL_LOSS_CALCULATION = True  # this model calculates its loss during training(in train modes) internally.
    INTERNAL_MASKING = True # TODO: make configurable ? (use frame bb with internal masking or no frame bb with full signal input to transformer)

    def __init__(self, mode: str, conf: dict):
        super().__init__(mode) # check if mode supported and set self.mode
        self.fs, self.second = conf["fs"], conf["second"]
        self.time_window = conf["time_window"]
        self.time_step = conf["time_step"]
        self.use_sig_backbone = conf["use_sig_backbone"]
        self.mask_ratio = conf["mask_ratio"]

        self.num_patches, _ = frame_size(fs=self.fs, second=self.second, time_window=self.time_window,
                                        time_step=self.time_step)
        self.frame_backbone = SignalBackBone(fs=self.fs, window=self.time_window)
        self.input_size = self.frame_backbone.feature_num

        if not self.use_sig_backbone:
            self.input_size = conf["input_size"]
            self.num_patches = conf["num_patches"]
        # Setup Autoencoder
        self.autoencoder = AutoEncoderViT(input_size=self.input_size,
                                          encoder_embed_dim=conf["encoder_embed_dim"], num_patches=self.num_patches,
                                          encoder_heads=conf["encoder_heads"], encoder_depths=conf["encoder_depths"],
                                          decoder_embed_dim=conf["decoder_embed_dim"],
                                          decoder_heads=conf["decoder_heads"],
                                          decoder_depths=conf["decoder_depths"])

        # setup loss function for internal loss
        if not 'loss' in conf.keys():
            raise ValueError("Need to define loss function for internal loss calculation (between masked latent frames and their reconstructions)")
        self.criterion = LOSS_MAP[conf['loss']](**(conf['loss_params'] if 'loss_params' in conf.keys() else {}))

        # setup projection networks that project frames before passed to encoder
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
            # in current dataloader, dummy dimension is in data ex (B, 1, 3000), We remove this dimension for this transformer model to work.
            x = torch.squeeze(x, 1)

        if self.use_sig_backbone:
            x = self.make_frame(x)
            x = self.frame_backbone(x)

        if self.mode == 'pretrain_mp':
            # Masked Prediction
            _, pred1, mask1 = self.autoencoder.forward_mask(x, self.mask_ratio)
            recon_loss1 = self.forward_mp_loss(x, pred1, mask1)
            return recon_loss1

        elif mode == 'pretrain':
            # Contrastive Learning
            latent1, pred1 = self.autoencoder.forward(x)  # todo: adapt to actual dataloading in this repo!
            latent2, pred2 = self.autoencoder.forward(x)
            o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()
            o1, o2 = self.projectors(o1), self.projectors(o2)
            contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)

            return contrastive_loss, (labels, logits)
        else:
            # If not in above train modes we want to return the output of the encoder so it can be used for benchmarking
            # or classification
            pred = self.autoencoder.forward(x)
            return pred  # TODO: check shape is real embedding and not frames


    def contrastive_loss(self, o1: torch.Tensor, o2: torch.Tensor):

        # TODO
        raise NotImplementedError()
        
        return 


    def forward_mp_loss(self,
                        real: torch.Tensor,
                        pred: torch.Tensor,
                        mask: torch.Tensor) -> float:
        """
        Computes the masked prediction loss between original frames after frame backbone and the ones after
        masked autoencoder (Transformer)
        params:
            real: original image/latent data before masking
            pred: predicted reconstruction of image/latent after masking
            mask: binary mask, telling where mask was applied
        """
        if self.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.e-6) ** .5

        loss = self.criterion(pred, real, reduction='none')
        if loss.ndim > mask.ndim: # if loss is not already compressing (like cos similarity) we need to calc mean over token this is for ex used in l2 loss
            loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


    def make_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates frames of a certain size from the epoch that is given.
        Ex. a 30s time series is split into frames of 5s with a step size of 2s, meaning frames can overlap.
        """
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frames = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frames.append(sample)
        frame = torch.stack(frames, dim=1)
        return frame

def frame_size(fs, second, time_window, time_step):
    """
    Simulates how the dimension of a set of generated frames from a given epoch will be.
    Returns shape of the data (set of frames from the epoch)
    """
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
    # number of frames, size of frames
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
        "input_size": 3000, # if use_sig_backbone: False, needs to be same as length of signal
        "num_patches": 50, # if use_sig_backbone: False, needs to be same as batch_size of signal
        "mask_ratio": 0.75
    }
    x0 = torch.randn((50, 3000)) # Transformer needs input in this form without middle dim ex (50, 1, 3000) but this is the case for CNN!!
    m1 = TransformerBackbone(mode, conf)
    pred = m1.forward(x0)
    print(f"Pred: {type(pred)}, len={len(pred)}")
    print(f"Pred-shape: {pred[0].shape}") # with sig backbone: (50, 51, 1472), without (50, 51, 500)
    # Regarding shape:  51 is probably num of frames/patches made from the 30s epoch! 500 is 5s-window in 10ms parts