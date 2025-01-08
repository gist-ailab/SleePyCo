import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ReconstructionLoss(nn.Module):
    """
    Basic reconstruction loss as in MAEEG (https://arxiv.org/pdf/2211.02625).
    Masking needs to be done before! (take out valid values before)
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, reduction:str='mean', mask=None, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Goal: take single channel EEG epoch and the reconstructed output (after a Masked AutoEncoder for ex.) and calculate
        a normalized reconstruction loss.

        If shape is has more then 2 dimensions. We assume 1st dim is batch dim and the rest dim will be merged

        inputs: raw EEG epoch from dataset which was input to model
        outputs: reconstructed EEG epoch from model
        labels: ground truth epoch from dataset but these are not needed for this loss
        """
        assert inputs.shape == outputs.shape
        if mask is not None:
            assert mask.shape == inputs.shape

        #if inputs.ndim > 2:
        #    batch_size = inputs.shape[0]
        #    inputs = inputs.reshape(batch_size, -1)
        #    outputs = outputs.reshape(batch_size, -1)
        #    if mask is not None:
        #        mask = mask.reshape(batch_size, -1)

        # apply mask to set values equal where masked, so they don't contribute to similarity
        if mask is not None:
            inputs = inputs * mask
            outputs = outputs * mask

        # treat non batch dim as vector to calculate similarity form and then calc mean over batch dim to get number
        batched_cos_sim = F.cosine_similarity(inputs, outputs, dim=-1)
        if reduction == 'mean':
            return 1 - batched_cos_sim.mean()
        elif reduction == 'none':
            return torch.ones_like(batched_cos_sim) - batched_cos_sim
        else:
            raise ValueError('Unknown reduction: {}'.format(reduction))


class L2Loss(nn.Module):
    """
    Wrapper for basic L2 loss. Masking needs to be done before!
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, reduction: str='mean', mask = None, labels: torch.Tensor = None) -> torch.Tensor:
        """
        If shape is has more then 2 dimensions. We assume 1st dim is batch dim and the rest dim will be merged

        inputs: raw EEG epoch from dataset which was input to model
        outputs: reconstructed EEG epoch from model
        labels: ground truth epoch from dataset but these are not needed for this loss
        """
        assert inputs.shape == outputs.shape
        if mask is not None:
            assert inputs.shape == mask.shape

        square_dist = (outputs - inputs).pow(2)
        if reduction == 'mean':
            if mask is not None:
                return (square_dist * mask).sum() / mask.sum()
            else:
                return torch.mean(square_dist)
        elif reduction == 'none':
            return square_dist
        else:
            raise ValueError('Unknown reduction method: {}'.format(reduction))

class CrossEntropyLoss(nn.Module):
    """
    Wrapper for basic L2 loss. Masking needs to be done before!
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, reduction: str='mean', mask = None, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates classification loss

        inputs: None expected
        outputs: Predicted logits
        labels: ground truth class labels for epochs
        """
        return self.ce(outputs, labels)


class NTXentLoss(nn.Module):
    """NT-Xent Loss for unsupervised contrastive learning (SimCLR).

    This implementation expects:
    features: [batch_size, 2, embedding_dim]
    i.e., exactly two augmented views per sample.

    No labels are required. If provided, they will be ignored or raise an error.
    """

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, reduction: str='mean', mask = None, labels: torch.Tensor = None):
        """
        features: [B, n_views, ...], typically n_views=2
        labels: not required for NT-Xent (will raise error if provided)

        Returns:
            A scalar loss value.
        """
        device = outputs.device
        if labels is not None:
            raise ValueError("NTXentLoss does not use labels; got labels as input.")

        if len(outputs.shape) < 3:
            raise ValueError('`features` must be [bsz, n_views, feature_dim] at least.')
        if len(outputs.shape) > 3:
            outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)

        bsz, n_views, dim = outputs.shape
        if n_views != 2:
            raise ValueError("NT-Xent loss requires exactly 2 views.")

        # Split into two views
        f1 = outputs[:, 0, ...]  # [B, dim]
        f2 = outputs[:, 1, ...]  # [B, dim]

        # Normalize embeddings
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        # Concatenate embeddings
        embeddings = torch.cat([f1, f2], dim=0)  # [2B, dim]

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        # sim_matrix shape: [2B, 2B]

        # Mask self-similarities
        mask = torch.eye(2 * bsz, dtype=torch.bool).to(device)
        sim_matrix = sim_matrix.masked_fill_(mask, float('-inf'))

        # Subtract max logits for numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # The positive samples for each embedding are the corresponding augmented pair:
        # For i in [0, B-1], the positive of f1[i] is f2[i] at index i+B.
        # For i in [B, 2B-1], the positive of f2[i-B] is f1[i-B] at index i-B.
        # Create labels accordingly:
        # The target for f1[i] (row i) is i+B
        # The target for f2[i] (row i+B) is i
        labels = torch.arange(bsz, dtype=torch.long, device=device)
        labels = torch.cat([labels + bsz, labels], dim=0)  # [2B]

        # Compute NT-Xent loss using CrossEntropy
        loss = self.ce(sim_matrix, labels)
        return loss

LOSS_MAP = {
    "supcon_sleepyco": SupConLoss,
    "reconstruction_loss_maeeg": ReconstructionLoss,
    "l2": L2Loss,
    "cross_entropy": CrossEntropyLoss,
    "NTXent": NTXentLoss,
}

SUPPORTED_LOSS_FUNCTIONS = list(LOSS_MAP.keys())