import torch


def probabilistic_random_masking(input_tensor: torch.Tensor, masking_ratio:float = 0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes an input tensor and performs masking on it according to the given ratio.
    It returns the original input data, the mask as well as the masked values.
    THE PORTION OF KEPT VALUES CAN VARY as the positions to keep are selected based on random noise value in this position

    Expected Input: tensor of shape (B, X) where B is the batch size and X is the signal dimension if a 1D EEG signal
    """
    # Validate masking_ratio
    if not (0.0 <= masking_ratio <= 1.0):
        raise ValueError("masking_ratio must be between 0 and 1.")
    if not input_tensor.ndim == 2:
        raise ValueError("input_tensor must have 2 dimensions (one batch and one signal dimension)")

    # Get the shape of the input tensor
    B, X = input_tensor.shape

    # Generate a random mask with the same shape as the input tensor
    mask = torch.rand(B, X) > masking_ratio

    # Apply the mask to the input tensor
    masked_tensor = input_tensor.clone()
    masked_tensor[~mask] = 0  # Set masked positions to 0 (or another value)
    mask = mask.long()

    return input_tensor, masked_tensor, mask


def fixed_proportion_random_masking(x: torch.Tensor, masking_ratio: float = 0.75, mask_value = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes an input tensor and performs masking on it according to the given ratio.
    It returns the original input data, the mask as well as the masked tensor.
    Compared it probabilistic_random_masking, this method sorts random noise and then takes a fixed portion from smallest to largest.
    This makes sure to always have the same amount of masking.

    Sets masked-value to 0.

    Expected Input: tensor of shape (B, X) where B is the batch size and X is the signal dimension.
    Outputs: original input tensor, masked tensor, mask, all having same dimension.
    """
    # Validate masking_ratio
    if not (0.0 <= masking_ratio <= 1.0):
        raise ValueError("masking_ratio must be between 0 and 1.")
    if not x.ndim == 2:
        raise ValueError("input_tensor must have 2 dimensions (one batch and one signal dimension)")

    n, l = x.shape  # batch, length, dim
    len_keep = int(l * (1 - masking_ratio))

    noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

    # sort noise for each sample (get indexes sorting signal dimension)
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

    # generate the binary mask: 1 is keep, 0 is remove
    mask = torch.zeros([n, l], device=x.device)
    mask[:, :len_keep] = 1
    mask = mask.scatter(1, ids_shuffle, mask).long() # shuffle mask according to random indices

    x_masked = x.clone() * mask

    # sanity check
    assert x_masked.shape == mask.shape == x.shape

    return x, x_masked, mask


MASKING_MAP = {
    "probabilistic_random": probabilistic_random_masking,
    "fixed_proportion_random": fixed_proportion_random_masking
}

SUPPORTED_MASKING = list(MASKING_MAP.keys())
