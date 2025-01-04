import abc
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for PyTorch neural network models.
    Enforces specification of supported modes and required init parameters.

    Attributes:
        SUPPORTED_MODES:            Subclasses can define which running modes they support
        INTERNAL_LOSS_CALCULATION:  Subclasses can define whether they calculate their loss internally or not.
                                    This is for ex. the case for the Transformer model which does not operate on one
                                    epoch but spits it internally in many different ones for training.
        INTERNAL_MASKING:           Indicates whether masking of inputs for masked prediction is done internally already.
    """

    SUPPORTED_MODES = []  # Should be overridden by subclasses
    INTERNAL_LOSS_CALCULATION = False
    INTERNAL_MASKING = False

    def __init__(self, mode, *args, **kwargs):
        """
        BaseModel initialization. enforces child-models to have the param 'mode' which is used to
        set the model behavior according to use-case.

        Args:
            mode (str): mode to use.

        Raises:
            NotImplementedError: If `SUPPORTED_MODES` is not specified in the subclass.
            OSError: If mode is not supported by the model.
        """
        super(BaseModel, self).__init__()
        if not self.SUPPORTED_MODES:
            raise NotImplementedError(
                "Child classes must specify a SUPPORTED_MODES list."
            )

        self.mode = mode
        if not self.mode in self.SUPPORTED_MODES:
            raise OSError(
                "Mode {} is not supported.".format(self.mode)
            )

    @abc.abstractmethod
    def forward(self, x):
        """
        Defines the forward pass of the model.
        Must be implemented by subclasses.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def is_mode_supported(self, mode_to_check: str) -> bool:
        return mode_to_check in self.SUPPORTED_MODES

    def is_using_internal_loss(self) -> bool:
        return self.INTERNAL_LOSS_CALCULATION

    def is_using_internal_masking(self) -> bool:
        return self.INTERNAL_MASKING


# Example subclass
"""
class ExampleModel(BaseModel):
    SUPPORTED_MODES = ["pretrain", "pretrain_mp"]

    def __init__(self, mode):
        super(ExampleModel, self).__init__(mode)
        ...

    def forward(self, x):
        pass
"""
