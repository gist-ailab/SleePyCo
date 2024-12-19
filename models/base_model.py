import abc
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for PyTorch neural network models.
    Enforces specification of supported modes and required init parameters.
    """

    SUPPORTED_MODES = []  # Should be overridden by subclasses

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
