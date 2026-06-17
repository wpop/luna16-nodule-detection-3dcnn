"""
Early stopping utility for validation loss tracking.
"""


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping state.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0

    def should_stop(self, validation_loss: float) -> bool:
        """
        Return True when validation loss has not improved for patience steps.
        """

        if (
            self.best_loss is None
            or validation_loss < self.best_loss - self.min_delta
        ):
            self.best_loss = validation_loss
            self.counter = 0
            return False

        self.counter += 1

        return self.counter >= self.patience
