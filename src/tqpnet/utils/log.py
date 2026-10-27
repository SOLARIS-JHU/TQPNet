class TableLogger:
    """Tabular logger for training progress.

    Example output:
       Epoch         Loss     Residual
    ----------------------------------
           1     2.341567   9.4e+00
         100     0.523412   6.9e-01
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self._header_printed = False

    def log(self, epoch: int, total_epochs: int, loss: float, residual: float) -> None:
        """Print one row for ``epoch`` if it is a logging epoch.

        Args:
            epoch:        current epoch (1-based)
            total_epochs: planned number of epochs
            loss:         epoch-mean training loss log(1 + residual)
            residual:     epoch-mean raw residual ‖r‖² (weighted)
        """
        if not (epoch % self.log_every == 0 or epoch == 1 or epoch == total_epochs):
            return
        if not self._header_printed:
            header = f"{'Epoch':>8} {'Loss':>12} {'Residual':>12}"
            print(header)
            print("-" * len(header))
            self._header_printed = True
        print(f"{epoch:>8d} {loss:>12.6f} {residual:>12.3e}")
