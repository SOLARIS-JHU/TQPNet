from typing import Dict, Optional


class TableLogger:
    """Tabular logger for training progress.

    Prints a formatted table with loss breakdown and gradient norm.
    Column names are inferred dynamically from the breakdown dict keys.

    Example output:
       Epoch         Loss        Stat        Comp        Prim    GradNorm
    ---------------------------------------------------------------------
           1     2.341567    1.823412    0.412341    0.105814    3.241500
         100     0.523412    0.401234    0.089341    0.032837    1.102341
    """

    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self._header_printed = False
        self._keys: Optional[list] = None

    def log(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        breakdown: Optional[Dict[str, float]] = None,
        grad_norm: Optional[float] = None,
    ) -> None:
        """Log training metrics for one epoch.

        Args:
            epoch:        current epoch number
            total_epochs: total number of epochs
            train_loss:   total loss value
            breakdown:    dict of per-term losses (optional)
            grad_norm:    global gradient L2 norm (optional)
        """
        if not (epoch % self.log_every == 0 or epoch == 1 or epoch == total_epochs):
            return

        if breakdown is not None and self._keys is None:
            self._keys = list(breakdown.keys())

        if not self._header_printed:
            self._print_header(breakdown, grad_norm)
            self._header_printed = True

        row = f"{epoch:>8d} {train_loss:>12.6f}"
        if breakdown is not None and self._keys is not None:
            for k in self._keys:
                row += f"  {breakdown[k]:>10.6f}"
        if grad_norm is not None:
            row += f"  {grad_norm:>10.6f}"
        print(row)

    def _print_header(
        self,
        breakdown: Optional[Dict[str, float]],
        grad_norm: Optional[float],
    ) -> None:
        header = f"{'Epoch':>8} {'Loss':>12}"
        if breakdown is not None and self._keys is not None:
            for k in self._keys:
                header += f"  {k.capitalize():>10}"
        if grad_norm is not None:
            header += f"  {'GradNorm':>10}"
        sep = "-" * len(header)
        print(header)
        print(sep)