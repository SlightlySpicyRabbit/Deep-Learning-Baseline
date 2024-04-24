import torch
from torch.optim.optimizer import Optimizer


class Signum(Optimizer):
    """
    Signum optimizer, based on arXiv:1802.04434
    """

    def __init__(self, params, lr: float = 1e-4, beta: float = 0.9):
        """
        Initialize the optimizer
        :param params: Neural Network Parameters
        :param lr: Learning rate
        :param beta: Factor for first-order moment estimate
        """
        # Calling the constructor of the parent class and pass the parameter
        defaults = {"lr": lr, "betas": beta}
        super().__init__(params, defaults)

    # Overriding the update formula in pytorch
    @torch.no_grad()
    def step(self):
        # Update parameter
        for group in self.param_groups:
            for p in filter(lambda p_i: p_i.grad is not None, group["params"]):
                # Initialize first moment
                if len(self.state[p]) == 0:
                    self.state[p]["exp_avg"] = torch.zeros_like(p)
                # First moment update
                self.state[p]["exp_avg"].mul_(group["betas"]).add_(p.grad, alpha=1 - group["betas"])
                # Updating Neural Network Parameters
                update = self.state[p]["exp_avg"].clone().sign_()
                p.add_(update, alpha=-group["lr"])
        # Return
        return None
