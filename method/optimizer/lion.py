import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer, based on arXiv:2302.06675
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99)):
        """
        Initialize the optimizer
        :param params: Neural Network Parameters
        :param lr: Learning rate
        :param betas: Factor for first-order moment estimate and factor for weighted update
        """
        # Calling the constructor of the parent class and pass the parameter
        defaults = {"lr": lr, "betas": betas}
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
                beta_1, beta_3 = group["betas"]
                # Updating Neural Network Parameters
                update = self.state[p]["exp_avg"].clone().mul_(beta_3).add(p.grad, alpha=1 - beta_3).sign_()
                p.add_(update, alpha=-group["lr"])
                # First moment update
                self.state[p]["exp_avg"].mul_(beta_1).add_(p.grad, alpha=1 - beta_1)
        # Return
        return None
