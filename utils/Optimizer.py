import torch
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    r"""Implements Lion algorithm."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float): learning rate (default: 1e-4)
            betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float): weight decay (L2 penalty) (default: 0)
        """
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lion, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Get hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                # Update biased first moment estimate
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad - exp_avg, grad - exp_avg, value=1 - beta2)

                # Compute the bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = lr / bias_correction1

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

        return loss