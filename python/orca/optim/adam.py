import orca
from typing import Iterable, Tuple
from .optimizer import Optimizer
from orca.nn.parameter import Parameter

class Adam(Optimizer):
    """
    Implements Adam algorithm.
    
    Args:
        parameters (Iterable[Parameter]): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Default: 0.001.
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.999).
        eps (float, optional): term added to the denominator to improve numerical stability. Default: 1e-8.
    """
    def __init__(self, parameters: Iterable[Parameter], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # Initialize state
        self.m = []
        self.v = []
        for p in self.parameters:
            shape = p.tensor.shape
            device = p.tensor.device
            self.m.append(orca.Tensor.zeros(shape, device=device))
            self.v.append(orca.Tensor.zeros(shape, device=device))

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.tensor.requires_grad:
                grad = p.tensor.grad()
                if grad is None:
                    continue
                
                device = p.tensor.device
                
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m_prev = self.m[i]
                m_new = m_prev * self.beta1 + grad * (1.0 - self.beta1)
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v_prev = self.v[i]
                v_new = v_prev * self.beta2 + (grad * grad) * (1.0 - self.beta2)
                
                self.m[i] = m_new
                self.v[i] = v_new
                
                # Bias correction
                m_hat = m_new * (1.0 / (1.0 - self.beta1**self.t))
                v_hat = v_new * (1.0 / (1.0 - self.beta2**self.t))
                
                # Parameter update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                eps_tensor = orca.Tensor.from_list([self.eps], device=device).expand(v_hat.shape)
                denom = v_hat.sqrt() + eps_tensor
                
                update = (m_hat / denom) * self.lr
                new_tensor = p.tensor - update
                
                # Preserve requires_grad manually by detaching and re-enabling graph tracking
                p_new = new_tensor.detach()
                p_new.require_grad()
                p.update(p_new)
