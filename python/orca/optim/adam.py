import orca
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
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

    def step(self):
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
                
                # Preserve requires_grad manually by recreating or using update logic
                p_new = orca.Tensor.from_list(new_tensor.to_list(), shape=new_tensor.shape, requires_grad=True, device=device)
                p.update(p_new)
