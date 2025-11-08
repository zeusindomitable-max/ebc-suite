import torch
from typing import Callable

def act_guardian(model, min_act: float = 1e-3, noise: float = 1e-4) -> Callable[[], None]:
    hooks = {}
    def hook(m, i, o): m._act = o.detach()
    for n, m in model.named_modules():
        if hasattr(m, 'weight'):
            hooks[n] = m.register_forward_hook(hook)

    def guard():
        for m in model.modules():
            if hasattr(m, '_act'):
                if m._act.norm() < min_act:
                    m._act += torch.randn_like(m._act) * noise
    return guard
