import torch
from typing import Callable

def energy_budget_clip(model, ratio: float = 0.1) -> Callable[[], None]:
    hooks = {}
    def hook(m, i, o): m._act = o.detach()
    for n, m in model.named_modules():
        if hasattr(m, 'weight'):
            hooks[n] = m.register_forward_hook(hook)

    def clip():
        for p in model.parameters():
            if p.grad is not None and hasattr(p, '_act'):
                g_norm = p.grad.norm()
                a_norm = p._act.norm()
                if g_norm > ratio * a_norm:
                    p.grad.data *= (ratio * a_norm) / g_norm
    return clip
