import torch
from ebc_suite import energy_budget_clip

def test_ebc():
    m = torch.nn.Linear(10, 10)
    x = torch.randn(1, 10)
    clip = energy_budget_clip(m)
    (m(x)**2).mean().backward()
    clip()
    assert m.weight.grad.norm() < 1.0
