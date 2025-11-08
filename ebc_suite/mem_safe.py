import torch
import torch.utils.checkpoint as cp

def mem_safe_forward(model, x, chunk=4096, checkpoint=True):
    chunks = x.split(chunk, dim=1)
    outs = []
    for c in chunks:
        def fwd(c=c): return model.transformer(c)
        out = cp.checkpoint(fwd) if checkpoint and x.requires_grad else fwd()
        outs.append(out)
    return torch.cat(outs, dim=1)
