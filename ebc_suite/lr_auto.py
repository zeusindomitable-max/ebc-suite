def lr_auto(opt, step, total, lr_max=3e-4, lr_min=1e-5, warmup=1000):
    if step < warmup:
        lr = lr_min + (lr_max - lr_min) * step / warmup
    else:
        lr = lr_max * (1 - (step - warmup) / (total - warmup))
    for g in opt.param_groups:
        g['lr'] = max(lr, lr_min)
