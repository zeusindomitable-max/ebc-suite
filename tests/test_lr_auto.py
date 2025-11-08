from ebc_suite import lr_auto

def test_lr():
    opt = torch.optim.SGD([torch.tensor(1.0)], lr=0)
    lr_auto(opt, 500, 1000)
    assert opt.param_groups[0]['lr'] > 1e-5
