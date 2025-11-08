from ebc_suite import mem_safe_forward

def test_mem():
    m = torch.nn.Transformer()
    x = torch.randn(1, 16384, 64)
    out = mem_safe_forward(m, x, chunk=4096, checkpoint=False)
    assert out.shape[1] == 16384
