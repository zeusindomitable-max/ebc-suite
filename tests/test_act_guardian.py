from ebc_suite import act_guardian

def test_guard():
    m = torch.nn.Linear(10, 10)
    guard = act_guardian(m)
    m._act = torch.zeros(1, 10)
    guard()
    assert m._act.norm() > 0
