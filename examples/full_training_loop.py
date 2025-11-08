from ebc_suite import *
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
opt = torch.optim.AdamW(model.parameters(), lr=0)

clip = energy_budget_clip(model)
guard = act_guardian(model)
T = 1000

for step in range(T):
    loss = model(torch.randint(0, 1000, (1, 512))).loss
    loss.backward()
    clip(); guard(); lr_auto(opt, step, T); opt.step(); opt.zero_grad()

