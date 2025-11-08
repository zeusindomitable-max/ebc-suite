from ebc_suite import mem_safe_forward
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model.eval()

x = tok("hello " * 60000, return_tensors="pt").input_ids
with torch.no_grad():
    out = mem_safe_forward(model, x, chunk=8192, checkpoint=False)
print(out.shape)
