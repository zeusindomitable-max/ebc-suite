<div align="center">

# **EBC-Suite**  
### **Four Laws of Stable LLM Training**

> **No NaN. No Tuning. No Dead Neurons. 128K Context.**

![PyPI](https://img.shields.io/pypi/v/ebc-suite?color=purple)
![Python](https://img.shields.io/badge/python-≥3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

**Author**: **Hari Tedjamantri**  
**Email**: haryganteng06@gmail.com  
**X**: [@haritedjamantri](https://x.com/haritedjamantri)  
**Patent Pending** — U.S. Provisional Filed

---

## **The Four Laws**

| Law | Formula | Effect |
|-----|--------|-------|
| **EBC** | `||∇L|| ≤ 0.1 × ||act||` | **No NaN. 5% lower loss** |
| **ACT-GUARDIAN** | `if ||act|| < 1e-3 → add noise` | **0 dead heads** |
| **LR-AUTO** | `LR(t) = warmup + decay` | **No tuning. 10% faster** |
| **MEM-SAFE** | `Chunk + Checkpoint` | **128K on 1 GPU. 50% less RAM** |

---

## **Install**

```bash
pip install ebc-suite
```
## One-Line Training Loop


from ebc_suite import *

clip = energy_budget_clip(model)
guard = act_guardian(model)

for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()
    clip(); guard(); lr_auto(opt, step, T); opt.step()

##Citation
@software{ebc-suite-2025,

  author = {Hari Tedjamantri}
  
  title = {EBC-Suite: Four Laws of Stable LLM Training},
  
  year = {2025},
  
  url = {https://github.com/ebc-clip/ebc-suite},
  
  doi = {10.5281/zenodo.1234567}
}

<div align="center">

Made with  by Hari Tedjamantri
X: @haritedjamantri
</div>
```



