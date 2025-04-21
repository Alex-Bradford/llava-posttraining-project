# ---- requirements -----------------------------------------------------------
# pip install safetensors            # if you don't already have it
# -----------------------------------------------------------------------------


import os, json, shutil, torch, glob
from collections import defaultdict
from safetensors.torch import load_file          # <-- new
from peft import PeftModel                       # only if you later want to test/merge

# ------------------------------------------------------------------
# 1)  Where are the checkpoints?
# ------------------------------------------------------------------
paths = [
    "models/SFT1/checkpoint-450",
    "models/SFT2/checkpoint-450",
    "models/SFT3/checkpoint-450",
    "models/SFT4/checkpoint-450",
]

out_dir = "models/SFTe"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------
# 2)  Helper to load a state‑dict (bin *or* safetensors)
# ------------------------------------------------------------------
def load_state_dict(ckpt_dir):
    # prefer safetensors if present
    cand = glob.glob(os.path.join(ckpt_dir, "adapter_model.*"))
    if not cand:
        raise FileNotFoundError(f"No adapter_model.[bin|safetensors] in {ckpt_dir}")
    file = cand[0]
    if file.endswith(".bin"):
        return torch.load(file, map_location="cpu")
    else:
        return load_file(file, device="cpu")

# ------------------------------------------------------------------
# 3)  Accumulate running sum on CPU
# ------------------------------------------------------------------
running_sum  = defaultdict(lambda: 0)
first_sd_ref = None
n = len(paths)

for idx, p in enumerate(paths, 1):
    sd = load_state_dict(p)

    if first_sd_ref is None:
        first_sd_ref = sd                       # keep for dtype & non‑LoRA params

    for k, v in sd.items():
        if "lora" in k:
            running_sum[k] += v.to(torch.float32)

    print(f"✔  loaded {idx}/{n}: {p}")

# ------------------------------------------------------------------
# 4)  Build averaged state‑dict
# ------------------------------------------------------------------
avg_sd = {}
for k, v in first_sd_ref.items():
    if "lora" in k:
        avg_sd[k] = (running_sum[k] / n).to(v.dtype)
    else:
        avg_sd[k] = v

# ------------------------------------------------------------------
# 5)  Copy adapter_config and save averaged weights as safetensors
# ------------------------------------------------------------------
shutil.copy2(os.path.join(paths[0], "adapter_config.json"),
             os.path.join(out_dir,  "adapter_config.json"))
torch.save(avg_sd, os.path.join(out_dir, "adapter_model.bin"))   # can also use save_file

print(f"\n🎉  Averaged adapter written to → {out_dir}")