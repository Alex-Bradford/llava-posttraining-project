# inspect_sfte_vs_dpo.py
import os, glob, re, torch, json
from collections import defaultdict
from safetensors.torch import load_file

CKPT_A = "models/SFTe"        # plain SFT
CKPT_B = "models/SFTe_DPO"    # after DPO

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def load_state_dict(ckpt_dir):
    f = glob.glob(os.path.join(ckpt_dir, "adapter_model.*"))
    if not f:
        raise FileNotFoundError(f"no adapter_model.* in {ckpt_dir}")
    f = f[0]
    return torch.load(f, map_location="cpu") if f.endswith(".bin") else load_file(f, device="cpu")

def split_adapter_key(full_key):
    """
    Returns (adapter_name, param_key_without_prefix)

    • default (single‑adapter) key::
        base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        -> ('default', same_string)

    • multi‑adapter key::
        myAdapter.base_model.model.layers.0.*.lora_A.weight
        -> ('myAdapter',  string after 'myAdapter.')
    """
    m = re.match(r"([^.]*)\.(base_model\..*)", full_key)
    if m:
        name, param = m.groups()
        return ("default", full_key) if name == "base_model" else (name, param)
    # Fallback: treat as default
    return ("default", full_key)

def group_by_adapter(sd):
    bag = defaultdict(dict)
    for k,v in sd.items():
        if "lora_" not in k:        # skip non‑LoRA params
            continue
        ad, param = split_adapter_key(k)
        bag[ad][param] = v
    return bag

def pretty_delta(w1, w2):
    diff = (w1 - w2).abs()
    return diff.mean().item(), diff.max().item(), torch.linalg.norm(diff).item()

# ------------------------------------------------------------
# load & organise
# ------------------------------------------------------------
print("loading checkpoints …")
sd_A = load_state_dict(CKPT_A)
sd_B = load_state_dict(CKPT_B)

grp_A = group_by_adapter(sd_A)
grp_B = group_by_adapter(sd_B)

print("\n=== adapters discovered ===")
print("SFTe      :", sorted(grp_A.keys()))
print("SFTe_DPO  :", sorted(grp_B.keys()))

common = sorted(set(grp_A) & set(grp_B))
if not common:
    print("\nNo adapter names in common → nothing to compare.")
    raise SystemExit

# ------------------------------------------------------------
# compare
# ------------------------------------------------------------
for ad in common:
    left  = grp_A[ad]
    right = grp_B[ad]
    keys  = sorted(set(left) & set(right))
    if not keys:
        continue

    print(f"\n--- adapter: {ad}  (shared matrices: {len(keys)}) ---")
    print(f"{'parameter path':60s} |     meanΔ      maxΔ    frob‑norm")
    print("-"*100)
    for k in keys:
        m, M, fn = pretty_delta(left[k].float(), right[k].float())
        print(f"{k[:58]:60s} | {m:10.4e} {M:10.4e} {fn:10.4e}")
