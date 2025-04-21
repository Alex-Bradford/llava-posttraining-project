'''
Outputs metrics: loss, bleu and rouge

Also outputs the responses from all models for all test data points

'''

# evaluate_multi_adapters.py
import torch, copy, io, time, pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
)
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# -----------------  CONFIG  -----------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
EXPERIMENTS = [                           # nickname,   checkpoint‑dir
    ("SFT1",        "models/SFT1/checkpoint-450"),
    ("SFT2",        "models/SFT2/checkpoint-450"),
    ("SFT3",        "models/SFT3/checkpoint-450"),
    ("SFT4",        "models/SFT4/checkpoint-450"),
    ("SFTe",        "models/SFTe"),
    ("SFTe_DPO2",    "models/SFTe_DPO2"),
]
DATA_PATH  = "data\\test.parquet"
OUT_CSV    = "output_files\\evaluation_results.csv"
CHAT_TEMPLATE = """{% for m in messages %}{% if m['from']=='human' %}USER: {{m['value']+'\n'}}{% elif m['from']=='gpt' %}ASSISTANT: {{m['value']+'\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
# -------------------------------------------

# ---------- tokenizer / processor ----------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.chat_template = CHAT_TEMPLATE
proc = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
proc.tokenizer = tok
proc.tokenizer.padding_side = "left"

# ---------------- base model ---------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb_cfg,
    ignore_mismatched_sizes=True).eval()

# ----------- helper: load & merge ----------
def load_merged_adapter(base, ckpt):
    model = PeftModel.from_pretrained(copy.deepcopy(base), ckpt, is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True).eval()
    for n, p in merged.named_parameters():          # cast LoRA halves to fp16
        if "lora" in n:
            p.data = p.data.half()
    return merged

# ---------- load every experiment ----------
models = [("Base", base_model)]
for nick, path in EXPERIMENTS:
    print(f"→ loading {nick} from {path}")
    models.append((nick, load_merged_adapter(base_model, path)))

# ---------- dataset prep (unchanged) -------
def expand_conversation(row):
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    convo = row["conversations"]
    if hasattr(convo, "tolist"):
        convo = convo.tolist()
    if len(convo) < 2 or convo[-1]["from"] != "gpt":
        return []
    examples = []
    if convo[0]["from"] == "human" and convo[1]["from"] == "gpt":
        examples.append({"image": img, "conversations": [convo[0]], "target": convo[1]["value"]})
    # if convo[-2]["from"] == "human":
    #     examples.append({"image": img, "conversations": convo[:-1], "target": convo[-1]["value"]})
    return examples

df = pd.read_parquet(DATA_PATH)
raw = [ex for _, r in df.iterrows() for ex in expand_conversation(r)]
dataset = Dataset.from_list(raw)

def prep(ex):
    convo = ex["conversations"]
    img   = ex["image"]
    while True:
        prompt = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        ins = proc(text=prompt, images=img, return_tensors="pt",
                   padding="max_length", truncation=True, max_length=1024)
        ins = {k: v.squeeze(0) for k,v in ins.items()}
        tgt_ids = tok(ex["target"]+tok.eos_token, return_tensors="pt",
                      padding="max_length", truncation=True, max_length=1024)["input_ids"].squeeze(0)
        labels = tgt_ids.clone()
        labels[labels == tok.pad_token_id] = -100
        ins["labels"] = labels
        if tok.decode(ins["input_ids"], skip_special_tokens=True).strip().endswith("ASSISTANT:"):
            return ins
        convo = convo[:-2]

print("🔄 preprocessing …")
val_ds = dataset.map(prep, remove_columns=["conversations","target"], num_proc=1)

# ------------- evaluation loop -------------
smoothie = SmoothingFunction().method4
scorer   = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

agg = {nick: {"loss":0,"bleu":0,"rouge":0} for nick,_ in models}
results = []
start   = time.time()

for i in range(len(val_ds)):
    ex = val_ds[i]
    prompt_txt = tok.decode(ex["input_ids"], skip_special_tokens=True)
    tgt_ids    = torch.tensor(ex["labels"])
    tgt_ids[tgt_ids == -100] = tok.pad_token_id
    target_txt = tok.decode(tgt_ids, skip_special_tokens=True)

    # image to disk (optional; comment out if not needed)
    ex["image"].save(f"data\\images_test_dataset\\{i}.jpg")

    # move tensors to the first model’s device (GPU 0)
    dev = models[0][1].device
    batch = {k: torch.tensor(v).unsqueeze(0).to(dev) if isinstance(v,list) else v.unsqueeze(0).to(dev)
             for k,v in ex.items() if k!="image"}

    row = {"i":i, "prompt":prompt_txt, "target":target_txt}
    for nick, model in models:
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=200, do_sample=True, temperature=0.5,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            out_txt = tok.decode(gen[0], skip_special_tokens=True)
            reply = out_txt.split("ASSISTANT:",1)[-1].strip()

            loss = model(**batch).loss.item()
            bleu = sentence_bleu([target_txt.split()], reply.split(), smoothing_function=smoothie)
            rouge = scorer.score(target_txt, reply)['rougeL'].fmeasure

        agg[nick]["loss"]  += loss
        agg[nick]["bleu"]  += bleu
        agg[nick]["rouge"] += rouge
        row[f"clean_{nick}"] = reply

    results.append(row)

    if i % 20 == 0:
        done = i+1
        elapsed = time.time()-start
        print(f"{done}/{len(val_ds)} done • elapsed {elapsed/60:.1f} min")

# ---------- aggregate & print -------------
n = len(val_ds)
print("\n=== aggregate metrics ===")
for nick,_ in models:
    m = agg[nick]
    print(f"{nick:>10}  |  loss {m['loss']/n:.4f}  "
          f"BLEU {m['bleu']/n:.4f}  ROUGE‑L {m['rouge']/n:.4f}")

# -------------- CSV ----------------------
pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"\nCSV written ➜ {OUT_CSV}")
