# score_with_llava_critic.py
import re, os, copy, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import warnings

# ------------------------------------------------------------------
# 1)  critic model (loads once)
# ------------------------------------------------------------------
warnings.filterwarnings("ignore")
CRITIC_ID  = "lmms-lab/llava-critic-7b"
DEVICE     = "cuda"            # or "cpu"

model_name = "llava_qwen"   # ← auto‑detects "llava"
device_map = {"": DEVICE}

TOK, CRITIC, IMG_PROC, _ = load_pretrained_model(
    CRITIC_ID,
    None,
    model_name,
    device_map=device_map,
    offload_folder="./offload",          # optional
    attn_implementation="flash_attention_2",  # if FA2 is installed; otherwise drop
)
CRITIC.eval()

# ------------------------------------------------------------------
# 2)  data paths
# ------------------------------------------------------------------
CSV_IN      = "output_files\\evaluation_results.csv"
IMG_FOLDER  = "data\\images_test_dataset"
CSV_OUT     = "output_files\\evaluation_results_with_scores.csv"

df = pd.read_csv(CSV_IN)
resp_cols = [c for c in df.columns if c.startswith("clean_")]
print("Response columns found:", resp_cols)

# ------------------------------------------------------------------
# 3)  critic prompt template
# ------------------------------------------------------------------
CRITIC_TEMPLATE = (
    "Given an image and its question, serve as an unbiased judge to evaluate the "
    "quality of the answer provided by a Large Multimodal Model (LMM). "
    "Score the response on a scale of 0‑100 **just give the number first**, then "
    "a short justification.\n\n"
    "Question: [{question}]\n"
    "LMM response: [{response}]\n"
    "ASSISTANT:\n"
)

conv_template = "qwen_1_5"   # built‑in chat template for llava‑qwen critic

num_pat = re.compile(r"\b([0-9]{1,3})\b")   # first integer 0‑100

# ------------------------------------------------------------------
# 4)  helper to score one (img, question, answer) triple
# ------------------------------------------------------------------
@torch.inference_mode()
def score_triplet(image_path: str, question: str, answer: str) -> int | None:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], IMG_PROC, CRITIC.config)
    image_tensor = [_img.to(dtype=torch.float16, device=DEVICE) for _img in image_tensor]

    prompt_body = CRITIC_TEMPLATE.format(question=question, response=answer)
    prompt_full = DEFAULT_IMAGE_TOKEN + "\n" + prompt_body

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt_full)
    conv.append_message(conv.roles[1], None)
    prompt_tokens = tokenizer_image_token(conv.get_prompt(), TOK,
                                          IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = prompt_tokens.unsqueeze(0).to(DEVICE)

    out = CRITIC.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )
    text = TOK.batch_decode(out, skip_special_tokens=True)[0]
    m = num_pat.search(text)
    if m:
        val = int(m.group(1))
        return max(0, min(val, 100))      # clamp to [0,100]
    return None

# ------------------------------------------------------------------
# 5)  iterate rows and models
# ------------------------------------------------------------------
for col in resp_cols:
    model_tag = col.replace("clean_", "")
    score_col = f"score_{model_tag}"
    scores = []
    print(f"Scoring responses from column: {col}")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(IMG_FOLDER, f"{row['i']}.jpg")
        q        = row["prompt"]          # you can swap for a cleaner question if you have one
        a        = row[col]
        try:
            s = score_triplet(img_path, q, a)
        except Exception as e:
            print(f"[row {i}] error → {e}")
            s = None
        scores.append(s)
    df[score_col] = scores

# ------------------------------------------------------------------
# 6)  save
# ------------------------------------------------------------------
df.to_csv(CSV_OUT, index=False)
print(f"\n✓ critic scores written to {CSV_OUT}")
