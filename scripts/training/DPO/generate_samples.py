'''
1. Use 

do_sample=True
top_k=40
temperature=1.0

with the benchmark model to generate different responses to same prompt on the train/val data.

2. Ask the benchmark model to score each response

3. Use that as input into DPO

'''

import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig
from torchvision.transforms.functional import to_pil_image
from peft import PeftModel
import copy
import io
import pandas as pd
from PIL import Image
from datasets import Dataset
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

CHAT_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}USER: {{ message['value']+'\n' }}{% elif message['from'] == 'gpt' %}ASSISTANT: {{ message['value']+'\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
DATA_PATH = "data/train.parquet"


def generate_samples():

    # --- Load tokenizer and processor from latest version ---
    model_id = "llava-hf/llava-1.5-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    processor.tokenizer = tokenizer  # link the tokenizer
    processor.tokenizer.padding_side = "left"

    # --- Load base (not fine-tuned) model ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        # force_download=True,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_config
    )
    base_model.eval()

    # Load base model (fresh)
    ft_base_model = copy.deepcopy(base_model)
    bm_model = PeftModel.from_pretrained(ft_base_model, "models/SFTe")
    bm_model = bm_model.merge_and_unload(safe_merge=True)
    bm_model.eval()

    for name, param in bm_model.named_parameters():
        if "lora" in name:
            # Cast the adapter weights to float16
            param.data = param.data.half()
            # Optionally print statistics to debug
            print(f"Converted {name} to half precision: mean = {param.data.mean():.4f}, std = {param.data.std():.4f}")

    # read parquet
    df = pd.read_parquet(DATA_PATH)
    raw_data = []
    for _, row in df.head(500).iterrows():
        raw_data.extend(expand_conversation_to_examples(row))
    dataset = Dataset.from_list(raw_data)

    print("🔄 Preprocessing dataset...")
    val_dataset = dataset.map(preprocess_example, remove_columns=["conversations", "target"], num_proc=1, fn_kwargs={"tokenizer": tokenizer, "processor": processor})

    # --- Select one sample ---
    results = []
    start_time = time.time()
    for i in range(0,len(val_dataset)):
        iter_start = time.time()
        temp_dataset = val_dataset
        
        example = temp_dataset[i]
        # reconstruct prompt
        prompt = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        # reconstruct target
        label_ids = torch.tensor(example["labels"])  # if needed
        cleaned_labels = label_ids.clone()
        if tokenizer.pad_token_id is not None:
            cleaned_labels[cleaned_labels == -100] = tokenizer.pad_token_id
        target = tokenizer.decode(cleaned_labels, skip_special_tokens=True)
        # reconstruct image
        example['image'].save("images_train_dataset/"+str(i)+".jpg")
        temp_dataset = temp_dataset.remove_columns(['image'])
        example = temp_dataset[i]
        device = base_model.device  # usually 'cuda:0'
        example = {
            k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, list) else v.unsqueeze(0).to(device)
            for k, v in example.items()
        }

        with torch.no_grad():
            output_bm1 = bm_model.generate(
                **example,
                max_new_tokens=200,
                do_sample=True,
                top_k=40,
                temperature=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            output_bm2 = bm_model.generate(
                **example,
                max_new_tokens=200,
                do_sample=True,
                top_k=40,
                temperature=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        response_bm1 = tokenizer.decode(output_bm1[0], skip_special_tokens=True)
        response_bm2 = tokenizer.decode(output_bm2[0], skip_special_tokens=True)
        clean_bm1 = extract_reply(response_bm1)
        clean_bm2 = extract_reply(response_bm2)

        # Append results for CSV (without metrics)
        results.append({
            "i": i,
            "prompt": prompt,
            "target": target,
            "clean_bm1": clean_bm1,
            "clean_bm2": clean_bm2
        })

        iter_end = time.time()
        iteration_time = iter_end - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining_iters = len(val_dataset) - (i + 1)
        eta = avg_time * remaining_iters
        if i % 20 == 0:
            print(f"Datapoint {i+1}/{len(val_dataset)} processed | Iteration: {iteration_time:.2f}s | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")

    # Output CSV file containing per-datapoint responses
    results_df = pd.DataFrame(results)
    results_df.to_csv("output_files\\candidate_DPO_samples.csv", index=False)
    print("CSV file saved as candidate_DPO_samples.csv")


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_reply(output):
    if "ASSISTANT:" in output:
        return output.split("ASSISTANT:")[-1].strip()
    return output.strip()


# --- Load saved deterministic train/val split ---
def expand_conversation_to_examples(row):
    image_bytes = row["image"]["bytes"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    conversations = row["conversations"]
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()

    if len(conversations) < 2 or conversations[-1]["from"] != "gpt":
        return []

    examples = []

    if conversations[0]["from"] == "human" and conversations[1]["from"] == "gpt":
        examples.append({
            "image": image,
            "conversations": [conversations[0]],
            "target": conversations[1]["value"]
        })

    return examples


def preprocess_example(example,tokenizer,processor):
    convo = example["conversations"]
    image = example["image"]

    # loop until we can fully encode the convo
    while True:
        prompt = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Prepare labels - tokenize the target
        target_text = example["target"] + tokenizer.eos_token
        target_ids = tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        )["input_ids"].squeeze(0)

        # pad the label
        labels = target_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

        reconstructed_input = tokenizer.decode(inputs['input_ids'], skip_special_tokens=True)

        # if it was fully encoded properly, break the loop
        if reconstructed_input.strip().endswith("ASSISTANT:"):
            break
        # else, drop the last 2 turns and repeat
        else:
            convo = convo[:-2]
    return inputs


if __name__ == '__main__':
    generate_samples()
