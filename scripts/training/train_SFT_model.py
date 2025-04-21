import os
import io
import torch
import pandas as pd
import wandb
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, TrainingArguments,
    Trainer, BitsAndBytesConfig, default_data_collator
)
from peft.tuners.lora.config import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training

# --- Constants ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATA_PATH = "data/train.parquet"
SPLIT_PATH = "data/train_val_split_dataset"


OUTPUT_DIR = "models/SFT4/"
WANDB_RUN_ID = "llava-sft4-001"

# --- Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Tokenizer and Processor ---
CHAT_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}USER: {{ message['value']+'\n' }}{% elif message['from'] == 'gpt' %}ASSISTANT: {{ message['value']+'\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, force_download=True)
tokenizer.chat_template = CHAT_TEMPLATE
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, force_download=True)
processor.tokenizer = tokenizer
processor.tokenizer.padding_side = "left"

# --- Load Model ---
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    force_download=True,
    ignore_mismatched_sizes=True,
    quantization_config=config
)
model = prepare_model_for_kbit_training(model)

# --- Expand Conversations ---
def expand_conversation_to_dual_examples(row):
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

    truncated_convo = conversations[:-1]
    assistant_msg = conversations[-1]
    if truncated_convo[-1]["from"] == "human":
        examples.append({
            "image": image,
            "conversations": truncated_convo,
            "target": assistant_msg["value"]
        })

    return examples

# --- Load and Preprocess Dataset ---
df = pd.read_parquet(DATA_PATH)
dataset = []
for _, row in df.iterrows():
    dataset.extend(expand_conversation_to_dual_examples(row))

if os.path.exists(SPLIT_PATH):
    print(f"📂 Loading existing split from: {SPLIT_PATH}")
    hf_dataset = DatasetDict.load_from_disk(SPLIT_PATH)
else:
    print(f"📁 No saved split found — creating and saving to: {SPLIT_PATH}")
    hf_dataset = Dataset.from_list(dataset).train_test_split(test_size=0.1, seed=42)
    # hf_dataset.save_to_disk(SPLIT_PATH)

train_dataset = hf_dataset["train"]
val_dataset = hf_dataset["test"]

# --- Preprocessing Function ---
def preprocess_example(example):
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

print("🔄 Preprocessing dataset...")
train_dataset = train_dataset.map(preprocess_example, remove_columns=["image", "conversations", "target"], num_proc=1)
val_dataset = val_dataset.map(preprocess_example, remove_columns=["image", "conversations", "target"], num_proc=1)

# --- QLoRA ---
print("⚙️ Applying QLoRA...")
config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# --- Training Setup ---
print("🚀 Starting training...")
training_args = TrainingArguments(
    label_names=["labels"],
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_total_limit=5,
    fp16=True,
    report_to=["wandb"],
    run_name="llava-benchmark",
    # remove_unused_columns=False,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator
)


# --- Step 5: Train ---
print("--- Step 5: Train ---")
wandb.init(
    project="llava-benchmark",
    id=WANDB_RUN_ID,
    resume="allow"
)
trainer.train(resume_from_checkpoint=False)
# trainer.save_model(OUTPUT_DIR)
trainer.model.save_pretrained(OUTPUT_DIR)