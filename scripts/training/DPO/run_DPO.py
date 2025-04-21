'''
1. Use 

do_sample=True
top_k=40
temperature=1.0

with the benchmark model to generate different responses to same prompt on the train/val data.

2. Ask the benchmark model to score each response

3. Use that as input into DPO

'''

import pandas as pd
from PIL import Image
from datasets import Dataset
import wandb

import torch
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import copy

OUTPUT_DIR = "models/SFTe_DPO2/"
WANDB_RUN_ID = "llava-SFTe_DPO2-001"

def run_DPO():
    # set up DPO data
    df = pd.read_csv('output_files\\candidate_DPO_samples.csv')
    df = df.iloc[::2, :].copy()
    df_preferred = pd.read_csv('output_files\\preferred_response.csv')
    df = df.merge(df_preferred,how='left',on='i')
    df = df[(df['response']=='response1') | (df['response']=='response2')].copy()
    raw_data = []
    for _, row in df.iterrows():
        raw_data.extend(format_row(row))
    dataset = Dataset.from_list(raw_data)

    """
    dataset should look like this:

    {'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=L size=980x812 at 0x154505570>],
    'prompt': 'User:<image>how many families?<end_of_utterance>\n',
    'rejected': 'Assistant: The image does not provide any information about families.<end_of_utterance>\n',
    'chosen': 'Assistant: The image shows a Union Organization table setup with 18,000 families.<end_of_utterance>\n'}
    """
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    # processor.tokenizer = tokenizer  # link the tokenizer
    # processor.tokenizer.padding_side = "left"
    tokenizer = processor.tokenizer

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
    base_model.config.use_cache = False

    # Load base model (fresh)
    model_for_training = copy.deepcopy(base_model)
    model_for_reference = copy.deepcopy(base_model)

    # Path to your adapter checkpoint
    adapter_path = "models/SFTe"

    # Load the adapter into each instance:
    # - The training model is set as trainable.
    # - The reference model is kept frozen (is_trainable=False).
    model_train = PeftModel.from_pretrained(
        model_for_training,
        adapter_path,
        is_trainable=True,
        adapter_name="training_adaptor"
    )
    model_ref = PeftModel.from_pretrained(
        model_for_reference,
        adapter_path,
        is_trainable=False,
        adapter_name="reference_adaptor"
    )

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        model_adapter_name="training_adaptor",
        ref_adapter_name="reference_adaptor",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        dataset_num_proc=1,  # tokenization will use 32 processes
        dataloader_num_workers=1,  # data loading will use 32 workers
        logging_steps=30,
        learning_rate=1e-04,
        force_use_ref_model=True
    )

    config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    trainer = DPOTrainer(
        model=model_train,
        ref_model=model_ref,  # not needed when using peft
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=config
    )

    # --- Step 5: Train ---
    print("--- Step 5: Train ---")
    wandb.init(
        project="llava-benchmark",
        id=WANDB_RUN_ID,
        resume="allow"
    )
    trainer.train()
    # trainer.save_model(OUTPUT_DIR)
    trainer.model.save_pretrained(OUTPUT_DIR)


def format_row(row):
    image = Image.open("data\\images_train_dataset\\"+str(row["i"])+".jpg").convert("RGB")

    prompt = 'User:<image>'+row['prompt'].split('\n')[1]+'<end_of_utterance>\n'
    if row['response']=='response1':
        chosen = 'Assistant: '+row['clean_bm1']+'<end_of_utterance>\n'
        rejected = 'Assistant: '+row['clean_bm2']+'<end_of_utterance>\n'
    else:
        chosen = 'Assistant: '+row['clean_bm2']+'<end_of_utterance>\n'
        rejected = 'Assistant: '+row['clean_bm1']+'<end_of_utterance>\n'

    examples = []
    examples.append({
        "images": image,
        "prompt": prompt,
        "rejected": rejected,
        "chosen": chosen,
    })

    return examples

if __name__ == "__main__":
    run_DPO()