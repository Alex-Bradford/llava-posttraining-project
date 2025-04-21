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
from peft import PeftModel
import copy
import pandas as pd
from PIL import Image
from datasets import Dataset
import time
import torch.nn.functional as F

CHAT_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}USER: {{ message['value']+'\n' }}{% elif message['from'] == 'gpt' %}ASSISTANT: {{ message['value']+'\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

def pick_preferred_sample():
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


    df = pd.read_csv('output_files\\candidate_DPO_samples.csv')
    raw_data_1 = []
    raw_data_2 = []
    for _, row in df.iterrows():
        raw_data_1.extend(format_row(row,method=1))
        raw_data_2.extend(format_row(row,method=2))
    dataset_1 = Dataset.from_list(raw_data_1)
    dataset_2 = Dataset.from_list(raw_data_2)
    val_dataset_1 = dataset_1.map(preprocess_example2, remove_columns=["conversations"], num_proc=1,fn_kwargs={"tokenizer": tokenizer, "processor": processor})
    val_dataset_1 = val_dataset_1.remove_columns(['image'])
    val_dataset_2 = dataset_2.map(preprocess_example2, remove_columns=["conversations"], num_proc=1,fn_kwargs={"tokenizer": tokenizer, "processor": processor})
    val_dataset_2 = val_dataset_2.remove_columns(['image'])

    # --- Select one sample ---
    results = []
    start_time = time.time()
    for i in range(0,len(val_dataset_1)):
        iter_start = time.time()
        device = base_model.device  # usually 'cuda:0'
        
        example_1 = val_dataset_1[i]
        example_1 = {
            k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, list) else v.unsqueeze(0).to(device)
            for k, v in example_1.items()
        }

        example_2 = val_dataset_2[i]
        example_2 = {
            k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, list) else v.unsqueeze(0).to(device)
            for k, v in example_2.items()
        }

        with torch.no_grad():
            output_bm_1 = bm_model.generate(
                **example_1,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            output_bm_2 = bm_model.generate(
                **example_2,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # --- Decode outputs ---
        def extract_reply(output):
            if "ASSISTANT:" in output:
                return output.split("ASSISTANT:")[-1].strip()
            return output.strip()

        response_bm_1 = tokenizer.decode(output_bm_1[0], skip_special_tokens=True)
        clean_bm_1 = extract_reply(response_bm_1)

        response_bm_2 = tokenizer.decode(output_bm_2[0], skip_special_tokens=True)
        clean_bm_2 = extract_reply(response_bm_2)

        if (clean_bm_1 == clean_bm_2) and (clean_bm_1 == 'response X'):
            response = 'response1'
        elif (clean_bm_1 == clean_bm_2) and (clean_bm_1 == 'response Y'):
            response = 'response2'
        else:
            response = "neither"

        # Append results for CSV (without metrics)
        results.append({
            "i": i,
            "response": response,
            "Xfirst_response": clean_bm_1,
            "Yfirst_response": clean_bm_2,
        })

        iter_end = time.time()
        iteration_time = iter_end - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining_iters = len(val_dataset_1) - (i + 1)
        eta = avg_time * remaining_iters
        if i % 50 == 0:
            print(f"Datapoint {i+1}/{len(val_dataset_1)} processed | Iteration: {iteration_time:.2f}s | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")

    # Output CSV file containing per-datapoint responses
    results_df = pd.DataFrame(results)
    results_df.to_csv("output_files\\preferred_response.csv", index=False)
    print("CSV file saved as preferred_response.csv")






def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_row(row,method):
    image = Image.open("data\\images_train_dataset\\"+str(row["i"])+".jpg").convert("RGB")

    prompt = '\"'+row['prompt'].split('\n')[1]+'\"'
    response1 = '\"'+row['clean_bm1']+'\"'
    response2 = '\"'+row['clean_bm2']+'\"'
    if method==1:
        conversations = [{'from': 'human', 'value': '<image>\nConsider the image, prompt and responses. \nprompt: '+prompt+'\nresponse X: '+response1+'\nresponse Y: '+response2+'\nIs response X or response Y better? Answer in 1 word.'}]
    elif method==2:
        conversations = [{'from': 'human', 'value': '<image>\nConsider the image, prompt and responses. \nprompt: '+prompt+'\nresponse Y: '+response2+'\nresponse X: '+response1+'\nIs response Y or response X better? Answer in 1 word.'}]

    examples = []
    examples.append({
        "image": image,
        "conversations": conversations
    })

    return examples

def preprocess_example2(example,tokenizer,processor):
    convo = example["conversations"]
    image = example["image"]

    # loop until we can fully encode the convo
    prompt = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    prompt = prompt + 'response'

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs

if __name__ == '__main__':
    pick_preferred_sample()