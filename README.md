# 🔍 Post-training open-source vision-language models with additional rounds of SFT, rejection sampling with self-critique, DPO and ablation studies.

## 🧑‍💻 Review of Existing Work

Before I dive into my specific project, I wanted to present my high level takeaways from reading each series of Llama, Qwen, DeepSeek, BLIP-2, and LLaVA (an open-source vision-language model) papers. I focussed on avenues for future research in post-training to create the best vision-language model. 

<ins>Task</ins>: Assume you are starting with a very powerful, pre-trained image encoder and LLM. Now, we want to create an adapter to connect their features/embeddings/encodings. How would you do it?

### Data

- For both pre-training and post-training, you want high density and diverse data. You could spend a lot of time optimising the deduplication, filtering and remixing strategies.
- A lot of papers just say “we improve the data” but the devil is really in the detail I would expect. Llama 3 had some good details on their exact methods.
- Post-training: optimise for real user prompts - make use of real user prompt data and augment with AI to generate additional synthetic data.
    - Again, you want high density and diverse data. Dedup, filter, remix.
    - Additional sources: Academic datasets and high quality, human curated examples

### Architecture

Take inspiration from:
- Flamingo: 
    - MLP to convert image tokens to same dimension as text tokens
    - Cross-attention layers to relate image tokens to text tokens
    - Allows image and text tokens to interleave in a prompt
- BLIP:
    - Additional loss functions could have potential
- <ins>MoE / Sparse models</ins>:
    - Investigate using MoE architecture for that adapter, maybe it makes sense that there is a different expert for different tasks (and their combinations:
        - Image type: document, nature, indoor setting, human recognition, etc.
        - Prompt type: general, creative, math, coding, etc.
     
### Pre-training the ViT-LLM adapter

- Data: focus on quality and breadth (probably same data as used in the image encoder). You want descriptive image-text pairs (not just 1 line text description).
- Training recipe considerations: simple vs complicated (eg. <ins>BLIP-2 has a complicated training recipe, can you make use of any ideas from that?</ins>)

### Post-training the ViT-LLM adapter

<ins>2 potential methods</ins>:
1. Post-train the adapter and freeze all other params, or
2. Lock everything except for the LLM, to focus on instruction following (same approach as done by Qwen2-VL (Oct 2024))

Training recipe considerations:
- SFT
- Train a reward model/critic with human annotated preference data
- Rejection sampling, winner chosen with critic, DPO
- <ins>Model averaging</ins>
    - Llama 3: Use model averaging across various checkpoints sourced from the RM, SFT and DPO stages (across various versions of data and hyper params).
        - Eg. You may have 10 versions of the RM trained with different data and hyperparams. Same for SFT and DPO. Average the weights across all 30 models.
    - This is a similar concept to the residual connections in the transformer block (eg. position encoded values are added back to the output from the attention mechanism, even though they were input into the attention mechanism).
 
### System prompting

<ins>Can we explore any behind-the-scenes system prompting logic to better guide the model to generating better responses?</ins>


## 🧪 Research Question

**Can we improve the performance of an already post-trained vision-language model (VLM) like LLaVA-1.5 with additional rounds of post-training including self-critique?**

---

## 🧠 Project Overview

Resource constraint: 10 hours of compute with a H100 Nvidia GPU.

**Data:**
- The LLaVA-1.5 model is already a post-trained VLM. I want to run SFT on unseen (although not out-of-scope) data at a manageable scale.
- The LLaVA project has compiled and shared the [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) in Aug 2024, it is a collection of 3.7m [image, prompt, response] triplets which they use for training.
- There is a subset of it called LRV Chart (1.8k rows), which I use as my dataset. I split this into train (1.6k) and test (0.2k).
- LLaVA 1.5 model was released in Oct 2023, and the specific LRV Chart subset was not mentioned as included data in te LLaVA-1.5 paper, so I believe this specific subset is unseen by the model.

**Method:**

Taking into account the resource constraint, I have used QLoRA to run SFT on top of the base model. This means that all base model weights are frozen and only a low-rank adapter is added on top of the base model and trained. I have tested 4 different QLoRA configurations:
- SFT1: r=16, alpha=16, dropout=0.05
- SFT2: r=16, alpha=8, dropout=0.05
- SFT3: r=16, alpha=16, dropout=0.10
- SFT4: r=16, alpha=8, dropout=0.10
- SFTe: an average of the above 4 model weights.

Using a consistent *r* hyperparamer for the QLoRA config means that all 4 adapters have the same architecture, which will allow me to easily take an average of the weights across all 4 configurations to test what I denote the **SFTe** (short for SFT ensemble) model. The *alpha* hyperparam controls how much influence the adapter has on the final model output (higher = more influence), and the *dropout* hyperparam controls regularisation (higher = more regularisation).

Then, I investigate if the model can self-critique without the need for fitting an explicit reward model. The specific process is:
1. Generate 2 outputs for a given [image, prompt] pair
2. Ask the SFTe model to pick the preferred response
3. Use this [image, prompt, chosen, rejected] data to run DPO on the SFTe model.

I denote this model **SFTe_DPO**, since DPO training has been run on top of the SFT ensemble model.

**Evaluation:**

I use an AI critic to evaluate the responses for each model on the test set. In October 2024, the LLaVA project released the [LLaVA-Critic](https://llava-vl.github.io/blog/2024-10-03-llava-critic/) model which has been specifically trained on 113k [image, prompt, response A, response B, critic evaluation] datapoints to be a generalist VLM evaluator. I present the average score across all datapoints in the test set for each model.

**Results:**

|                   | Base mod | SFT1   | SFT2    | SFT3    | SFT4    | SFTe    | SFTe_DPO |
|-------------------|----------|--------|---------|---------|---------|---------|----------|
| **Critic score**  | 41.34   | 42.97  | 40.67   | 38.24   | 36.91   | **43.62**   | 41.35    |

A few key observations:
- The higher *dropout* used in SFT1 vs SFT3 and SFT2 vs SFT4 led to degraded the performance.
- The higher *alpha* used in SFT3 vs SFT4 and SFT1 vs SFT2 led to increased performance.
- Despite the including the weaker models (SFT3 and SFT4) in the SFT ensemble, it still produced the best performance. Model averaging looks promising.
- The additional round of DPO led to degraded performance. It seems the LLaVA-1.5-7b model (with additional SFTe layer) is not powerful enough to critique itself and add value with additional DPO rounds. I would be interested to see if the same effect occurred for larger, more powerful models.

---

## 🚀 Steps to Reproduce Results

Infrastructure required: 1 NVIDIA H100 GPU

The below steps consume ~5hrs of compute time with 1 NVIDIA H100 GPU.

Steps to reproduce:
1. Create a env with Python 3.10 and the packages detailed in requirements.txt
2. Clone the repo.
3. Run 4 of your chosen SFT configurations using scripts.training.train_SFT_model.py, ensure all 4 models have the same *r* param for model averaging later on.
4. Run scripts.training.create_SFTe_model.py to create the SFTe model.
5. Run the scripts.training.train_DPO_model.py to generate the samples from the SFTe model, then again use SFTe to pick the preferred response, then run DPO on top of SFTe using that newly created preference data.
6. Run scripts.evaluation.generate_test_responses_and_metrics.py to generate responses for all models on the test set.
7. Run scripts.evaluation.evaluate_with_critic.py to calculate the critic scores.

---

## Future Work

### 1. Enhanced Post‑Training
- I’ve used QLoRA to do SFT and then DPO on the same adapter.  
- **Next step:** update **all model parameters**, not just the QLoRA adapter, for more powerful fine‑tuning.

### 2. Data Improvements
- Increase data **quality** (cleaner, better‑annotated examples).  
- Broaden data **breadth** (diverse image–text domains, more edge cases).

### 3. Zooming out, considering the entire LLaVA model
- They glue together a pre-trained LLM and a pre-trained ViT, by post-training an MLP adapter on instruction data. It’s simple and appears to work well. But perhaps some combination of the BLIP-2/Flamingo adapter is more powerful.
    - Llama 3 (and Qwen2-VL) weaves in the image encoded tokens into the VLM by using cross-attention layers after every 4th layer of self-attention in the LLM. For Llama 3, this increases the model params from 400B to 500B (eg. extra 100B params). Llama 3 runs pre-training on these cross-attention layers. Eg. they are trained on more than just instruction data (lots of image-text pairs, visual grounding data, screenshot parsing, some question-answer pairs, etc.)
    - BLIP-2 is very similar to Llama 3, but they use a “Q-Former” (instead of just cross-attention) and additional loss functions. Trained on more than just instruction data.
    - Flamingo is similar again, they allow for interleaved images and text prompts as they treat/inject/project an image token into the same dimensions as the text token. Again, they have a frozen LLM and frozen ViT, but they have trainable cross-attention layers and trainable projection matrices. They use more than just instruction data too. 
- Increase the scale of this adapter to make it more powerful
- Architecture and loss functions: consider a more sophisticated connection between the LLM and ViT (eg. BLIP-2)
- MoE adaptor, various experts for different tasks? (Charts vs landscape vs document, etc.?)
- Model averaging: average the weights across numerous checkpoints  eg. different SFT & DPO stages trained with various versions of data/hyperparams. (similar idea to residual connections in a transformer block, eg. allows the attention mechanism to solely focus on attention, instead of trying to maintain the position encoding too). 

---

## 🧑‍💻 Author

Built by Alex Bradford

---
