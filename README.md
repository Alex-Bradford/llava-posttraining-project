# 🔍 Post-training open-source vision-language models with additional rounds of SFT, rejection sampling with self-critique, DPO and ablation studies.

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

For completeness, I also include the loss (on the test set), BLEU and ROUGE-L scores.

**Results:**

|                   | Base mod | SFT1   | SFT2    | SFT3    | SFT4    | SFTe    | SFTe_DPO |
|-------------------|----------|--------|---------|---------|---------|---------|----------|
| **Loss**          | 11.65    | **10.94**  | 11.38   | <u>11.11</u>   | 11.36   | 11.60   | 11.65    |
| **BLEU**          | **0.1138**   | 0.077  | 0.1045  | 0.0817  | <u>0.1093</u>  | 0.0947  | 0.1005   |
| **ROUGE‑L**       | **0.2885**   | 0.2409 | <u>0.2826</u>  | 0.2617  | 0.273   | 0.2791  | 0.2653   |
| **Critic score**  | <u>41.34</u>    | 42.97  | 40.67   | 38.24   | 36.91   | **43.62**   | <u>41.35</u>    |
<u>0.1045</u>
---

## 🏗️ Project Structure

| Milestone | Description |
|----------|-------------|
| 1 | Setup & baseline inference on LLaVA-1.5 |
| 2 | Curation of a high-quality multimodal dataset |
| 3 | QLoRA fine-tuning to create the benchmark model |
| 4 | Evaluation suite: prompts, rubric, auto-grading |
| 5 | Improved model: ablation studies + new techniques |
| 6 | Final evaluation and comparison of all 3 models |

---

## 📁 Dataset

- ~1000 instruction-image-response triplets
- Topics include: cooking, repair, travel, fashion, everyday objects
- Formats: JSON (structured), PNG/JPEG (image)
- Metadata: domain, ambiguity score, prompt type

---

## 🧪 Evaluation Suite

We created a held-out test set of 100 examples, scored using:
- ✅ Correctness
- 🧭 Steerability
- 💬 Helpfulness
- ❌ Hallucination Avoidance
- ⚠️ Safety/Refusal when appropriate

Scores were generated using a combination of:
- GPT-4 scoring (auto-eval)
- Manual review on a small subset
- Visualization: bar charts, side-by-sides

---

## 🛠️ Tools & Libraries

| Category         | Tools / Frameworks                        |
|------------------|-------------------------------------------|
| Model            | `llava-hf/llava-1.5-7b-hf`                |
| Fine-tuning      | `transformers`, `peft`, `bitsandbytes`    |
| Data prep        | `datasets`, `Pillow`, `OpenAI`            |
| Logging          | `Weights & Biases`                        |
| Evaluation       | `pandas`, `matplotlib`, `OpenAI GPT-4`    |
| Slides           | Google Slides (exported backup in `/slides`) |

---

## 📌 Acknowledgments

- LLaVA GitHub Repo
- Hugging Face Transformers
- OpenAI GPT-4 (for evaluation)

---

## 🧑‍💻 Author

Built by Alex Bradford

---

## 🚀 Setup instructions

# Create a new conda env with Python 3.10
conda create -n llava-env python=3.10
conda activate llava-env
# Install PyTorch with correct CUDA (e.g., 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Add Hugging Face and other dependencies
pip install transformers accelerate peft bitsandbytes matplotlib pillow




# Run base model inference
python notebooks/01_base_inference.ipynb

# Fine-tune benchmark model
python scripts/train_benchmark.py --config configs/benchmark.yaml

# Run evaluation
python scripts/evaluate.py --model benchmark

# Fine-tune improved model
python scripts/train_improved.py --config configs/improved_loss.yaml
