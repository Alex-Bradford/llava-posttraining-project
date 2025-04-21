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
Taking into account the resource constraint, I have used QLoRA to run SFT on top of the base model. I have tested 4 different QLoRA configurations:
1. SFT1: r=16, alpha=16, dropout=0.05
2. SFT2: r=16, alpha=8, dropout=0.05
3. SFT3: r=16, alpha=16, dropout=0.10
4. SFT4: r=16, alpha=8, dropout=0.10

Using a consistent *r* hyperparamer for the QLoRA config means that all 4 adapters have the same architecture, which will allow me to easily take an average of the weights across all 4 configurations to test what I denote the **SFTe** (short for SFT ensemble) model.

1. Using QLoRA, test 4 different configurations


3. 

**Evaluation:**

This project investigates the effect of various post-training strategies on **LLaVA-1.5 7B**, a strong open-source vision-language model.

We develop and compare **three versions** of the model:
1. **Base Model**: Zero-shot, no post-training
2. **Benchmark Model**: Fine-tuned on curated instruction-image data using QLoRA
3. **Improved Model**: Fine-tuned with architectural and data-centric enhancements + ablation studies

The final results are summarized in a Google Slides presentation.

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
