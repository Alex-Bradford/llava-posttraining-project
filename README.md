# 🔍 Post-Training LLaVA-1.5 for Real-World Task Understanding

A research-style portfolio project exploring how to improve the instruction-following capabilities of vision-language models through post-training, evaluation, and ablation.

---

## 🧠 Project Overview

This project investigates the effect of various post-training strategies on **LLaVA-1.5 7B**, a strong open-source vision-language model.

We develop and compare **three versions** of the model:
1. **Base Model**: Zero-shot, no post-training
2. **Benchmark Model**: Fine-tuned on curated instruction-image data using QLoRA
3. **Improved Model**: Fine-tuned with architectural and data-centric enhancements + ablation studies

The final results are summarized in a Google Slides presentation.

---

## 🧪 Research Question

**Can we enhance the instruction-following and visual reasoning capabilities of a VLM like LLaVA-1.5 using principled fine-tuning methods and targeted improvements — and quantify those gains through fair, task-aligned evaluation?**

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
