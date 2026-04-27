# 🌐 English–Urdu Neural Machine Translation & Summarization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

*Domain-adaptive fine-tuning of MarianMT for low-resource English→Urdu translation,\ncombined with a BART-based cascade summarization pipeline.*

</div>

---

## 📌 Overview

This project investigates two core research questions in low-resource NLP:

1. **Translation Quality** — Can domain-adaptive fine-tuning of a pretrained MarianMT model on a conversational English–Urdu parallel corpus measurably improve translation quality?
2. **Urdu Summarization** — Can a cascade pipeline (BART abstractive summarization → MarianMT translation) produce semantically faithful Urdu summaries of English texts?

Both questions are answered affirmatively through rigorous evaluation using **BLEU**, **chrF**, **TER**, **ROUGE**, and **BERTScore** metrics.

---

## 🏗️ Pipeline Architecture

```
English Text
     │
     ▼
┌─────────────────────┐
│  BART-large-CNN     │  ← English Abstractive Summarization (Zero-shot)
│  facebook/bart-large│
└─────────────────────┘
     │ English Summary
     ▼
┌─────────────────────┐
│  MarianMT (En→Ur)   │  ← Fine-tuned on TED Talks En–Ur TMX corpus
│  Helsinki-NLP/      │     (13,577 sentence pairs, 5 epochs)
│  opus-mt-en-ur      │
└─────────────────────┘
     │
     ▼
Urdu Translation + Urdu Summary
```

---

## 📊 Key Results

### Translation Quality (RQ1)

| Metric | Baseline (Zero-shot) | Fine-tuned | Δ Change |
|--------|:--------------------:|:----------:|:--------:|
| BLEU ↑ | 15.20 | **16.28** | +1.08 ✅ |
| chrF ↑ | 37.97 | **41.56** | +3.59 ✅ |
| TER ↓ | 75.98 | **71.31** | −4.67 ✅ |

> **chrF (+3.59)** is the most meaningful gain — character-level evaluation is critical for morphologically rich Urdu script.

### Cascade Summarization Pipeline (RQ2)

| Metric | Score | Interpretation |
|--------|:-----:|----------------|
| Avg ROUGE-1 | 27.65 | Moderate lexical overlap |
| Avg ROUGE-2 | 8.17 | Expected for abstractive output |
| Avg ROUGE-L | 22.17 | Acceptable LCS coverage |
| **Avg BERTScore-F1** | **84.6** | **High semantic fidelity ✅** |

> BERTScore of **84.6** (above the 0.85 threshold) confirms the pipeline preserves meaning faithfully across all tested domains.

---

## 🔬 Research Question

> *To what extent does domain-adaptive fine-tuning of a pretrained neural machine translation model (MarianMT) on a conversational English–Urdu parallel corpus improve translation quality, and how effectively can a cascade pipeline — combining English abstractive summarization (BART) with the fine-tuned translation model — produce accurate Urdu summaries of English texts?*

**Answer:** Both claims are affirmed. See the [Research Discussion](#) section in the notebook for a full error analysis, ablation study, and contextualisation of scores against published benchmarks.

---

## 📁 Repository Structure

```
en-ur-nmt-summarization/
│
├── 📓 notebooks/
│   └── NLP_Project.ipynb          # Full implementation: training, eval, pipeline
│
├── 📄 paper/
│   ├── main.tex                   # LaTeX source for research paper
│   ├── references.bib             # Bibliography
│   └── figures/                   # Figures & plots for the paper
│
├── 📊 results/
│   └── (evaluation outputs, plots, metric logs)
│
├── 🖼️ assets/
│   └── (architecture diagrams, visualizations)
│
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: Tesla T4 or better)
- Google Colab (recommended environment)

### Installation

```bash
git clone https://github.com/MoazKashif/en-ur-nmt-summarization.git
cd en-ur-nmt-summarization
pip install -r requirements.txt
```

### Running the Notebook

1. Open `notebooks/NLP_Project.ipynb` in **Google Colab**
2. Upload your TMX parallel corpus file (e.g., `en-ur.tmx`) when prompted
3. Run all cells in order

The notebook is fully self-contained with the following sequential stages:

| Step | Description |
|------|-------------|
| 1 | Parse & clean TMX dataset |
| 2 | Fine-tune `Helsinki-NLP/opus-mt-en-ur` (MarianMT) |
| 3 | English summarization using `facebook/bart-large-cnn` |
| 4 | Urdu summarization via cascade pipeline |
| 5 | Full evaluation: BLEU, ROUGE, chrF, BERTScore |
| 6 | Error & ablation analysis |

---

## 🧩 Models Used

| Model | Role | Source |
|-------|------|--------|
| `Helsinki-NLP/opus-mt-en-ur` | En→Ur translation (fine-tuned) | HuggingFace Hub |
| `facebook/bart-large-cnn` | English abstractive summarization | HuggingFace Hub |

---

## 📦 Requirements

```txt
transformers>=4.35.0
datasets>=2.14.0
sentencepiece>=0.1.99
sacrebleu>=2.3.1
rouge-score>=0.1.2
bert-score>=0.3.13
accelerate>=0.24.0
evaluate>=0.4.0
sacremoses>=0.1.1
torch>=2.0.0
```

---

## 📈 Dataset

- **Source:** TED Talks English–Urdu TMX parallel corpus
- **Raw pairs:** ~15,086 sentence pairs
- **After cleaning:** ~13,577 sentence pairs
- **Train/Val split:** 90% / 10%
- **Filtering:** Min 3 tokens, max 100 tokens per English sentence

---

## 📝 Research Paper

The LaTeX source for the accompanying research paper is available in the [`paper/`](./paper/) directory. The paper covers:

- Motivation and related work in low-resource NMT
- Methodology and experimental setup
- Full results and statistical analysis
- Error analysis and future directions

> 🚧 *Paper manuscript — coming soon.*

---

## 👥 Authors

| Name | Roll No |
|------|---------|
| Student 1 | 23L-2626 |
| Student 2 | 23i-2635 |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for the opus-mt model family
- [Facebook AI Research](https://huggingface.co/facebook) for BART-large-CNN
- [HuggingFace Transformers](https://github.com/huggingface/transformers) library
- TED Talks corpus contributors for the En–Ur parallel data

---

<div align="center">

*Built with ❤️ for NLP research on low-resource language pairs*

</div>
