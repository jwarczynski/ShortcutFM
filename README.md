# ShortcutFM
> Official Codebase for [*ShortcutFM*](to be added).

---

## 📜 License
This project is licensed under the **MIT License**. This ensures the code is open for academic and commercial use, provided that the original copyright notice is included.

## 🎯 Intended Use
This repository is designed to facilitate the **reproducibility** of the experiments described in our paper. It provides the complete pipeline, including:
* Model architecture definitions.
* Training scripts.
* Evaluation benchmarks.

## 🛠 Installation & Setup
This project uses `pyproject.toml` and `uv` for deterministic dependency management.

### Prerequisites
1. Install **uv** (if you haven't already):
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
Python version: 3.12 or higher (as specified in pyproject.toml).

Environment Setup
To install all dependencies (including exact versions from uv.lock):

Bash
uv sync
This will automatically create a virtual environment and install the required libraries.

## Dataset
Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use four datasets in our paper.

| Task | Datasets | Source |
|-|-|-|
| Open-domain Dialogue | CommonsenseConversation | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing)|
| Question Generation | Quasar-T |[download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing) |
| Text Simplification | Wiki-alignment | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing)|
| Paraphrase | QQP-Official |[download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing) |
| Machine Translation | iwslt14-de-en | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing)

## 🚀 How to Run
To run the main experiment or reproduction script, use:## Training
### training

```bash
# qqp:
uv run python -m shortcutfm configs/qqp.yaml
# others: modify learning_steps, dataset, data_dir, notes
```

### Decoding
```bash
uv run python -m shortcutfm configs/qqp-decode.yaml
# core parameters: step and td
```

### Evaluation
```bash
uv run python -m shortcutfm configs/qqp-eval.yaml
# you can eval single file or multiple file which are in the same folder (mbr in default)
```

## Citation
Please add the citation if our paper or code helps you.
```tex

```





