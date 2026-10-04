# TrOCR Handwriting Recognition (IAM Words)

Fine-tuning and evaluation of [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) for handwritten word recognition on the IAM Handwriting Database.

## Requirements

- Python 3.9 or newer
- About 3 GB of free disk space for the data
- A GPU is recommended (NVIDIA CUDA or Apple Silicon). The code also runs on CPU, but training is very slow.

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The TrOCR model is downloaded from Hugging Face on first use.

## Data

1. Download the IAM word images from Kaggle (~830 MB):

   ```bash
   bash scripts/download_data.sh
   ```

   If the script fails, download the dataset manually from
   <https://www.kaggle.com/datasets/teykaicong/iamondb-handwriting-dataset>
   and put `words.tgz` and `xml.tgz` into the `data/` folder.

2. Create the train / validation / test files:

   ```bash
   python src/prepare_iam.py
   ```

   This writes `data/iam_words/train.csv`, `val.csv` and `test.csv` using the standard Aachen split in `splits/`.

## Usage

Evaluate the pretrained model:

```bash
python src/evaluate.py --name baseline
```

Fine-tune:

```bash
python src/train.py --run_name ft --epochs 2
```

Evaluate the fine-tuned model:

```bash
python src/evaluate.py --model outputs/ft/best --name ft --num_beams 4
```

Analyze errors:

```bash
python src/analyze_errors.py results/baseline_predictions.csv results/ft_predictions.csv
```

Run all experiments at once:

```bash
bash scripts/run_experiments.sh
```

### Training options

| Option | Description | Default |
|---|---|---|
| `--epochs` | Number of epochs | 3 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 2e-5 |
| `--keep_ratio` | Pad images to a square before resizing | off |
| `--augment` | Data augmentation | off |
| `--train_limit` | Use only N training samples (for quick tests) | all |

Your own data can be used with any CSV file that has `path` (image file) and `text` (transcription) columns:

```bash
python src/evaluate.py --model outputs/ft/best --csv my_data.csv --name my_data
```

## Project structure

```
src/
  prepare_iam.py      builds train/val/test CSV files
  common.py           dataset, preprocessing, metrics
  train.py            fine-tuning
  evaluate.py         evaluation (CER, WER, word accuracy)
  analyze_errors.py   error type analysis
scripts/
  download_data.sh    downloads the data
  run_experiments.sh  runs all experiments
splits/               Aachen split lists
results/              metrics of our runs
```

Created when running: `data/` (dataset), `outputs/` (trained models), `results/` (predictions and metrics).

## Results

IAM word test set (25,882 words):

| Model | CER | WER | Word accuracy |
|---|---|---|---|
| TrOCR-base-handwritten | 39.55% | 76.50% | 37.78% |
| Fine-tuned | 6.64% | 13.34% | 86.72% |
| Fine-tuned + beam search | 6.42% | 13.12% | 86.94% |

CER and WER: lower is better. Word accuracy: higher is better.

## Data license

The IAM Handwriting Database is free for non-commercial research. If you use it, please register at <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database> and cite:

> Marti, U.-V., & Bunke, H. (2002). The IAM-database: An English sentence database for offline handwriting recognition. *International Journal on Document Analysis and Recognition*, 5(1), 39–46.

The Aachen split lists are from [Laia](https://github.com/jpuigcerver/Laia).
