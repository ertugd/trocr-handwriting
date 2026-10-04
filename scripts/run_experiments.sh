#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY="python -W ignore"
EPOCHS=${EPOCHS:-2}
mkdir -p logs results

step() { echo "=== $(date '+%H:%M') $*"; }

step "baseline"
$PY src/evaluate.py --name baseline --batch_size 64

step "train"
$PY src/train.py --run_name ft --epochs $EPOCHS
$PY src/train.py --run_name ft_ratio --keep_ratio --epochs $EPOCHS
$PY src/train.py --run_name ft_aug --augment --epochs $EPOCHS
$PY src/train.py --run_name ft_ratio_aug --keep_ratio --augment --epochs $EPOCHS

step "test"
$PY src/evaluate.py --model outputs/ft/best --name ft --batch_size 64
$PY src/evaluate.py --model outputs/ft/best --name ft_beam4 --num_beams 4 --batch_size 32
$PY src/evaluate.py --model outputs/ft_ratio/best --keep_ratio --name ft_ratio --batch_size 64
$PY src/evaluate.py --model outputs/ft_aug/best --name ft_aug --batch_size 64
$PY src/evaluate.py --model outputs/ft_ratio_aug/best --keep_ratio --name ft_ratio_aug --batch_size 64
$PY src/evaluate.py --model outputs/ft_ratio_aug/best --keep_ratio --name ft_ratio_aug_beam4 --num_beams 4 --batch_size 32

step "error analysis"
$PY src/analyze_errors.py results/{baseline,ft,ft_beam4,ft_ratio,ft_aug,ft_ratio_aug,ft_ratio_aug_beam4}_predictions.csv

step "done"
