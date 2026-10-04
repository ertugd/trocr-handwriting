#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p data
curl -L --retry 5 -C - -o data/archive.zip \
  "https://www.kaggle.com/api/v1/datasets/download/teykaicong/iamondb-handwriting-dataset"
unzip -q -o data/archive.zip -d data
rm data/archive.zip
