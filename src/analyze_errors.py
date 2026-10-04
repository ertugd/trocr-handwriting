import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd

PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def norm(w: str) -> str:
    return PUNCT.sub("", str(w)).lower()


def build_lexicon(train_csv: str) -> set:
    words = pd.read_csv(train_csv)["text"].astype(str).map(norm)
    return {w for w in words if w}


def error_type(ref: str, pred: str, lexicon: set) -> str:
    if ref == pred:
        return "correct"
    if ref.lower() == pred.lower():
        return "case_only"
    if norm(ref) == norm(pred):
        return "punct_only"
    return "real_word" if norm(pred) in lexicon else "non_word"


def analyze(pred_csv: Path, lexicon: set) -> dict:
    df = pd.read_csv(pred_csv, keep_default_na=False)
    df["text"], df["pred"] = df["text"].astype(str), df["pred"].astype(str)
    df["type"] = [error_type(r, p, lexicon) for r, p in zip(df["text"], df["pred"])]

    alpha = df[df["text"].map(norm).str.len() > 0]
    oov = alpha[~alpha["text"].map(norm).isin(lexicon)]
    oov_wrong = oov[oov["type"] != "correct"]

    counts = df["type"].value_counts()
    errors = len(df) - counts.get("correct", 0)
    return {
        "file": pred_csv.name,
        "n": len(df),
        "accuracy": counts.get("correct", 0) / len(df),
        **{f"share_of_errors_{t}": counts.get(t, 0) / max(errors, 1)
           for t in ["case_only", "punct_only", "real_word", "non_word"]},
        "oov_n": len(oov),
        "oov_accuracy": (oov["type"] == "correct").mean() if len(oov) else float("nan"),
        "in_vocab_accuracy": (alpha[alpha["text"].map(norm).isin(lexicon)]["type"] == "correct").mean(),
        "oov_normalization_rate": (oov_wrong["type"] == "real_word").sum() / max(len(oov), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csvs", nargs="+")
    ap.add_argument("--train_csv", default="data/iam_words/train.csv")
    ap.add_argument("--out", default="results/error_analysis.csv")
    args = ap.parse_args()

    lexicon = build_lexicon(args.train_csv)
    rows = [analyze(Path(p), lexicon) for p in args.pred_csvs]
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.T.to_string(header=False))


if __name__ == "__main__":
    main()
