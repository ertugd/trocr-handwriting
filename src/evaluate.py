import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from common import compute_metrics, configure_model, get_device, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--csv", default="data/iam_words/test.csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--keep_ratio", action="store_true")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = get_device()
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = configure_model(VisionEncoderDecoderModel.from_pretrained(args.model), processor).to(device)

    df["pred"] = predict(model, processor, df, device, keep_ratio=args.keep_ratio,
                         num_beams=args.num_beams, batch_size=args.batch_size)
    metrics = {"name": args.name, "model": args.model, "csv": args.csv, "keep_ratio": args.keep_ratio,
               "num_beams": args.num_beams, **compute_metrics(df["text"], df["pred"])}

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    df[["id", "text", "pred"]].to_csv(out / f"{args.name}_predictions.csv", index=False)
    (out / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
