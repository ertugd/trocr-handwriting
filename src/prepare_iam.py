import argparse
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image


def extract_if_needed(data_dir: Path, name: str) -> Path:
    target = data_dir / name
    if target.exists():
        return target
    archive = next(data_dir.rglob(f"{name}.tgz"))
    print(f"{archive} açılıyor...")
    target.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target)
    return target


def read_split_lists(split_dir: Path) -> dict:
    line_to_split = {}
    for split, fname in [("train", "aachen_tr.lst"), ("val", "aachen_va.lst"), ("test", "aachen_te.lst")]:
        for line_id in (split_dir / fname).read_text().split():
            line_to_split[line_id] = split
    return line_to_split


def collect_words(xml_dir: Path, words_dir: Path, line_to_split: dict) -> pd.DataFrame:
    rows = []
    for xml_file in sorted(xml_dir.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        writer = root.get("writer-id")
        for line in root.iter("line"):
            split = line_to_split.get(line.get("id"))
            if split is None:
                continue
            for word in line.iter("word"):
                wid = word.get("id")
                form_dir = "-".join(wid.split("-")[:2])
                path = words_dir / wid.split("-")[0] / form_dir / f"{wid}.png"
                rows.append({
                    "id": wid,
                    "path": str(path),
                    "text": word.get("text"),
                    "writer": writer,
                    "line_seg_ok": line.get("segmentation") == "ok",
                    "split": split,
                })
    return pd.DataFrame(rows)


def is_readable(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
            return im.size[0] > 1 and im.size[1] > 1
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--splits_dir", default="splits")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    words_dir = extract_if_needed(data_dir, "words")
    xml_dir = extract_if_needed(data_dir, "xml")
    line_to_split = read_split_lists(Path(args.splits_dir))

    df = collect_words(xml_dir, words_dir, line_to_split)
    print(f"XML'den okunan kelime: {len(df)}")

    df = df[df["path"].map(lambda p: Path(p).exists())]
    df = df[df["path"].map(is_readable)]
    df = df[df["text"].str.len() > 0]
    print(f"Görüntüsü sağlam kelime: {len(df)}")

    out_dir = data_dir / "iam_words"
    out_dir.mkdir(exist_ok=True)
    for split, part in df.groupby("split"):
        part.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"{split:5s}: {len(part):6d} kelime, {part['writer'].nunique():3d} yazar, "
              f"satır segmentasyonu hatalı: {(~part['line_seg_ok']).sum()}")


if __name__ == "__main__":
    main()
