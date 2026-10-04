import random

import jiwer
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_to_square(img: Image.Image, fill=255) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class HandwritingAugment:
    def __init__(self, p: float = 0.7):
        self.p = p
        self.affine = transforms.RandomAffine(
            degrees=3, translate=(0.03, 0.05), scale=(0.9, 1.05), shear=(-15, 15), fill=255
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img = self.affine(img)
        r = random.random()
        if r < 0.2:
            img = img.filter(ImageFilter.MinFilter(3))
        elif r < 0.4:
            img = img.filter(ImageFilter.MaxFilter(3))
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        return img


def load_image(path: str, keep_ratio: bool) -> Image.Image:
    img = ImageOps.grayscale(Image.open(path))
    if keep_ratio:
        img = pad_to_square(img)
    return img


class HandwritingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, keep_ratio=False, augment=False, max_target_len=32):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.keep_ratio = keep_ratio
        self.augment = HandwritingAugment() if augment else None
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row["path"], self.keep_ratio)
        if self.augment:
            img = self.augment(img)
        pixel_values = self.processor(images=img.convert("RGB"), return_tensors="pt").pixel_values[0]

        tok = self.processor.tokenizer
        ids = tok(str(row["text"]), add_special_tokens=False).input_ids[: self.max_target_len - 1]
        labels = ids + [tok.sep_token_id]
        labels += [-100] * (self.max_target_len - len(labels))
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def configure_model(model, processor):
    tok = processor.tokenizer
    model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id or tok.sep_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gen = model.generation_config
    gen.decoder_start_token_id = model.config.decoder_start_token_id
    gen.pad_token_id = tok.pad_token_id
    gen.eos_token_id = tok.sep_token_id
    gen.max_new_tokens = 32
    return model


@torch.no_grad()
def predict(model, processor, df, device, keep_ratio=False, num_beams=1, batch_size=32, progress=True):
    from tqdm import tqdm

    model.eval()
    preds = []
    idxs = range(0, len(df), batch_size)
    for start in tqdm(idxs, desc="tahmin", disable=not progress):
        batch = df.iloc[start:start + batch_size]
        images = [load_image(p, keep_ratio).convert("RGB") for p in batch["path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values, num_beams=num_beams)
        preds.extend(processor.batch_decode(out, skip_special_tokens=True))
    return [p.strip() for p in preds]


def compute_metrics(refs, preds) -> dict:
    refs = [str(r) for r in refs]
    preds = [str(p) for p in preds]
    return {
        "CER": jiwer.cer(refs, preds),
        "WER": jiwer.wer(refs, preds),
        "word_accuracy": sum(r == p for r, p in zip(refs, preds)) / len(refs),
        "n": len(refs),
    }
