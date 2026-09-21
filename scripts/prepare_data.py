"""Download RSICD from the Hugging Face hub and write train/valid/test CSVs in the
format the rest of the code expects (captions as a stringified list, image as a
stringified {'bytes': ..., 'path': ...} dict)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

from datasets import Image, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset("arampacha/rsicd")
    for split in ("train", "valid", "test"):
        d = ds[split].cast_column("image", Image(decode=False))
        df = d.to_pandas()
        df["captions"] = df["captions"].apply(lambda c: str(list(c)))
        df["image"] = df["image"].apply(str)
        path = os.path.join(args.out, f"{split}.csv")
        df[["filename", "captions", "image"]].to_csv(path, index=False)
        print(f"{split:6} {len(df):5} rows -> {path}")


if __name__ == "__main__":
    main()
