"""Run a frozen ImageNet CNN over every image once and save one .pt feature per image.
Decoders train on these, so epochs take seconds instead of minutes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rscap.data import ImageOnlyDataset, load_split
from rscap.encoders import CNNEncoder
from rscap.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="mobilenet_v2")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="features")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = pick_device()
    enc = CNNEncoder(args.backbone).to(device).eval()
    for split in ("train", "valid", "test"):
        df = load_split(os.path.join(args.data, f"{split}.csv"))
        out_dir = os.path.join(args.out, args.backbone, split)
        os.makedirs(out_dir, exist_ok=True)
        loader = DataLoader(ImageOnlyDataset(df), batch_size=args.batch_size)
        with torch.no_grad():
            for images, ids in tqdm(loader, desc=f"{args.backbone}/{split}"):
                feats = enc(images.to(device)).cpu()
                for f, i in zip(feats, ids):
                    torch.save(f, os.path.join(out_dir, f"{i}.pt"))


if __name__ == "__main__":
    main()
