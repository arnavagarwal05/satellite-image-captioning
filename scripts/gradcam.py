"""Caption one test image end-to-end and show where the encoder looked."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from rscap.analysis import gradcam_for_row
from rscap.data import IMAGE_TRANSFORMS, Vocabulary, ensure_nltk, load_split
from rscap.decoders import FullEncoderDecoder, build_model
from rscap.encoders import FEATURE_DIMS, CNNEncoder
from rscap.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, help="filename substring, e.g. airport_355")
    ap.add_argument("--data", default="data")
    ap.add_argument("--save", help="write the figure here instead of showing it")
    args = ap.parse_args()

    ensure_nltk()
    device = pick_device()
    cfg = json.load(open(args.checkpoint.replace(".pth", ".json")))
    vocab = Vocabulary.load(os.path.join(os.path.dirname(args.checkpoint), "vocab.pt"))
    wrapper = build_model(cfg["decoder"], FEATURE_DIMS[cfg["backbone"]], len(vocab)).to(device)
    wrapper.load_state_dict(torch.load(args.checkpoint, map_location=device))
    full = FullEncoderDecoder(CNNEncoder(cfg["backbone"]).to(device), wrapper.decoder).eval()

    df = load_split(os.path.join(args.data, "test.csv"))
    rows = df[df["filename"].str.contains(args.image, na=False)]
    if rows.empty:
        raise SystemExit(f"no test image matching {args.image!r}")
    row = rows.iloc[0]
    img, overlay, words = gradcam_for_row(full, row, IMAGE_TRANSFORMS, vocab, device)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"generated: {' '.join(words)}\nreference: {row['captions'][0]}", fontsize=11)
    ax[0].imshow(img); ax[0].set_title(row["filename"]); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].set_title("Grad-CAM"); ax[1].axis("off")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
