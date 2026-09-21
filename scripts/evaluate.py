"""Compute BLEU-1..4, METEOR, length and repetition stats on the test split,
optionally on keyword-defined slices too."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from rscap.analysis import bleu4_on_slice, slice_by_keyword
from rscap.data import CaptionDataset, Vocabulary, collate_fn, ensure_nltk, load_split
from rscap.decoders import build_model
from rscap.encoders import FEATURE_DIMS
from rscap.train import compute_metrics, format_metrics, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="e.g. checkpoints/mobilenet_v2_transformer.pth")
    ap.add_argument("--data", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--slices", nargs="*", default=[], help="keywords, e.g. rails rivers cars")
    args = ap.parse_args()

    ensure_nltk()
    device = pick_device()
    cfg = json.load(open(args.checkpoint.replace(".pth", ".json")))
    vocab = Vocabulary.load(os.path.join(os.path.dirname(args.checkpoint), "vocab.pt"))
    model = build_model(cfg["decoder"], FEATURE_DIMS[cfg["backbone"]], len(vocab)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_df = load_split(os.path.join(args.data, "test.csv"))
    fdir = os.path.join(args.features, cfg["backbone"], "test")
    loader = DataLoader(CaptionDataset(test_df, vocab, fdir, cfg["max_len"]), batch_size=32, collate_fn=collate_fn)
    m = compute_metrics(model, loader, vocab, device, cfg["max_len"])
    print(f"== {cfg['backbone']} + {cfg['decoder']} on test ==\n{format_metrics(m)}")

    for kw in args.slices:
        sub = slice_by_keyword(test_df, kw)
        b4 = bleu4_on_slice(sub, model, vocab, fdir, cfg["max_len"], device)
        print(f"slice {kw!r:10} n={len(sub):4d}  BLEU-4 {b4:.4f}  ({b4 - m['BLEU-4']:+.4f} vs overall)")


if __name__ == "__main__":
    main()
