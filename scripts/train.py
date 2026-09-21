"""Train an LSTM or Transformer decoder on cached features."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rscap.data import PAD, CaptionDataset, Vocabulary, choose_max_length, collate_fn, ensure_nltk, load_split
from rscap.decoders import build_model
from rscap.encoders import FEATURE_DIMS
from rscap.train import pick_device, sample_captions, train_one_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="mobilenet_v2")
    ap.add_argument("--data", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--vocab-size", type=int, default=10000)
    args = ap.parse_args()

    ensure_nltk()
    device = pick_device()
    train_df = load_split(os.path.join(args.data, "train.csv"))
    val_df = load_split(os.path.join(args.data, "valid.csv"))

    vocab = Vocabulary().build(train_df["captions"], args.vocab_size)
    max_len = choose_max_length(train_df, vocab)
    print(f"vocab {len(vocab)} tokens, max_len {max_len}, device {device}")

    fdir = os.path.join(args.features, args.backbone)
    train_loader = DataLoader(CaptionDataset(train_df, vocab, f"{fdir}/train", max_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(CaptionDataset(val_df, vocab, f"{fdir}/valid", max_len), batch_size=args.batch_size, collate_fn=collate_fn)

    model = build_model(args.decoder, FEATURE_DIMS[args.backbone], len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    os.makedirs(args.out, exist_ok=True)
    name = f"{args.backbone}_{args.decoder}"
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, vocab, device)
        scheduler.step()
        print(f"epoch {epoch:2d}/{args.epochs}  loss {loss:.4f}  lr {optimizer.param_groups[0]['lr']:.0e}")
        for s in sample_captions(model, val_loader, vocab, device, max_len, n=2):
            print(f"   gen: {s['generated']}\n   ref: {s['references'][0]}")

    torch.save(model.state_dict(), os.path.join(args.out, f"{name}.pth"))
    vocab.save(os.path.join(args.out, "vocab.pt"))
    json.dump({"backbone": args.backbone, "decoder": args.decoder, "max_len": max_len}, open(os.path.join(args.out, f"{name}.json"), "w"))
    print(f"saved {args.out}/{name}.pth")


if __name__ == "__main__":
    main()
