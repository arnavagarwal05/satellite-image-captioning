"""Training loops and metric computation shared by the CLI scripts."""

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

from .data import PAD
from .decoders import LSTMDecoder, TransformerDecoderModel, causal_mask, padding_mask


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, vocab, device):
    """Works for both decoder types. Gradients are clipped to 1.0; without it the
    LSTM's caption loss swamps the image signal early in training."""
    model.train()
    total = 0.0
    is_tr = isinstance(model.decoder, TransformerDecoderModel)
    vocab_size = len(vocab)
    for features, captions, _ in tqdm(loader, desc="train", leave=False):
        features, captions = features.to(device), captions.to(device)
        optimizer.zero_grad()
        if is_tr:
            tgt_in, tgt_out = captions[:, :-1], captions[:, 1:]
            logits = model(features, tgt_in, causal_mask(tgt_in.size(1), device), padding_mask(tgt_in, vocab.stoi[PAD]))
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        else:
            logits = model(features, captions)
            loss = criterion(logits.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def generate(model, feature, vocab, max_len, beam_width=3):
    dec = model.decoder
    if isinstance(dec, LSTMDecoder):
        return dec.generate_caption_beam_search(feature, vocab, max_len=max_len, beam_width=beam_width)
    return dec.generate_caption(feature, vocab, max_len=max_len)


def sample_captions(model, loader, vocab, device, max_len, n=5):
    model.eval()
    features, _, refs = next(iter(loader))
    features = features.to(device)
    out = []
    for i in range(min(n, len(features))):
        out.append({"generated": " ".join(generate(model, features[i : i + 1], vocab, max_len)), "references": refs[i]})
    return out


@torch.no_grad()
def compute_metrics(model, loader, vocab, device, max_len):
    """Corpus BLEU-1..4, mean METEOR, generated-length stats, and the fraction of
    captions with a token repeated three or more times in a row."""
    model.eval()
    references, hypotheses, lengths, degenerate = [], [], [], 0
    for features, _, refs in tqdm(loader, desc="eval", leave=False):
        features = features.to(device)
        for i in range(features.size(0)):
            hyp = generate(model, features[i : i + 1], vocab, max_len)
            references.append([vocab.tokenizer(r) for r in refs[i]])
            hypotheses.append(hyp)
            lengths.append(len(hyp))
            degenerate += any(hyp[j] == hyp[j + 1] == hyp[j + 2] for j in range(len(hyp) - 2))
    w = lambda *ws: corpus_bleu(references, hypotheses, weights=ws)
    return {
        "BLEU-1": w(1, 0, 0, 0),
        "BLEU-2": w(0.5, 0.5, 0, 0),
        "BLEU-3": w(1 / 3, 1 / 3, 1 / 3, 0),
        "BLEU-4": w(0.25, 0.25, 0.25, 0.25),
        "METEOR": float(np.mean([meteor_score(r, h) for r, h in zip(references, hypotheses)])),
        "avg_len": float(np.mean(lengths)),
        "std_len": float(np.std(lengths)),
        "repetition_pct": 100.0 * degenerate / max(len(hypotheses), 1),
    }


def format_metrics(m):
    lines = [f"{k:<12} {v:.4f}" for k, v in m.items() if k.startswith(("BLEU", "METEOR"))]
    lines.append(f"{'length':<12} {m['avg_len']:.2f} ± {m['std_len']:.2f}")
    lines.append(f"{'repetition':<12} {m['repetition_pct']:.2f}%")
    return "\n".join(lines)
