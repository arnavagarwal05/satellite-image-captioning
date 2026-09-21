"""Dataset loading, caption cleaning, vocabulary, and feature-cached datasets for RSICD."""

import ast
import io
import os
import random
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def ensure_nltk():
    for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean_and_split_captions(captions_str):
    """RSICD stores captions as a stringified list whose elements sometimes run several
    sentences together. Flatten to one string, split on periods, drop empties."""
    try:
        captions_list = ast.literal_eval(captions_str)
    except (ValueError, SyntaxError):
        return []
    full_text = " ".join(captions_list)
    return [s.strip() for s in full_text.split(".") if s.strip()]


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    df["captions"] = df["captions"].apply(clean_and_split_captions)
    return df


def decode_image(row):
    """The image column holds a stringified dict {'bytes': b'...', 'path': ...}."""
    image_bytes = ast.literal_eval(row["image"])["bytes"]
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_id(row):
    return os.path.splitext(os.path.basename(row["filename"]))[0]


class Vocabulary:
    def __init__(self):
        self.itos = {0: PAD, 1: BOS, 2: EOS, 3: UNK}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        """Lowercase, word-tokenize, then split any token that still glues letters to
        punctuation ("boats.a" -> ["boats", "a"]). RSICD has many such captions."""
        tokens = []
        for tok in nltk.tokenize.word_tokenize(text.lower()):
            if tok.isalpha() or tok == "n't":
                tokens.append(tok)
            else:
                tokens.extend(re.findall(r"[a-z]+", tok))
        return tokens

    def build(self, captions_series, vocab_size=10000):
        freq = Counter()
        for caption_list in captions_series:
            for sentence in caption_list:
                freq.update(self.tokenizer(sentence))
        idx = len(self.itos)
        for word, _ in freq.most_common(vocab_size - len(self.itos)):
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
        return self

    def numericalize(self, text):
        ids = [self.stoi[BOS]]
        ids.extend(self.stoi.get(t, self.stoi[UNK]) for t in self.tokenizer(text))
        ids.append(self.stoi[EOS])
        return ids

    def decode(self, ids):
        return [self.itos[i] for i in ids if i not in (self.stoi[BOS], self.stoi[EOS], self.stoi[PAD])]

    def save(self, path):
        torch.save({"itos": self.itos}, path)

    @classmethod
    def load(cls, path):
        v = cls()
        v.itos = {int(k): w for k, w in torch.load(path)["itos"].items()}
        v.stoi = {w: k for k, w in v.itos.items()}
        return v


def caption_length_stats(df, vocab):
    lengths = [len(vocab.tokenizer(s)) for caps in df["captions"] for s in caps]
    p90, p95, p98 = (int(np.percentile(lengths, p)) for p in (90, 95, 98))
    return {"lengths": lengths, "p90": p90, "p95": p95, "p98": p98}


def choose_max_length(df, vocab, slack=2):
    """98th percentile of token length, plus BOS/EOS, plus a little slack."""
    return caption_length_stats(df, vocab)["p98"] + 2 + slack


def vocab_coverage(df, vocab):
    total = oov = 0
    for caps in df["captions"]:
        for s in caps:
            toks = vocab.tokenizer(s)
            total += len(toks)
            oov += sum(t not in vocab.stoi for t in toks)
    return {"coverage": 100 * (1 - oov / total), "oov_pct": 100 * oov / total}


class ImageOnlyDataset(Dataset):
    """Yields (image_tensor, image_id). Used to precompute encoder features."""

    def __init__(self, df, transform=IMAGE_TRANSFORMS):
        self.df, self.transform = df, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.transform(decode_image(row)), image_id(row)


class CaptionDataset(Dataset):
    """Yields (cached_feature, padded_caption_ids, all_reference_captions).

    One reference caption is sampled per item each epoch, so every pass sees a
    different pairing of image and caption."""

    def __init__(self, df, vocab, feature_dir, max_len):
        self.df, self.vocab, self.feature_dir, self.max_len = df, vocab, feature_dir, max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature = torch.load(os.path.join(self.feature_dir, f"{image_id(row)}.pt"))
        captions = row["captions"] or ["a remote sensing image"]
        ids = self.vocab.numericalize(random.choice(captions))
        padded = torch.full((self.max_len,), self.vocab.stoi[PAD], dtype=torch.long)
        end = min(len(ids), self.max_len)
        padded[:end] = torch.tensor(ids[:end])
        return feature, padded, captions


def collate_fn(batch):
    features, captions, refs = zip(*batch)
    return torch.stack(features), torch.stack(captions), refs
