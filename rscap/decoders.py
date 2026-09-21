"""Caption decoders. Both consume a single pooled image feature vector.

LSTMDecoder      concatenates the image feature to every word embedding (the feature
                 is re-injected at each step rather than used only as the initial state).
TransformerDecoderModel
                 projects the feature into `memory_len` pseudo-tokens and cross-attends
                 to them with a standard causal Transformer decoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BOS, EOS


class LSTMDecoder(nn.Module):
    def __init__(self, encoder_dim, embed_dim, hidden_dim, vocab_size, num_layers=2, dropout_p=0.5):
        super().__init__()
        self.num_layers, self.hidden_dim = num_layers, hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        """Teacher forcing. `captions` is [B, T] including BOS; predicts positions 1..T-1."""
        captions = captions[:, :-1]
        emb = self.embedding(captions)
        feat = features.unsqueeze(1).expand(-1, emb.size(1), -1)
        out, _ = self.lstm(self.dropout(torch.cat([emb, feat], dim=2)))
        return self.fc(out)

    def _init_state(self, features):
        h = torch.zeros(self.num_layers, 1, self.hidden_dim, device=features.device)
        return h, h.clone()

    @torch.no_grad()
    def generate_caption(self, features, vocab, max_len=24):
        """Greedy decoding for a single image feature [1, encoder_dim]."""
        self.eval()
        ids, states = [vocab.stoi[BOS]], self._init_state(features)
        for _ in range(max_len):
            emb = self.embedding(torch.tensor([ids[-1]], device=features.device))
            out, states = self.lstm(torch.cat([emb, features], dim=1).unsqueeze(1), states)
            nxt = self.fc(out.squeeze(1)).argmax(1).item()
            ids.append(nxt)
            if nxt == vocab.stoi[EOS]:
                break
        return vocab.decode(ids)

    @torch.no_grad()
    def generate_caption_beam_search(self, features, vocab, max_len=24, beam_width=3):
        self.eval()
        bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
        beam = [([bos], 0.0, self._init_state(features))]
        completed = []
        for _ in range(max_len):
            candidates = []
            for seq, logp, states in beam:
                if seq[-1] == eos:
                    completed.append((seq, logp))
                    continue
                emb = self.embedding(torch.tensor([seq[-1]], device=features.device))
                out, new_states = self.lstm(torch.cat([emb, features], dim=1).unsqueeze(1), states)
                logprobs = F.log_softmax(self.fc(out.squeeze(1)), dim=1)
                top_lp, top_ix = logprobs.topk(beam_width, dim=1)
                for lp, ix in zip(top_lp[0].tolist(), top_ix[0].tolist()):
                    candidates.append((seq + [ix], logp + lp, new_states))
            if not candidates:
                break
            beam = sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_width]
        completed.extend((s, lp) for s, lp, _ in beam)
        best = max(completed, key=lambda c: c[1] / len(c[0]))[0]
        return vocab.decode(best)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerDecoderModel(nn.Module):
    def __init__(self, encoder_dim, d_model=512, nhead=8, num_layers=4, vocab_size=10000, memory_len=4):
        super().__init__()
        self.d_model, self.memory_len = d_model, memory_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.memory_projection = nn.Linear(encoder_dim, memory_len * d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def _memory(self, features):
        return self.memory_norm(self.memory_projection(features).view(-1, self.memory_len, self.d_model))

    def forward(self, features, tgt_in, tgt_mask, tgt_padding_mask):
        x = self.pos(self.embedding(tgt_in) * math.sqrt(self.d_model))
        out = self.decoder(x, self._memory(features), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.fc(out)

    @torch.no_grad()
    def generate_caption(self, features, vocab, max_len=24):
        self.eval()
        memory = self._memory(features)
        seq = torch.tensor([[vocab.stoi[BOS]]], device=features.device)
        for _ in range(max_len - 1):
            mask = causal_mask(seq.size(1), features.device)
            x = self.pos(self.embedding(seq) * math.sqrt(self.d_model))
            nxt = self.fc(self.decoder(x, memory, tgt_mask=mask)[:, -1]).argmax(1)
            seq = torch.cat([seq, nxt.unsqueeze(0)], dim=1)
            if nxt.item() == vocab.stoi[EOS]:
                break
        return vocab.decode(seq.squeeze(0).tolist())


def causal_mask(size, device):
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def padding_mask(seq, pad_idx):
    return seq == pad_idx


class EncoderDecoder(nn.Module):
    """Feature -> LSTMDecoder. Trained on cached encoder features."""

    def __init__(self, encoder_dim, embed_dim=512, hidden_dim=512, vocab_size=10000, num_layers=2):
        super().__init__()
        self.decoder = LSTMDecoder(encoder_dim, embed_dim, hidden_dim, vocab_size, num_layers)

    def forward(self, features, captions):
        return self.decoder(features, captions)


class EncoderDecoderTransformer(nn.Module):
    """Feature -> TransformerDecoderModel. Trained on cached encoder features."""

    def __init__(self, encoder_dim, d_model=512, nhead=8, num_layers=4, vocab_size=10000):
        super().__init__()
        self.decoder = TransformerDecoderModel(encoder_dim, d_model, nhead, num_layers, vocab_size)

    def forward(self, features, tgt_in, tgt_mask, tgt_padding_mask):
        return self.decoder(features, tgt_in, tgt_mask, tgt_padding_mask)


class FullEncoderDecoder(nn.Module):
    """CNN encoder + trained decoder, for end-to-end inference on raw images."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    @torch.no_grad()
    def caption(self, image_tensor, vocab, beam_width=3):
        feats = self.encoder(image_tensor)
        if hasattr(self.decoder, "generate_caption_beam_search"):
            return self.decoder.generate_caption_beam_search(feats, vocab, beam_width=beam_width)
        return self.decoder.generate_caption(feats, vocab)


def build_model(decoder_type, encoder_dim, vocab_size):
    if decoder_type == "lstm":
        return EncoderDecoder(encoder_dim, vocab_size=vocab_size)
    if decoder_type == "transformer":
        return EncoderDecoderTransformer(encoder_dim, vocab_size=vocab_size)
    raise ValueError(f"decoder must be 'lstm' or 'transformer', got {decoder_type!r}")
