"""Fast checks that need no data or GPU: tokenizer behaviour, model shapes, decoding."""

import torch

from rscap.data import Vocabulary, ensure_nltk
from rscap.decoders import EncoderDecoder, EncoderDecoderTransformer, causal_mask, padding_mask
from rscap.encoders import CNNEncoder


def _vocab():
    ensure_nltk()
    v = Vocabulary()
    v.build([["a few boats are near the pier", "many white planes are parked"]], vocab_size=50)
    return v


def test_tokenizer_splits_glued_punctuation():
    ensure_nltk()
    assert Vocabulary.tokenizer("A few boats.a are here.") == ["a", "few", "boats", "a", "are", "here"]
    assert Vocabulary.tokenizer("circle.some near.the") == ["circle", "some", "near", "the"]


def test_numericalize_roundtrip():
    v = _vocab()
    ids = v.numericalize("many boats are parked")
    assert ids[0] == v.stoi["<bos>"] and ids[-1] == v.stoi["<eos>"]
    assert v.decode(ids) == ["many", "boats", "are", "parked"]


def test_encoders_output_shapes():
    x = torch.randn(2, 3, 224, 224)
    assert CNNEncoder("resnet18", pretrained=False)(x).shape == (2, 512)
    assert CNNEncoder("mobilenet_v2", pretrained=False)(x).shape == (2, 1280)


def test_lstm_forward_and_decode():
    v = _vocab()
    m = EncoderDecoder(encoder_dim=512, vocab_size=len(v)).eval()
    feats, caps = torch.randn(3, 512), torch.randint(0, len(v), (3, 12))
    assert m(feats, caps).shape == (3, 11, len(v))
    words = m.decoder.generate_caption_beam_search(feats[:1], v, max_len=8)
    assert isinstance(words, list) and len(words) <= 8


def test_transformer_forward_and_decode():
    v = _vocab()
    m = EncoderDecoderTransformer(encoder_dim=1280, vocab_size=len(v)).eval()
    feats, caps = torch.randn(3, 1280), torch.randint(0, len(v), (3, 12))
    tgt = caps[:, :-1]
    out = m(feats, tgt, causal_mask(tgt.size(1), feats.device), padding_mask(tgt, v.stoi["<pad>"]))
    assert out.shape == (3, 11, len(v))
    words = m.decoder.generate_caption(feats[:1], v, max_len=8)
    assert isinstance(words, list) and len(words) <= 8
