"""Slice evaluation and Grad-CAM on the encoder."""

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CaptionDataset, collate_fn, decode_image
from .train import compute_metrics


def slice_by_keyword(df, keyword):
    """Rows whose reference captions mention `keyword` anywhere."""
    return df[df["captions"].apply(lambda caps: any(keyword in c for c in caps))]


def bleu4_on_slice(df, model, vocab, feature_dir, max_len, device, batch_size=32):
    if len(df) == 0:
        return float("nan")
    loader = DataLoader(CaptionDataset(df, vocab, feature_dir, max_len), batch_size=batch_size, collate_fn=collate_fn)
    return compute_metrics(model, loader, vocab, device, max_len)["BLEU-4"]


def gradcam_for_row(full_model, row, transform, vocab, device, target_layer_type=nn.BatchNorm2d):
    """Grad-CAM over the encoder's last `target_layer_type` layer, using the summed
    feature vector as the target, alongside the caption the full model produces.

    Returns (pil_image, heatmap_overlay_uint8, caption_words)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    img = decode_image(row)
    x = transform(img).unsqueeze(0).to(device)
    cnn = full_model.encoder.cnn.eval()
    layers = [m for m in cnn.modules() if isinstance(m, target_layer_type)][-1:]
    if not layers:
        raise ValueError(f"no {target_layer_type.__name__} layer in encoder")

    class SumTarget:
        def __call__(self, out):
            return out.sum()

    cam = GradCAM(model=cnn, target_layers=layers)(input_tensor=x, targets=[SumTarget()])[0]
    words = full_model.caption(x, vocab)
    overlay = show_cam_on_image(np.asarray(img.resize((224, 224))) / 255.0, cam, use_rgb=True)
    return img, overlay, words
