# -*- coding: utf-8 -*-
"""
PyQt5 GUI for FireMoE single-model fire recognition + attention map.

功能：
1. 选择一个 .pth 模型权重；
2. 图片识别、视频识别、摄像头识别；
3. 左侧显示原图 / 视频帧，右侧显示 mix attention map 叠加图；
4. 默认使用模型内置阈值，也可以在界面手动覆盖；
5. 状态区展示 prob、global/local attention 路径、专家路由权重；
7. 导出当前会话中所有已识别帧的 attention map 和预测数据。

运行：
    python GUI.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import transforms

# 摄像头实时显示时，限制 OpenCV / PyTorch 过度占用 CPU，避免 GUI 主线程被抢占。
try:
    cv2.setNumThreads(1)
    torch.set_num_threads(max(1, min(2, (os.cpu_count() or 2) // 2)))
except Exception:
    pass

from PyQt5 import QtCore, QtGui, QtWidgets

from fire_model import (
    DEFAULT_OWLVIT_NEGATIVE_PROMPTS,
    DEFAULT_OWLVIT_POSITIVE_PROMPTS,
    DEFAULT_OWLVIT_SUSPICIOUS_PROMPTS,
    FireArbiterMoELite,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


from .model_files import (
    DEFAULT_FIRE_LITE_PATH,
    MODELS_DIR,
    gui_local_model_loading,
    resolve_fire_lite_checkpoint,
    resolve_gui_convnext_weight,
    resolve_gui_owl_source,
)


# =============
# 与训练 / 推理 notebook 保持一致的基础参数
# =============
DEFAULT_THRESHOLD = 0.50
DEFAULT_HEIGHT_RESIZE_ENABLED = True
DEFAULT_INFER_TARGET_HEIGHT = 640
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "gui_config.json"
BOTTOM_MODEL_BAR_HEIGHT = 116

IMAGE_EXTS = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
VIDEO_EXTS = "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BATCH_TMP_ROOT = Path.cwd() / "tmp"
BATCH_AUDIT_FILENAME = "batch_audit_labels.json"
BATCH_CURRENT_PREDICTIONS_FILENAME = "batch_predictions_current.json"
BATCH_CACHE_VERSION = 5
# I/O optimization: batch detection should not write every diagnostic image by default.
# Full attention-map caching can generate 14 PNGs per input image, so keep only
# the mixed map/overlay when attention caching is explicitly enabled.
BATCH_ATTENTION_CACHE_KEYS = ("mix",)
BATCH_OVERLAY_JPG_QUALITY = 85
BATCH_AUDIT_AUTOSAVE_INTERVAL = 25

PROMPT_GROUP_DEFS = {
    "positive": ("正向", DEFAULT_OWLVIT_POSITIVE_PROMPTS),
    "negative": ("负向", DEFAULT_OWLVIT_NEGATIVE_PROMPTS),
    "suspicious": ("疑似", DEFAULT_OWLVIT_SUSPICIOUS_PROMPTS),
}


global_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


MOLD_STAGE_PRESETS = [
    {"name": "stage_00_original_reference", "params": {"jpeg_quality": 95}},
    {"name": "stage_01_progressive_loss", "params": {"jpeg_quality": 90}},
    {"name": "stage_02_progressive_loss", "params": {"jpeg_quality": 75}},
    {"name": "stage_03_progressive_loss", "params": {"downscale": 0.85, "jpeg_quality": 70}},
    {"name": "stage_04_progressive_loss", "params": {"downscale": 0.70, "blur": 0.4, "jpeg_quality": 62}},
    {"name": "stage_05_progressive_loss", "params": {"downscale": 0.58, "blur": 0.7, "jpeg_quality": 55, "saturation": 1.04}},
    {"name": "stage_06_progressive_loss", "params": {"downscale": 0.48, "blur": 1.0, "jpeg_quality": 48, "contrast": 0.96}},
    {"name": "stage_07_progressive_loss", "params": {"downscale": 0.50, "blur": 1.0, "jpeg_quality": 45}},
    {"name": "stage_08_progressive_loss", "params": {"downscale": 0.40, "blur": 1.3, "jpeg_quality": 38, "saturation": 1.08}},
    {"name": "stage_09_progressive_loss", "params": {"downscale": 0.33, "blur": 1.8, "jpeg_quality": 30, "contrast": 0.92}},
    {"name": "stage_10_progressive_loss", "params": {"downscale": 0.27, "blur": 2.4, "jpeg_quality": 24, "saturation": 1.12}},
    {"name": "stage_11_progressive_loss", "params": {"downscale": 0.22, "blur": 3.0, "jpeg_quality": 18, "contrast": 0.88}},
    {"name": "stage_12_progressive_loss", "params": {"downscale": 0.18, "blur": 3.8, "jpeg_quality": 12, "saturation": 1.15, "contrast": 0.84}},
]



def apply_exposure_mode_rgb(rgb: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Add randomized over-exposure / bloom patches, biased toward red-yellow regions.

    The mold test uses this as an optional hard-negative augmentation: locations are
    random, but most centers are sampled from pixels with strong red/yellow content
    so lamps, warm objects, flames, sparks, and similar areas are stressed more.
    """
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())

    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8))

    seed = params.get("exposure_seed", None)
    try:
        seed = int(seed) if seed is not None else None
    except Exception:
        seed = None
    rng = np.random.default_rng(seed)

    strength = float(params.get("exposure_strength", 0.85))
    strength = max(0.05, min(2.50, strength))
    size_scale = float(params.get("exposure_size_scale", params.get("exposure_range_scale", 1.0)))
    size_scale = max(0.10, min(4.00, size_scale))
    warm_bias = float(params.get("exposure_warm_bias", 0.75))
    warm_bias = max(0.0, min(1.0, warm_bias))
    if "exposure_spots" in params:
        try:
            spot_count = int(params.get("exposure_spots", 2))
        except Exception:
            spot_count = 2
    else:
        spot_count = int(rng.integers(1, 4))
    spot_count = max(1, min(8, spot_count))

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    red_score = np.maximum(r - np.maximum(g, b) * 0.60, 0.0)
    yellow_score = np.maximum(np.minimum(r, g) - b * 0.72, 0.0)
    brightness = (r + g + b) / 765.0
    warm_score = (red_score + yellow_score * 1.25) * (0.35 + 0.65 * brightness)
    if np.isfinite(warm_score).all():
        warm_score = np.maximum(warm_score - np.percentile(warm_score, 60.0), 0.0)
    else:
        warm_score = np.zeros((h, w), dtype=np.float32)
    # Use a cumulative distribution instead of np.random.choice(..., p=...).
    # Generator.choice is strict about p.sum() == 1 and can raise
    # "Probabilities do not sum to 1" with float32/large-image rounding.
    # The cumulative method is numerically stable and falls back to uniform
    # sampling when no reliable warm/red-yellow region is available.
    warm_flat = np.asarray(warm_score.reshape(-1), dtype=np.float64)
    warm_flat[~np.isfinite(warm_flat)] = 0.0
    warm_flat = np.maximum(warm_flat, 0.0)
    warm_total = float(warm_flat.sum(dtype=np.float64))
    warm_cdf = np.cumsum(warm_flat, dtype=np.float64) if warm_total > 1e-9 else None

    yy_all = np.arange(h, dtype=np.float32)
    xx_all = np.arange(w, dtype=np.float32)

    for _ in range(spot_count):
        if warm_cdf is not None and warm_total > 1e-9 and rng.random() < warm_bias:
            pick = float(rng.random() * warm_total)
            idx = int(np.searchsorted(warm_cdf, pick, side="right"))
            if idx < 0 or idx >= h * w:
                idx = int(rng.integers(0, max(1, h * w)))
            cy, cx = divmod(idx, w)
            # Small jitter keeps the effect random while remaining near warm regions.
            cx = int(np.clip(cx + rng.normal(0.0, max(1.0, w * 0.035)), 0, w - 1))
            cy = int(np.clip(cy + rng.normal(0.0, max(1.0, h * 0.035)), 0, h - 1))
        else:
            cx = int(rng.integers(0, max(1, w)))
            cy = int(rng.integers(0, max(1, h)))

        sigma_x = max(4.0, float(rng.uniform(0.045, 0.16) * w * size_scale))
        sigma_y = max(4.0, float(rng.uniform(0.045, 0.16) * h * size_scale))
        x0 = max(0, int(cx - sigma_x * 3.0))
        x1 = min(w, int(cx + sigma_x * 3.0) + 1)
        y0 = max(0, int(cy - sigma_y * 3.0))
        y1 = min(h, int(cy + sigma_y * 3.0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        xs = xx_all[x0:x1]
        ys = yy_all[y0:y1]
        dx2 = ((xs[None, :] - float(cx)) / sigma_x) ** 2
        dy2 = ((ys[:, None] - float(cy)) / sigma_y) ** 2
        mask = np.exp(-0.5 * (dx2 + dy2)).astype(np.float32)
        local_strength = strength * float(rng.uniform(0.65, 1.35))

        # Warm over-exposure: yellow/orange center plus a white-ish saturation halo.
        tint = np.array([
            1.00,
            float(rng.uniform(0.76, 1.00)),
            float(rng.uniform(0.26, 0.62)),
        ], dtype=np.float32)
        patch = arr[y0:y1, x0:x1, :]
        m = mask[:, :, None]
        patch += (255.0 - patch) * m * local_strength * np.array([1.0, 0.92, 0.70], dtype=np.float32)
        patch += 255.0 * m * (0.26 * local_strength) * tint
        arr[y0:y1, x0:x1, :] = patch

    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_overall_exposure_rgb(rgb: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Brighten the whole image with selectable weighting.

    Modes:
    - global: lift exposure across the full image;
    - highlights: brighten bright regions more strongly;
    - shadows: lift dark regions more strongly while preserving highlights.
    """
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())

    try:
        strength = float(params.get("overall_exposure_strength", 0.45))
    except Exception:
        strength = 0.45
    strength = max(0.0, min(2.50, strength))
    if strength <= 1e-6:
        return np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8))

    mode = str(params.get("overall_exposure_strategy", "global")).strip().lower()
    if mode not in {"global", "highlights", "shadows"}:
        mode = "global"

    arr01 = arr / 255.0
    luminance = (0.299 * arr01[:, :, 0] + 0.587 * arr01[:, :, 1] + 0.114 * arr01[:, :, 2]).astype(np.float32)

    def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    if mode == "highlights":
        weight = 0.18 + 0.82 * smoothstep(0.42, 0.92, luminance)
    elif mode == "shadows":
        weight = 0.18 + 0.82 * (1.0 - smoothstep(0.12, 0.72, luminance))
    else:
        weight = np.ones_like(luminance, dtype=np.float32)

    # Screen-like exposure lift. It increases brightness while naturally saturating
    # near white; the per-pixel weight implements highlights/shadows priority.
    lift = np.clip(0.42 * strength * weight[:, :, None], 0.0, 1.6)
    out = arr01 + (1.0 - arr01) * lift

    # Add a small EV-style gain so very dark pixels visibly lift in shadow mode.
    if mode == "shadows":
        out = np.minimum(1.0, out * (1.0 + 0.18 * strength * weight[:, :, None]))
    elif mode == "global":
        out = np.minimum(1.0, out * (1.0 + 0.08 * strength))

    return np.ascontiguousarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))

def mold_degrade_rgb(rgb: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    image = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).convert("RGB")
    if "downscale" in params:
        scale = float(params["downscale"])
        width, height = image.size
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        small = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        image = small.resize((width, height), Image.Resampling.BILINEAR)
    if "blur" in params and float(params["blur"]) > 0:
        image = image.filter(ImageFilter.GaussianBlur(float(params["blur"])))
    if "saturation" in params:
        image = ImageEnhance.Color(image).enhance(float(params["saturation"]))
    if "contrast" in params:
        image = ImageEnhance.Contrast(image).enhance(float(params["contrast"]))
    if bool(params.get("overall_exposure_mode", False)):
        image = Image.fromarray(apply_overall_exposure_rgb(np.asarray(image, dtype=np.uint8), params)).convert("RGB")
    if bool(params.get("exposure_mode", False)):
        image = Image.fromarray(apply_exposure_mode_rgb(np.asarray(image, dtype=np.uint8), params)).convert("RGB")
    quality = int(round(float(params.get("jpeg_quality", 90))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False, progressive=False, subsampling=2)
    buffer.seek(0)
    return np.ascontiguousarray(np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8).copy())


def default_prompt_groups() -> Dict[str, List[str]]:
    return {key: list(defaults) for key, (_label, defaults) in PROMPT_GROUP_DEFS.items()}


def parse_prompt_text(text: str) -> List[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def prompt_groups_from_mapping(groups: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        key: parse_prompt_text(groups[key]) if isinstance(groups[key], str)
        else [str(item).strip() for item in groups[key]
              if str(item).strip()]
        for key in PROMPT_GROUP_DEFS
    }


def safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_threshold_from_checkpoint_dict(checkpoint, default_threshold=DEFAULT_THRESHOLD):
    """Read the fixed inference threshold saved by training; returns (threshold, source_key)."""
    if isinstance(checkpoint, dict):
        candidates = []
        if isinstance(checkpoint.get("model_config"), dict):
            candidates.append(("model_config", checkpoint["model_config"]))
        if isinstance(checkpoint.get("metadata"), dict):
            candidates.append(("metadata", checkpoint["metadata"]))
        candidates.append(("checkpoint", checkpoint))
        for prefix, obj in candidates:
            for key in ["inference_threshold", "threshold"]:
                if key not in obj:
                    continue
                try:
                    value = float(obj[key])
                    if np.isfinite(value):
                        return min(max(value, 0.0), 1.0), f"{prefix}.{key}"
                except Exception:
                    pass
    return float(default_threshold), "default"


def finite_float(value, default=None):
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().reshape(-1)[0].item()
        elif isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32).reshape(-1)[0].item()
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def extract_result_float(result: Dict[str, Any] | None, keys, default=None):
    if not isinstance(result, dict):
        return default
    for key in keys:
        value = finite_float(result.get(key), None)
        if value is not None:
            return value

    nested_result = result.get("result")
    if isinstance(nested_result, dict):
        value = extract_result_float(nested_result, keys, None)
        if value is not None:
            return value

    model_results = result.get("model_results")
    if isinstance(model_results, list) and model_results:
        for item in model_results:
            value = extract_result_float(item, keys, None)
            if value is not None:
                return value
    return default


def result_prob(result: Dict[str, Any] | None, default=None):
    return extract_result_float(result, ["prob_fire", "fire_prob", "prob", "probability", "score"], default)


def result_threshold(result: Dict[str, Any] | None, default=None):
    return extract_result_float(result, ["threshold", "used_threshold", "model_threshold"], default)


def prepare_rgb_for_inference(
    rgb: np.ndarray,
    height_resize_enabled: bool = DEFAULT_HEIGHT_RESIZE_ENABLED,
    target_height: int = DEFAULT_INFER_TARGET_HEIGHT,
    **legacy_resize_settings,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    src = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w = src.shape[:2]
    if "resize_enabled" in legacy_resize_settings:
        height_resize_enabled = bool(legacy_resize_settings["resize_enabled"])
    if "target_height" in legacy_resize_settings:
        target_height = int(legacy_resize_settings["target_height"])
    meta = {
        "height_resize_enabled": bool(height_resize_enabled),
        "resize_target_height": int(target_height),
        "resize_scale": 1.0,
        "resize_output_width": int(w),
        "resize_output_height": int(h),
        # Legacy aliases kept only for old result/export readers.
        "resize_enabled": bool(height_resize_enabled),
        "resize_keep_aspect": True,
        "resize_target_width": 0,
        "resize_pad_left": 0,
        "resize_pad_top": 0,
        "resize_content_width": int(w),
        "resize_content_height": int(h),
    }
    if not height_resize_enabled:
        return src, np.ones((h, w), dtype=np.float32), meta

    target_h = max(1, int(target_height))
    if h <= target_h:
        return src, np.ones((h, w), dtype=np.float32), meta
    scale = float(target_h) / float(max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = int(target_h)
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    meta.update({
        "resize_scale": float(scale),
        "resize_output_width": int(new_w),
        "resize_output_height": int(new_h),
        "resize_content_width": int(new_w),
        "resize_content_height": int(new_h),
    })
    return np.ascontiguousarray(resized), np.ones((new_h, new_w), dtype=np.float32), meta


def resize_rgb_for_inference(rgb: np.ndarray) -> np.ndarray:
    return prepare_rgb_for_inference(rgb)[0]


def normalize_map(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    denom = float(np.nanmax(arr))
    if denom > 1e-8:
        arr = arr / denom
    return arr.astype(np.float32)


def attention_to_canvas(attn, raw_w: int, raw_h: int) -> np.ndarray:
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().float().cpu()
    arr = np.asarray(attn, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    t = torch.tensor(arr)[None, None]
    up = F.interpolate(t, size=(raw_h, raw_w), mode="bilinear", align_corners=False)[0, 0].numpy()
    return normalize_map(up)


def paste_tile_canvas(tile_maps, tile_boxes, raw_w: int, raw_h: int, tile_weights=None) -> np.ndarray:
    canvas = np.zeros((raw_h, raw_w), dtype=np.float32)
    weight = np.zeros((raw_h, raw_w), dtype=np.float32)
    if tile_weights is None:
        tile_weights = np.ones(len(tile_boxes), dtype=np.float32)

    for i, box in enumerate(tile_boxes):
        if i >= len(tile_maps):
            break
        x1, y1, x2, y2 = [int(v) for v in box]
        tile_map = np.asarray(tile_maps[i], dtype=np.float32)
        if tile_map.ndim == 3:
            tile_map = tile_map[0]
        tile_up = F.interpolate(
            torch.tensor(tile_map)[None, None],
            size=(max(1, y2 - y1), max(1, x2 - x1)),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        w = float(tile_weights[i])
        canvas[y1:y2, x1:x2] += tile_up * w
        weight[y1:y2, x1:x2] += max(w, 1e-6)

    return normalize_map(canvas / np.maximum(weight, 1e-6))


def build_tile_value_canvas(tile_values, tile_boxes, raw_w: int, raw_h: int) -> np.ndarray:
    canvas = np.zeros((raw_h, raw_w), dtype=np.float32)
    weight = np.zeros((raw_h, raw_w), dtype=np.float32)
    values = np.asarray(tile_values, dtype=np.float32)
    for i, box in enumerate(tile_boxes):
        if i >= len(values):
            break
        x1, y1, x2, y2 = [int(v) for v in box]
        canvas[y1:y2, x1:x2] += float(values[i])
        weight[y1:y2, x1:x2] += 1.0
    return canvas / np.maximum(weight, 1e-6)


def heatmap_overlay(rgb: np.ndarray, attention_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.ascontiguousarray(rgb.astype(np.uint8))
    heat_u8 = np.uint8(np.clip(normalize_map(attention_map) * 255, 0, 255))
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(base, 1.0 - alpha, heat_rgb, alpha, 0)
    return out


def map_to_gray(attention_map: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(normalize_map(attention_map) * 255, 0, 255))


def safe_artifact_stem(key: str) -> str:
    text = str(key).replace("\\", "/").replace("/", "__")
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in "._-()[]" else "_")
    stem = "".join(keep).strip("._")
    return stem or "image"


def normalize_label_value(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return 1 if float(value) >= 0.5 else 0
    text = str(value).strip().lower()
    fire_words = {"1", "true", "yes", "y", "fire", "has_fire", "有火", "是", "阳性", "positive"}
    no_fire_words = {"0", "false", "no", "n", "no_fire", "nofire", "无火", "没火", "否", "阴性", "negative"}
    if text in fire_words:
        return 1
    if text in no_fire_words:
        return 0
    return None


def extract_label_from_record(record) -> int | None:
    if not isinstance(record, dict):
        return normalize_label_value(record)
    for key in [
        "has_fire", "fire", "is_fire", "label", "gt", "ground_truth", "truth",
        "manual_label", "final_label", "prediction_label", "pred",
    ]:
        if key in record:
            label = normalize_label_value(record.get(key))
            if label is not None:
                return label
    return None


def parse_ground_truth_payload(payload) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, dict):
            for key, record in items.items():
                label = extract_label_from_record(record)
                if label is not None:
                    labels[str(key).replace("\\", "/")] = int(label)
        elif isinstance(items, list):
            for record in items:
                if not isinstance(record, dict):
                    continue
                name = record.get("filename") or record.get("file_name") or record.get("name") or record.get("path") or record.get("key")
                label = extract_label_from_record(record)
                if name and label is not None:
                    labels[str(name).replace("\\", "/")] = int(label)
        for key, value in payload.items():
            if key in {"items", "stats", "created_at", "root_dir", "tmp_dir", "predictions"}:
                continue
            label = extract_label_from_record(value)
            if label is not None:
                labels[str(key).replace("\\", "/")] = int(label)
    elif isinstance(payload, list):
        for record in payload:
            if not isinstance(record, dict):
                continue
            name = record.get("filename") or record.get("file_name") or record.get("name") or record.get("path") or record.get("key")
            label = extract_label_from_record(record)
            if name and label is not None:
                labels[str(name).replace("\\", "/")] = int(label)
    return labels


def ground_truth_label_for_key(labels: Dict[str, int], key: str, path: str = "") -> int | None:
    candidates = [str(key).replace("\\", "/")]
    if path:
        p = Path(path)
        candidates.extend([str(path).replace("\\", "/"), p.name])
    candidates.append(Path(str(key)).name)
    for candidate in candidates:
        if candidate in labels:
            return int(labels[candidate])
    return None


def normalized_batch_input_size_settings(resize_settings: Dict[str, Any] | None) -> Dict[str, Any]:
    settings = dict(resize_settings or {})
    enabled = bool(settings.get("height_resize_enabled", settings.get("resize_enabled", DEFAULT_HEIGHT_RESIZE_ENABLED)))
    target_height = int(settings.get("target_height", settings.get("resize_target_height", DEFAULT_INFER_TARGET_HEIGHT)))
    return {
        "height_resize_enabled": enabled,
        "target_height": target_height if enabled else 0,
        "input_size_mode": f"height_{target_height}" if enabled else "original",
    }


def batch_cache_dir(root_dir: str, model_settings: List[Dict[str, Any]], resize_settings: Dict[str, Any]) -> Path:
    # 阈值不影响模型概率，因此不参与缓存键；prompt 会改变 OWL-ViT 语义分数，必须参与缓存键。
    model_paths = []
    for item in model_settings:
        path = str(item.get("path", ""))
        if path:
            try:
                path = str(Path(path).resolve())
            except Exception:
                path = str(Path(path))
            model_paths.append(path)
    prompt_settings = []
    for item in model_settings:
        prompt_settings.append({
            "positive": list(item["owl_positive_prompts"]),
            "negative": list(item["owl_negative_prompts"]),
            "suspicious": list(item["owl_suspicious_prompts"]),
        })

    payload = {
        "version": BATCH_CACHE_VERSION,
        "model_paths": sorted(model_paths),
        "input_size": normalized_batch_input_size_settings(resize_settings),
        "owlvit_prompts": prompt_settings,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return BATCH_TMP_ROOT / f"batch_cache_{digest}"


def reclassify_result_by_threshold(
    result: Dict[str, Any],
    model_settings: List[Dict[str, Any]] | None,
    default_model_threshold: float = DEFAULT_THRESHOLD,
    default_threshold_source: str = "model",
) -> Dict[str, Any]:
    """Reuse cached model probability and refresh binary prediction by current threshold."""
    out = dict(result or {})
    setting = dict(model_settings[0]) if model_settings else {}
    manual_override = bool(setting.get("manual_threshold", False))
    if manual_override:
        threshold = float(setting.get("threshold", default_model_threshold))
        threshold_source = "manual"
    else:
        threshold = float(setting.get("model_threshold", default_model_threshold))
        threshold_source = str(setting.get("threshold_source", default_threshold_source) or default_threshold_source)
    threshold = min(max(threshold, 0.0), 1.0)

    prob = result_prob(out, None)
    if prob is None:
        return out
    pred = int(float(prob) >= threshold)
    out.update({
        "pred": int(pred),
        "result": "fire" if pred else "no_fire",
        "result_cn": "有火" if pred else "无火",
        "threshold": float(threshold),
        "used_threshold": float(threshold),
        "manual_threshold": bool(manual_override),
        "threshold_source": threshold_source,
        "model_threshold": float(default_model_threshold),
    })

    model_results = out.get("model_results")
    if isinstance(model_results, list) and model_results:
        refreshed = []
        for model_result in model_results:
            if isinstance(model_result, dict):
                mr = dict(model_result)
                mr_prob = result_prob(mr, prob)
                mr_pred = int(float(mr_prob) >= threshold) if mr_prob is not None else pred
                mr.update({
                    "pred": int(mr_pred),
                    "result": "fire" if mr_pred else "no_fire",
                    "result_cn": "有火" if mr_pred else "无火",
                    "threshold": float(threshold),
                    "used_threshold": float(threshold),
                    "manual_threshold": bool(manual_override),
                    "threshold_source": threshold_source,
                    "model_threshold": float(default_model_threshold),
                })
                refreshed.append(mr)
            else:
                refreshed.append(model_result)
        out["model_results"] = refreshed
    return out

def side_by_side_rgb(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.ascontiguousarray(left.astype(np.uint8))
    right = np.ascontiguousarray(right.astype(np.uint8))
    h = max(left.shape[0], right.shape[0])

    def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        scale = float(target_h) / float(max(img.shape[0], 1))
        target_w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    left = resize_to_height(left, h)
    right = resize_to_height(right, h)
    gap = np.full((h, 12, 3), 245, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def imread_rgb(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_png(path: str | Path, image: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = image
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError(f"无法写入图片: {path}")
    buf.tofile(str(path))


def imwrite_jpg(path: str | Path, image: np.ndarray, quality: int = 92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = image
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", arr, params)
    if not ok:
        raise RuntimeError(f"无法写入图片: {path}")
    buf.tofile(str(path))


def rgb_to_qpixmap(rgb: np.ndarray, label: QtWidgets.QLabel, smooth: bool = False) -> QtGui.QPixmap:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w = rgb.shape[:2]
    target_w = max(1, int(label.width()))
    target_h = max(1, int(label.height()))
    if w > target_w or h > target_h:
        scale = min(float(target_w) / float(max(w, 1)), float(target_h) / float(max(h, 1)))
        scaled_w = max(1, int(round(w * scale)))
        scaled_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=interp)
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        h, w = rgb.shape[:2]
    ch = rgb.shape[2]
    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
    pix = QtGui.QPixmap.fromImage(qimg)
    mode = QtCore.Qt.SmoothTransformation if smooth else QtCore.Qt.FastTransformation
    return pix.scaled(label.size(), QtCore.Qt.KeepAspectRatio, mode)


