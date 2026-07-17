# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from fire_model import FireArbiterMoELite, IMAGENET_MEAN, IMAGENET_STD
from fire_training_plots import save_expert_epoch, save_training_curves


def build_train_config() -> Dict:
    """继续读取旧工程已经生成的全量 train/val CSV，不进行重新划分。"""
    root = Path(os.environ.get("FIRE_ROOT", "..")).resolve()
    split_dir = root / "fire_splits"
    expert_names = [
        "origin_competition",
        "open_fire_general",
        "indoor_cctv_smoke",
        "manual_local_crop",
        "traffic_light_hard_negative",
        "sunset_led_news_hard_negative",
        "ultra_low_quality_blurry",
    ]
    return {
        "root": root,
        "train_csv": split_dir / "train_all.csv",
        "val_csv": split_dir / "val_all_datasets_real_only.csv",
        "train_output_dir": Path("train_lite"),
        "model_output_dir": Path("models_lite"),
        "seed": 35,
        "batch_size": 8,
        "num_workers": 6,
        "persistent_workers": True,
        "image_target_height": 720,
        "validation_attention_sample_index": 0,
        "expert_names": expert_names,
        "image_degradation_config": {
            "offline_cache_enabled": True,
            "offline_cache_index_csv": str(split_dir / "degraded_image_cache_index.csv"),
            "offline_cache_require_label_match": True,
            "offline_cache_no_fire_prob": 0.55,
            "offline_cache_fire_prob": 0.30,
            "enabled": False,
            "no_fire_prob": 0.50,
            "fire_prob": 0.30,
            "max_stage": 12,
            "no_fire_stage_weights": [0.0, 0.0, 1.2, 1.4, 1.4, 1.2, 0.4, 0.4, 0.9, 0.8, 0.6, 0.3, 0.2],
            "fire_stage_weights": [0.0, 0.8, 0.8, 0.6, 0.4, 0.2, 0.1],
        },
        "model": {
            "owlvit_model_name": "google/owlvit-base-patch32",
            "owlvit_image_size": 768,
            "owlvit_feature_dim": 128,
            "owlvit_trainable_vision_layers": 2,
            "owlvit_trainable_text_layers": 0,
            "owlvit_train_class_head": True,
            "owlvit_train_box_head": False,
            "convnext_trainable_layers": 1,
            "expert_adapter_bottleneck_ratio": 4,
            "expert_adapter_dropout": 0.0,
            "expert_advice_dim": 16,
        },
        "stages": {
            "semantic_warmup": {
                "epochs": 3, "force_expert": False,
                "lr_head": 8.0e-5, "lr_backbone": 0.0, "lr_owl": 0.0,
                "weights": {"cls": 1.0, "branch": 0.30, "map": 0.25, "expert": 0.0, "calibration": 0.01},
            },
            "expert_specialization": {
                "epochs": 3, "force_expert": True,
                "lr_head": 8.0e-5, "lr_backbone": 0.0, "lr_owl": 0.0,
                "weights": {"cls": 0.40, "branch": 0.15, "map": 0.05, "expert": 0.80, "calibration": 0.0},
            },
            "joint_finetune": {
                "epochs": 8, "force_expert": False,
                "lr_head": 3.0e-5, "lr_backbone": 5.0e-7, "lr_owl": 0.0,
                "weights": {"cls": 1.0, "branch": 0.25, "map": 0.22, "expert": 0.25, "calibration": 0.01},
            },
            "owl_finetune": {
                "epochs": 4, "force_expert": False,
                "lr_head": 1.8e-5, "lr_backbone": 2.5e-7, "lr_owl": 1.0e-7,
                "weights": {"cls": 1.0, "branch": 0.22, "map": 0.25, "expert": 0.20, "calibration": 0.02},
            },
            "calibration": {
                "epochs": 3, "force_expert": False,
                "lr_head": 1.2e-5, "lr_backbone": 0.0, "lr_owl": 0.0,
                "weights": {"cls": 0.70, "branch": 0.10, "map": 0.0, "expert": 0.0, "calibration": 0.25},
            },
        },
        "optimizer": {"weight_decay": 1.0e-4},
        "run": {
            "mode": "start",              # start / resume / finetune
            "resume_path": "models_lite/latest.pth",
            "finetune_path": "models_lite/best.pth",
        },
    }
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


def read_expert_train_csvs(expert_dir: Path, expert_names: Sequence[str]) -> pd.DataFrame:
    parts = []
    for expert_name in expert_names:
        csv_path = expert_dir / f"expert_train_{expert_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"缺少专家训练集: {csv_path}")
        df = pd.read_csv(csv_path)
        df["expert_name"] = expert_name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def resolve_image_path(row: pd.Series, root: Path) -> Path:
    """优先使用 relative_path，方便迁移和多机训练；绝对 path 只作为兼容兜底。"""
    rel = _clean_text(row.get("relative_path", ""))
    if rel:
        p = root / rel
        if p.exists():
            return p
    path_text = _clean_text(row.get("path", ""))
    if path_text:
        p = Path(path_text)
        if not p.is_absolute():
            p = root / path_text
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到图片: relative_path={row.get('relative_path', '')}, path={row.get('path', '')}")


def resize_image_to_target_height(image: Image.Image, target_height: Optional[int]) -> Image.Image:
    """
    训练缩放规则：只以高度为参考等比缩小。

    target_height=640 表示高于 640 的图缩到高 640，宽度按比例变化；
    原图高度小于等于 640 时保持原尺寸，不放大，也不做空白填充。
    """
    if target_height is None:
        return image
    target_h = int(target_height)
    if target_h <= 0:
        return image
    width, height = image.size
    if height <= 0 or height <= target_h:
        return image
    scale = float(target_h) / float(height)
    new_width = max(1, int(round(width * scale)))
    new_height = int(target_h)
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image, dtype=np.uint8, copy=True).transpose(2, 0, 1)
    x = torch.from_numpy(arr).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std.clamp_min(1e-6)


def load_image_pil(row: pd.Series, root: Path, image_target_height: Optional[int] = None) -> Image.Image:
    image_path = resolve_image_path(row, root)
    image = Image.open(image_path).convert("RGB")
    return resize_image_to_target_height(image, image_target_height)


def load_image_path_pil(image_path: Path, image_target_height: Optional[int] = None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return resize_image_to_target_height(image, image_target_height)


def load_image_tensor(row: pd.Series, root: Path, image_target_height: Optional[int] = None) -> torch.Tensor:
    return pil_to_normalized_tensor(load_image_pil(row, root, image_target_height))


def _clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _normalize_path_key(value: str) -> str:
    return str(value).replace("\\", "/").strip()


class ImageDegradationAugmentor:
    """
    Online JPEG/downsample/blur degradation for robustness training.

    SVD/PCA low-rank degradation is intentionally excluded from the online path:
    it is much more expensive and should be generated offline only for a small
    hard-negative pool if needed.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.no_fire_prob = float(cfg.get("no_fire_prob", 0.50))
        self.fire_prob = float(cfg.get("fire_prob", 0.30))
        self.no_fire_stage_weights = list(cfg.get("no_fire_stage_weights", [0.0, 0.0, 1.2, 1.4, 1.4, 1.2, 0.4, 0.4, 0.9, 0.8, 0.6, 0.2, 0.1]))
        self.fire_stage_weights = list(cfg.get("fire_stage_weights", [0.0, 0.8, 0.8, 0.6, 0.4, 0.2, 0.1]))
        self.max_stage = int(cfg.get("max_stage", 10))

    @staticmethod
    def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(quality), optimize=False, progressive=False, subsampling=2)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def _down_up(image: Image.Image, scale: float) -> Image.Image:
        width, height = image.size
        new_w = max(1, int(round(width * float(scale))))
        new_h = max(1, int(round(height * float(scale))))
        small = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return small.resize((width, height), Image.Resampling.BILINEAR)

    def _sample_stage(self, label: int) -> int:
        weights = self.fire_stage_weights if int(label) == 1 else self.no_fire_stage_weights
        if not weights:
            return 0
        stage_count = min(len(weights) - 1, max(0, self.max_stage))
        choices = list(range(stage_count + 1))
        weights = [max(0.0, float(weights[i])) for i in choices]
        if sum(weights) <= 0:
            return 0
        return int(random.choices(choices, weights=weights, k=1)[0])

    @staticmethod
    def _params_for_stage(stage: int, label: int) -> Dict[str, float]:
        # Stages 2-5 and 8-10 are deliberately emphasized because they are the
        # observed fire-cloud false-positive band.
        table = {
            0: {"jpeg_quality": 92},
            1: {"jpeg_quality": 82},
            2: {"jpeg_quality": random.uniform(68, 78)},
            3: {"downscale": random.uniform(0.78, 0.90), "jpeg_quality": random.uniform(62, 74)},
            4: {"downscale": random.uniform(0.66, 0.78), "blur": random.uniform(0.25, 0.55), "jpeg_quality": random.uniform(55, 68)},
            5: {"downscale": random.uniform(0.54, 0.68), "blur": random.uniform(0.45, 0.85), "jpeg_quality": random.uniform(48, 60), "saturation": random.uniform(0.96, 1.08)},
            6: {"downscale": random.uniform(0.46, 0.58), "blur": random.uniform(0.75, 1.15), "jpeg_quality": random.uniform(42, 54)},
            7: {"downscale": random.uniform(0.40, 0.52), "blur": random.uniform(0.90, 1.35), "jpeg_quality": random.uniform(38, 50)},
            8: {"downscale": random.uniform(0.34, 0.46), "blur": random.uniform(1.10, 1.65), "jpeg_quality": random.uniform(32, 44), "saturation": random.uniform(1.00, 1.12)},
            9: {"downscale": random.uniform(0.28, 0.38), "blur": random.uniform(1.45, 2.10), "jpeg_quality": random.uniform(26, 36), "contrast": random.uniform(0.88, 0.98)},
            10: {"downscale": random.uniform(0.22, 0.32), "blur": random.uniform(1.90, 2.70), "jpeg_quality": random.uniform(20, 30), "saturation": random.uniform(1.02, 1.16)},
            11: {"downscale": random.uniform(0.18, 0.25), "blur": random.uniform(2.50, 3.40), "jpeg_quality": random.uniform(14, 22), "contrast": random.uniform(0.82, 0.92)},
            12: {"downscale": random.uniform(0.14, 0.20), "blur": random.uniform(3.20, 4.20), "jpeg_quality": random.uniform(10, 16), "contrast": random.uniform(0.78, 0.88)},
        }
        params = dict(table.get(int(stage), table[0]))
        if int(label) == 1:
            if "blur" in params:
                params["blur"] = min(float(params["blur"]), 0.85)
            if "downscale" in params:
                params["downscale"] = max(float(params["downscale"]), 0.50)
            if "jpeg_quality" in params:
                params["jpeg_quality"] = max(float(params["jpeg_quality"]), 42.0)
        return params

    def __call__(self, image: Image.Image, label: int) -> Image.Image:
        if not self.enabled:
            return image
        prob = self.fire_prob if int(label) == 1 else self.no_fire_prob
        if random.random() >= max(0.0, min(1.0, float(prob))):
            return image
        stage = self._sample_stage(int(label))
        if stage <= 0:
            return image
        params = self._params_for_stage(stage, int(label))
        out = image
        if "downscale" in params:
            out = self._down_up(out, float(params["downscale"]))
        if "blur" in params and float(params["blur"]) > 0:
            out = out.filter(ImageFilter.GaussianBlur(float(params["blur"])))
        if "saturation" in params:
            out = ImageEnhance.Color(out).enhance(float(params["saturation"]))
        if "contrast" in params:
            out = ImageEnhance.Contrast(out).enhance(float(params["contrast"]))
        out = self._jpeg_roundtrip(out, int(round(float(params.get("jpeg_quality", 85)))))
        return out


class OfflineDegradationCache:
    """按划分阶段生成的降质图片索引，训练时按概率替换原图读取。

    重要：初始化阶段不做 Path.exists()/Path.resolve()。
    大规模 degraded cache 在 DDP 下会被每个 rank 初始化一次；如果逐行 stat/resolve，
    会在训练开始前卡很久。这里仅构建字符串索引，真正抽到缓存图时再尝试读取；
    读取失败则在 FireDataset.__getitem__ 中回退原图。
    """

    def __init__(self, root: Path, config: Optional[Dict] = None):
        cfg = dict(config or {})
        self.root = Path(root)
        self.enabled = bool(cfg.get("offline_cache_enabled", False))
        self.no_fire_prob = float(cfg.get("offline_cache_no_fire_prob", cfg.get("no_fire_prob", 0.0)))
        self.fire_prob = float(cfg.get("offline_cache_fire_prob", cfg.get("fire_prob", 0.0)))
        self.require_label_match = bool(cfg.get("offline_cache_require_label_match", True))
        self.by_key: Dict[str, List[Dict]] = {}
        self.available = False
        if not self.enabled:
            return

        index_text = _clean_text(cfg.get("offline_cache_index_csv", ""))
        index_path = Path(index_text) if index_text else self.root / "fire_splits" / "degraded_image_cache_index.csv"
        if not index_path.is_absolute():
            index_path = self.root / index_path
        # 这里只检查索引 csv 本身是否存在；不检查每一张 degraded 图片。
        if not index_path.exists():
            print(f"[提示] 未找到离线模糊缓存索引，训练将读取原图: {index_path}")
            return

        try:
            df = pd.read_csv(index_path)
        except Exception as exc:
            print(f"[警告] 离线模糊缓存索引读取失败，训练将读取原图: {index_path} ({exc})")
            return

        needed = {
            "degraded_path",
            "relative_degraded_path",
            "original_path",
            "relative_original_path",
            "label",
            "degradation_stage",
        }
        for col in needed:
            if col not in df.columns:
                df[col] = ""

        # itertuples 比 iterrows 更快；这里全部按字符串建索引，不做磁盘 stat/resolve。
        for rec in df[list(needed)].itertuples(index=False, name=None):
            row = dict(zip(list(needed), rec))
            degraded_path = self._resolve_cached_path(row)
            if degraded_path is None:
                continue
            item = {
                "path": degraded_path,
                "stage": self._safe_int(row.get("degradation_stage", ""), -1),
                "label": self._safe_int(row.get("label", ""), -1),
            }
            for key in self._index_keys(row):
                self.by_key.setdefault(key, []).append(item)

        self.available = any(self.by_key.values())
        if self.available:
            total = sum(len(v) for v in self.by_key.values())
            print(f"已加载离线模糊缓存索引(懒检查): keys={len(self.by_key)}, variants={total}")
        else:
            print(f"[提示] 离线模糊缓存索引为空，训练将读取原图: {index_path}")

    @staticmethod
    def _safe_int(value, default: int = -1) -> int:
        text = _clean_text(value)
        if not text:
            return int(default)
        try:
            return int(float(text))
        except Exception:
            return int(default)

    def _path_from_text(self, path_text: str) -> Optional[Path]:
        text = _clean_text(path_text)
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = self.root / text
        return path

    def _resolve_cached_path(self, row) -> Optional[Path]:
        # 不做 path.exists()。只把 CSV 中的路径解析成 Path，读取失败时再回退。
        rel_text = _clean_text(row.get("relative_degraded_path", ""))
        if rel_text:
            return self.root / rel_text
        path_text = _clean_text(row.get("degraded_path", ""))
        return self._path_from_text(path_text)

    def _index_keys(self, row) -> List[str]:
        # 不做 Path.resolve()/exists()，只生成稳定字符串 key。
        keys = []
        rel_text = _clean_text(row.get("relative_original_path", ""))
        if rel_text:
            keys.append(_normalize_path_key(rel_text))
            keys.append(_normalize_path_key(str(self.root / rel_text)))
        path_text = _clean_text(row.get("original_path", ""))
        if path_text:
            keys.append(_normalize_path_key(path_text))
            path = Path(path_text)
            if not path.is_absolute():
                keys.append(_normalize_path_key(str(self.root / path_text)))
        return list(dict.fromkeys(keys))

    def _row_keys(self, row: pd.Series) -> List[str]:
        # 和 _index_keys 保持同一套字符串规则，不触发磁盘访问。
        keys = []
        rel_text = _clean_text(row.get("relative_path", ""))
        if rel_text:
            keys.append(_normalize_path_key(rel_text))
            keys.append(_normalize_path_key(str(self.root / rel_text)))
        path_text = _clean_text(row.get("path", ""))
        if path_text:
            keys.append(_normalize_path_key(path_text))
            path = Path(path_text)
            if not path.is_absolute():
                keys.append(_normalize_path_key(str(self.root / path_text)))
        return list(dict.fromkeys(keys))

    def sample_path(self, row: pd.Series, label: int) -> Tuple[Optional[Path], Optional[int]]:
        if not self.enabled or not self.available:
            return None, None
        prob = self.fire_prob if int(label) == 1 else self.no_fire_prob
        if random.random() >= max(0.0, min(1.0, float(prob))):
            return None, None
        candidates = []
        for key in self._row_keys(row):
            candidates.extend(self.by_key.get(key, []))
        if self.require_label_match:
            candidates = [item for item in candidates if int(item.get("label", -1)) == int(label)]
        if not candidates:
            return None, None
        picked = random.choice(candidates)
        return Path(picked["path"]), int(picked.get("stage", -1))


def resolve_mask_npz_path(row: pd.Series, root: Path) -> Optional[Path]:
    rel_text = _clean_text(row.get("relative_mask_npz_path", ""))
    if rel_text:
        path = root / rel_text
        if path.exists():
            return path
    path_text = _clean_text(row.get("mask_npz_path", ""))
    if path_text:
        path = Path(path_text)
        if not path.is_absolute():
            path = root / path_text
        if path.exists():
            return path
    return None


def resize_cached_mask(mask: np.ndarray, target_size) -> torch.Tensor:
    if isinstance(target_size, tuple):
        target_h, target_w = int(target_size[0]), int(target_size[1])
    else:
        target_h = target_w = int(target_size)
    if mask.ndim != 2:
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
    tensor = torch.from_numpy((mask > 0).astype(np.float32)).view(1, 1, int(mask.shape[0]), int(mask.shape[1]))
    if tuple(tensor.shape[-2:]) != (target_h, target_w):
        tensor = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
    return tensor.view(1, target_h, target_w)


def make_attention_target(row: pd.Series, root: Path, target_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    只读取划分/缓存阶段生成的 positive/negative mask。

    图片级 label 始终负责分类监督；mask 只负责 attention 监督。
    若 label=0 但缓存里存在 positive mask，训练端忽略 positive，只保留 negative。
    """
    if isinstance(target_size, tuple):
        target_h, target_w = int(target_size[0]), int(target_size[1])
    else:
        target_h = target_w = int(target_size)
    zero = torch.zeros(1, target_h, target_w)
    mask_path = resolve_mask_npz_path(row, root)
    if mask_path is None:
        return zero, zero, torch.tensor(False), torch.tensor(False)

    try:
        with np.load(mask_path) as data:
            pos_arr = np.asarray(data.get("positive_mask", np.zeros((target_h, target_w), dtype=np.uint8)))
            neg_arr = np.asarray(data.get("negative_mask", np.zeros((target_h, target_w), dtype=np.uint8)))
    except Exception as exc:
        print(f"[警告] mask npz 读取失败，跳过 attention 监督: {mask_path} ({exc})")
        return zero, zero, torch.tensor(False), torch.tensor(False)

    pos_mask = resize_cached_mask(pos_arr, target_size)
    neg_mask = resize_cached_mask(neg_arr, target_size)
    label = int(row.get("label", 0))
    pos_valid = bool(label == 1 and pos_mask.sum().item() > 0)
    neg_valid = bool(neg_mask.sum().item() > 0)
    if pos_valid and neg_valid:
        neg_mask = (neg_mask * (1.0 - pos_mask)).clamp(0.0, 1.0)
        neg_valid = bool(neg_mask.sum().item() > 0)
    if not pos_valid:
        pos_mask = zero.clone()
    if not neg_valid:
        neg_mask = zero.clone()
    return pos_mask, neg_mask, torch.tensor(pos_valid), torch.tensor(neg_valid)


class FireDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: Path,
        expert_to_index: Dict[str, int],
        generic_expert_names: Sequence[str] = (),
        image_target_height: Optional[int] = None,
        degradation_config: Optional[Dict] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.expert_to_index = dict(expert_to_index)
        self.generic_expert_names = {str(name) for name in generic_expert_names}
        self.image_target_height = image_target_height
        self.degrader = ImageDegradationAugmentor(degradation_config)
        self.degradation_cache = OfflineDegradationCache(root, degradation_config)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        expert_raw = row.get("expert_name", None)
        expert_name = str(expert_raw) if expert_raw is not None and not pd.isna(expert_raw) else ""
        expert_valid = expert_name in self.expert_to_index and expert_name not in self.generic_expert_names
        label_value = int(row.get("label", 0))
        cached_image_path, cached_stage = self.degradation_cache.sample_path(row, label_value)
        if cached_image_path is not None:
            try:
                image_pil = load_image_path_pil(cached_image_path, self.image_target_height)
            except Exception:
                # 离线 degraded cache 现在采用懒检查：缓存图缺失/损坏时回退原图。
                cached_image_path, cached_stage = None, None
                image_pil = load_image_pil(row, self.root, self.image_target_height)
                image_pil = self.degrader(image_pil, label_value)
        else:
            image_pil = load_image_pil(row, self.root, self.image_target_height)
            image_pil = self.degrader(image_pil, label_value)
        image = pil_to_normalized_tensor(image_pil)
        image_h, image_w = int(image.shape[1]), int(image.shape[2])
        attn_pos_mask, attn_neg_mask, attn_pos_valid, attn_neg_valid = make_attention_target(row, self.root, (image_h, image_w))
        return {
            "image": image,
            "label": torch.tensor(float(label_value), dtype=torch.float32),
            "expert_index": torch.tensor(self.expert_to_index.get(expert_name, 0), dtype=torch.long),
            "expert_valid": torch.tensor(bool(expert_valid), dtype=torch.bool),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
            "attention_pos_mask": attn_pos_mask.float(),
            "attention_neg_mask": attn_neg_mask.float(),
            "attention_pos_valid": attn_pos_valid.bool(),
            "attention_neg_valid": attn_neg_valid.bool(),
            "path": str(row.get("path", "")),
            "degraded_path": str(cached_image_path) if cached_image_path is not None else "",
            "degradation_stage": torch.tensor(int(cached_stage) if cached_stage is not None else -1, dtype=torch.long),
        }


def _pad_to_multiple(value: int, multiple: int = 32) -> int:
    return int(math.ceil(int(value) / float(multiple)) * int(multiple))


def fire_collate_fn(batch: Sequence[Dict]) -> Dict:
    max_h = _pad_to_multiple(max(int(item["image"].shape[1]) for item in batch), 32)
    max_w = _pad_to_multiple(max(int(item["image"].shape[2]) for item in batch), 32)

    images = []
    valid_masks = []
    attn_pos_masks = []
    attn_neg_masks = []
    out: Dict[str, list] = {}
    for item in batch:
        image = item["image"]
        _, h, w = image.shape
        padded = image.new_zeros((3, max_h, max_w))
        padded[:, :h, :w] = image
        mask = image.new_zeros((1, max_h, max_w))
        mask[:, :h, :w] = 1.0
        attn_pos = image.new_zeros((1, max_h, max_w))
        attn_neg = image.new_zeros((1, max_h, max_w))
        attn_pos[:, :h, :w] = item["attention_pos_mask"]
        attn_neg[:, :h, :w] = item["attention_neg_mask"]
        images.append(padded)
        valid_masks.append(mask)
        attn_pos_masks.append(attn_pos)
        attn_neg_masks.append(attn_neg)
        for key, value in item.items():
            if key in {"image", "attention_pos_mask", "attention_neg_mask"}:
                continue
            out.setdefault(key, []).append(value)

    collated: Dict = {
        "image": torch.stack(images, dim=0),
        "image_valid_mask": torch.stack(valid_masks, dim=0),
        "attention_pos_mask": torch.stack(attn_pos_masks, dim=0),
        "attention_neg_mask": torch.stack(attn_neg_masks, dim=0),
    }
    for key, values in out.items():
        first = values[0]
        if torch.is_tensor(first):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = list(values)
    return collated


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    root: Path,
    expert_to_index: Dict[str, int],
    generic_expert_names: Sequence[str],
    image_target_height: Optional[int],
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    train_degradation_config: Optional[Dict] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    train_ds = FireDataset(train_df, root, expert_to_index, generic_expert_names, image_target_height, train_degradation_config)
    val_ds = FireDataset(val_df, root, expert_to_index, generic_expert_names, image_target_height)
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=int(world_size),
        rank=int(rank),
        shuffle=True,
        drop_last=False,
    ) if distributed else None
    common = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(persistent_workers and num_workers > 0),
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        collate_fn=fire_collate_fn,
        **common,
    )
    # 验证集默认每个进程都可构建；running_train 只让 rank0 真正执行验证，避免重复写日志和重复评估。
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        collate_fn=fire_collate_fn,
        **common,
    )
    return train_loader, val_loader, train_sampler




def init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> FireArbiterMoELite:
    return model.module if isinstance(model, DDP) else model


def weighted_bce(logit: torch.Tensor, label: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logit.view(-1), label.view(-1), reduction="none")
    return (raw * sample_weight.view(-1)).sum() / sample_weight.sum().clamp_min(1e-6)


def expand_mask(mask: torch.Tensor, ratio: float = 0.12) -> torch.Tensor:
    size = max(mask.shape[-2:])
    kernel = max(3, int(round(size * ratio)))
    if kernel % 2 == 0:
        kernel += 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=kernel // 2)


def tolerant_map_loss(
    attention: torch.Tensor,
    batch: Dict,
    negative_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pos = F.interpolate(batch["attention_pos_mask"].float(), size=attention.shape[-2:], mode="nearest")
    neg = F.interpolate(batch["attention_neg_mask"].float(), size=attention.shape[-2:], mode="nearest")
    valid = F.interpolate(batch["image_valid_mask"].float(), size=attention.shape[-2:], mode="nearest")
    pos_valid = batch["attention_pos_valid"].view(-1).bool()
    neg_valid = batch["attention_neg_valid"].view(-1).bool()
    attention = attention.clamp(0.0, 1.0) * valid

    zero = attention.sum() * 0.0
    pos_loss = zero
    inside_ratio = zero.detach()
    far_leak = zero.detach()
    if pos_valid.any():
        att = attention[pos_valid]
        target = pos[pos_valid]
        allowed = expand_mask(target)
        inside_mean = (att * target).sum((1, 2, 3)) / target.sum((1, 2, 3)).clamp_min(1.0)
        inside_peak = (att * target).flatten(1).amax(dim=1)
        soft_coverage = (torch.sigmoid(12.0 * (att - 0.5)) * target).sum((1, 2, 3)) / target.sum((1, 2, 3)).clamp_min(1.0)
        outside = (valid[pos_valid] - allowed).clamp_min(0.0)
        far = (att * outside).sum((1, 2, 3)) / outside.sum((1, 2, 3)).clamp_min(1.0)
        energy = (att * target).sum((1, 2, 3)) / att.sum((1, 2, 3)).clamp_min(1e-6)
        pos_loss = (
            F.relu(0.40 - inside_mean).pow(2)
            + 0.35 * F.relu(0.70 - inside_peak).pow(2)
            + 0.40 * F.relu(0.20 - soft_coverage).pow(2)
            + 0.25 * far
        ).mean()
        inside_ratio = energy.mean().detach()
        far_leak = far.mean().detach()

    neg_loss = zero
    if neg_valid.any():
        att = attention[neg_valid]
        target = neg[neg_valid]
        neg_loss = ((att * target).sum((1, 2, 3)) / target.sum((1, 2, 3)).clamp_min(1.0)).mean()

    return pos_loss + negative_weight * neg_loss, {
        "inside_ratio": inside_ratio,
        "far_leak": far_leak,
    }


def compute_losses(output: Dict[str, torch.Tensor], batch: Dict, weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    label = batch["label"].float()
    sample_weight = batch["sample_weight"].float()
    cls = weighted_bce(output["logit"], label, sample_weight)
    branch = (
        0.35 * weighted_bce(output["global_logit"], label, sample_weight)
        + 0.45 * weighted_bce(output["local_logit"], label, sample_weight)
        + 0.20 * weighted_bce(output["semantic_logit"], label, sample_weight)
    )

    map_terms = []
    map_stats: Dict[str, torch.Tensor] = {}
    for name, coefficient, neg_weight in (
        ("global", 0.20, 1.00),
        ("local", 0.65, 1.00),
        ("semantic", 0.75, 0.45),
        ("mix", 1.00, 1.00),
    ):
        value, stats = tolerant_map_loss(output[f"attention_{name}"], batch, negative_weight=neg_weight)
        map_terms.append(coefficient * value)
        map_stats[f"{name}_inside_ratio"] = stats["inside_ratio"]
        map_stats[f"{name}_far_leak"] = stats["far_leak"]
    map_loss = sum(map_terms)

    expert_logits = output["expert_logits"]
    expanded_label = label.view(-1, 1).expand_as(expert_logits)
    expert_error = F.binary_cross_entropy_with_logits(expert_logits, expanded_label, reduction="none")
    posterior = output["expert_posterior_weights"].clamp_min(1e-7)
    prior = output["expert_prior_weights"].clamp_min(1e-7)
    router_target = torch.softmax(-expert_error.detach() / 0.5, dim=1)
    posterior_kl = F.kl_div(posterior.log(), router_target, reduction="batchmean")
    confidence_target = torch.exp(-expert_error.detach())
    confidence_loss = F.smooth_l1_loss(output["expert_confidence"], confidence_target)
    routed_loss = weighted_bce(output["posterior_routed_logit"], label, sample_weight)
    prior_guidance = output["logit"].sum() * 0.0
    expert_valid = batch["expert_valid"].bool()
    if expert_valid.any():
        indices = batch["expert_index"][expert_valid].long()
        prior_guidance = F.nll_loss(prior[expert_valid].log(), indices)
        selected_logits = expert_logits[expert_valid].gather(1, indices.unsqueeze(1)).squeeze(1)
        selected_loss = F.binary_cross_entropy_with_logits(selected_logits, label[expert_valid])
    else:
        selected_loss = output["logit"].sum() * 0.0
    mean_weight = posterior.mean(dim=0)
    uniform = torch.full_like(mean_weight, 1.0 / mean_weight.numel())
    balance = F.kl_div(mean_weight.log(), uniform, reduction="sum")
    expert = selected_loss + 0.5 * routed_loss + 0.5 * posterior_kl + 0.25 * confidence_loss + 0.15 * prior_guidance + 0.02 * balance

    calibration = (torch.sigmoid(output["logit"].view(-1)) - label).pow(2).mean()
    total = (
        float(weights["cls"]) * cls
        + float(weights["branch"]) * branch
        + float(weights["map"]) * map_loss
        + float(weights["expert"]) * expert
        + float(weights["calibration"]) * calibration
    )
    return {
        "total": total,
        "cls": cls,
        "branch": branch,
        "map": map_loss,
        "expert": expert,
        "calibration": calibration,
        "expert_selected": selected_loss,
        "expert_posterior_kl": posterior_kl,
        "expert_confidence": confidence_loss,
        **map_stats,
    }


def binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = probs >= threshold
    truth = labels >= 0.5
    tp = float(np.logical_and(pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "acc": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": 0.5 * (recall + specificity),
    }


def binary_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    positive = labels >= 0.5
    negative = ~positive
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    total = max(len(probs), 1)
    value = 0.0
    for left in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        right = left + 1.0 / bins
        selected = (probs >= left) & (probs < right if right < 1.0 else probs <= right)
        if selected.any():
            value += selected.sum() / total * abs(float(probs[selected].mean()) - float(labels[selected].mean()))
    return float(value)


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = binary_metrics(probs, labels, float(threshold))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(threshold), float(f1)
    return best_t, best_f1


class EpochCollector:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.loss_sums: Dict[str, float] = {}
        self.probs: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.diag_sums: Dict[str, float] = {}
        self.expert_correct = np.zeros(self.num_experts, dtype=np.float64)
        self.expert_count = np.zeros(self.num_experts, dtype=np.float64)

    def update(self, output: Dict[str, torch.Tensor], batch: Dict, losses: Dict[str, torch.Tensor]) -> None:
        batch_size = int(batch["label"].shape[0])
        self.count += batch_size
        for key, value in losses.items():
            if torch.is_tensor(value):
                self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        self.probs.append(output["prob_fire"].detach().view(-1).cpu().numpy())
        self.labels.append(batch["label"].detach().view(-1).cpu().numpy())

        semantic_scores = output["semantic_scores"].detach()
        prior = output["expert_prior_weights"].detach()
        posterior = output["expert_posterior_weights"].detach()
        confidence = output["expert_confidence"].detach()
        expert_probs = output["expert_probs"].detach()
        labels = batch["label"].view(-1, 1).to(expert_probs.device)
        diagnostics = {
            "semantic_positive": semantic_scores[:, 0].mean(),
            "semantic_negative": semantic_scores[:, 1].mean(),
            "semantic_suspicious": semantic_scores[:, 2].mean(),
            "semantic_confidence": output["semantic_confidence"].mean(),
            "global_weight": output["branch_weights"][:, 0].mean(),
            "local_weight": output["branch_weights"][:, 1].mean(),
            "router_prior_entropy": -(prior * prior.clamp_min(1e-7).log()).sum(1).mean(),
            "router_posterior_entropy": -(posterior * posterior.clamp_min(1e-7).log()).sum(1).mean(),
            "router_change_rate": (prior.argmax(1) != posterior.argmax(1)).float().mean(),
            "posterior_improvement": torch.sigmoid(output["prior_routed_logit"]).sub(labels).abs().mean()
                                     - torch.sigmoid(output["posterior_routed_logit"]).sub(labels).abs().mean(),
            "expert_confidence_mean": confidence.mean(),
        }
        for key in ("global_inside_ratio", "local_inside_ratio", "semantic_inside_ratio", "mix_inside_ratio",
                    "global_far_leak", "local_far_leak", "semantic_far_leak", "mix_far_leak"):
            diagnostics[key] = losses[key]
        for key, value in diagnostics.items():
            self.diag_sums[key] = self.diag_sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size

        expert_pred = expert_probs >= 0.5
        expert_true = labels >= 0.5
        correct = (expert_pred == expert_true).float().sum(dim=0).cpu().numpy()
        self.expert_correct += correct
        self.expert_count += batch_size
        for index in range(self.num_experts):
            for prefix, tensor in (("prior", prior), ("posterior", posterior), ("confidence", confidence)):
                key = f"expert_{index}_{prefix}_weight" if prefix != "confidence" else f"expert_{index}_confidence"
                self.diag_sums[key] = self.diag_sums.get(key, 0.0) + float(tensor[:, index].mean().cpu()) * batch_size

    def compute(self, threshold: float) -> Dict[str, float]:
        probs = np.concatenate(self.probs) if self.probs else np.zeros(0, dtype=np.float32)
        labels = np.concatenate(self.labels) if self.labels else np.zeros(0, dtype=np.float32)
        metrics = binary_metrics(probs, labels, threshold) if len(probs) else {}
        metrics["auc"] = binary_auc(probs, labels) if len(probs) else float("nan")
        metrics["ece"] = expected_calibration_error(probs, labels) if len(probs) else float("nan")
        metrics["brier"] = float(np.mean((probs - labels) ** 2)) if len(probs) else float("nan")
        bt, bf1 = best_threshold(probs, labels) if len(probs) else (threshold, float("nan"))
        metrics["best_threshold"] = bt
        metrics["best_f1"] = bf1
        for key, total in self.loss_sums.items():
            metrics["loss" if key == "total" else f"loss_{key}"] = total / max(self.count, 1)
        for key, total in self.diag_sums.items():
            metrics[key] = total / max(self.count, 1)
        for index in range(self.num_experts):
            metrics[f"expert_{index}_accuracy"] = float(self.expert_correct[index] / max(self.expert_count[index], 1.0))
        return metrics


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_optimizer(model: FireArbiterMoELite, stage: Dict, weight_decay: float):
    groups = {"head": [], "backbone": [], "owl": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            groups["backbone"].append(parameter)
        elif name.startswith("semantic_encoder.model."):
            groups["owl"].append(parameter)
        else:
            groups["head"].append(parameter)
    parameter_groups = []
    for key, lr_key in (("head", "lr_head"), ("backbone", "lr_backbone"), ("owl", "lr_owl")):
        if groups[key] and float(stage[lr_key]) > 0.0:
            parameter_groups.append({"params": groups[key], "lr": float(stage[lr_key]), "name": key})
    if not parameter_groups:
        raise RuntimeError("No trainable parameters for the current stage.")
    return torch.optim.AdamW(parameter_groups, weight_decay=float(weight_decay))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: Dict[str, float],
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    stage: str,
    stage_epoch: int,
    stage_total_epochs: int,
    force_expert: bool,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    core = unwrap_model(model)
    collector = EpochCollector(core.num_experts)
    progress = tqdm(
        loader,
        desc=(
            f"{'Train' if training else 'Val  '} "
            f"E{epoch:03d} ({stage_epoch}/{stage_total_epochs}) {stage}"
        ),
        disable=not is_main_process(),
    )
    running_tp = running_fp = running_fn = running_correct = running_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in progress:
            batch = move_batch(batch, device)
            forced = None
            if force_expert:
                forced = torch.where(batch["expert_valid"], batch["expert_index"], torch.full_like(batch["expert_index"], -1))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["image"], batch["image_valid_mask"], forced)
                losses = compute_losses(output, batch, weights)
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            collector.update(output, batch, losses)

            pred = output["prob_fire"].detach().view(-1) >= core.get_threshold()
            truth = batch["label"].view(-1) >= 0.5
            running_correct += int((pred == truth).sum())
            running_tp += int((pred & truth).sum())
            running_fp += int((pred & ~truth).sum())
            running_fn += int((~pred & truth).sum())
            running_count += int(truth.numel())
            precision = running_tp / max(running_tp + running_fp, 1)
            recall = running_tp / max(running_tp + running_fn, 1)
            running_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            progress.set_postfix(
                loss=f"{float(losses['total'].detach()):.4f}",
                acc=f"{running_correct / max(running_count, 1):.4f}",
                f1=f"{running_f1:.4f}",
            )

            # Collector只保留CPU标量/数组；及时解除完整输出图和batch的引用，
            # 避免上一批计算图存活到下一批前向期间而抬高峰值显存。
            del pred, truth, output, losses, batch, forced

    return collector.compute(core.get_threshold())


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_history(output_dir: Path, history: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "history.json", {"epochs": history})


def inference_checkpoint_path(path: Path) -> Path:
    suffix = path.suffix or ".pth"
    return path.with_name(f"{path.stem}_inference{suffix}")


def save_checkpoint(
    model: FireArbiterMoELite,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    path: Path,
    epoch: int,
    stage: str,
    stage_epoch: int,
    stage_total_epochs: int,
    history: List[Dict],
    metadata: Dict,
    best_score: float,
    best_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expert_names = list(model.expert_names)
    export_metadata = {
        **metadata,
        "epoch": int(epoch),
        "stage": stage,
        "stage_epoch": int(stage_epoch),
        "stage_total_epochs": int(stage_total_epochs),
        "expert_names": expert_names,
        "num_experts": int(model.num_experts),
    }

    # 完整训练断点：用于精确恢复阶段、优化器、调度器和混合精度状态。
    checkpoint = model.build_checkpoint(export_metadata)
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint["scaler_state_dict"] = scaler.state_dict()
    checkpoint["epoch"] = int(epoch)
    checkpoint["stage"] = stage
    checkpoint["stage_epoch"] = int(stage_epoch)
    checkpoint["stage_total_epochs"] = int(stage_total_epochs)
    checkpoint["history"] = history
    checkpoint["best_score"] = float(best_score)
    checkpoint["best_f1"] = float(best_f1)
    checkpoint["expert_names"] = expert_names
    torch.save(checkpoint, path)

    # 纯推理模型：保留模型结构、权重、阈值、metadata和专家名称，
    # 不包含优化器、调度器、GradScaler、history等训练状态。
    inference_checkpoint = model.build_checkpoint(export_metadata)
    inference_checkpoint["expert_names"] = expert_names
    torch.save(inference_checkpoint, inference_checkpoint_path(path))


def load_checkpoint(model: FireArbiterMoELite, path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != model.CHECKPOINT_FORMAT:
        raise RuntimeError(f"Only {model.CHECKPOINT_FORMAT} checkpoints are supported.")
    model.load_state_dict(checkpoint["state_dict"])
    model.set_threshold(float(checkpoint.get("threshold", 0.5)))
    return checkpoint


def denormalize_image(image: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    mask = valid_mask[0] > 0.5
    rows = torch.where(mask.any(dim=1))[0]
    cols = torch.where(mask.any(dim=0))[0]
    height = int(rows[-1]) + 1
    width = int(cols[-1]) + 1
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    rgb = (image[:, :height, :width] * std + mean).clamp(0, 1)
    return (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def save_expert_diagnostics_csv(
    output_dir: Path,
    epoch: int,
    stage: str,
    expert_names: Sequence[str],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    rows = []
    for index, name in enumerate(expert_names):
        rows.append({
            "epoch": int(epoch),
            "stage": stage,
            "expert_index": int(index),
            "expert_name": str(name),
            "train_prior_weight": float(train_metrics.get(f"expert_{index}_prior_weight", float("nan"))),
            "train_posterior_weight": float(train_metrics.get(f"expert_{index}_posterior_weight", float("nan"))),
            "train_confidence": float(train_metrics.get(f"expert_{index}_confidence", float("nan"))),
            "train_accuracy": float(train_metrics.get(f"expert_{index}_accuracy", float("nan"))),
            "val_prior_weight": float(val_metrics.get(f"expert_{index}_prior_weight", float("nan"))),
            "val_posterior_weight": float(val_metrics.get(f"expert_{index}_posterior_weight", float("nan"))),
            "val_confidence": float(val_metrics.get(f"expert_{index}_confidence", float("nan"))),
            "val_accuracy": float(val_metrics.get(f"expert_{index}_accuracy", float("nan"))),
        })
    folder = output_dir / "expert_diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        folder / f"epoch_{epoch:03d}_{stage}.csv", index=False, encoding="utf-8-sig"
    )


def save_attention_sample(
    model: FireArbiterMoELite,
    val_df: pd.DataFrame,
    root: Path,
    expert_to_index: Dict[str, int],
    image_target_height: Optional[int],
    device: torch.device,
    output_dir: Path,
    epoch: int,
    stage: str,
    sample_index: int,
) -> None:
    if len(val_df) == 0:
        return
    dataset = FireDataset(val_df.iloc[[sample_index % len(val_df)]], root, expert_to_index, (), image_target_height)
    batch = fire_collate_fn([dataset[0]])
    batch = move_batch(batch, device)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        output = model(batch["image"], batch["image_valid_mask"])
    rgb = denormalize_image(batch["image"][0], batch["image_valid_mask"][0])
    maps = [
        ("global", output["attention_global"]),
        ("local", output["attention_local"]),
        ("semantic", output["attention_semantic"]),
        ("mix", output["attention_mix"]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    for axis, (name, tensor) in zip(axes[1:], maps):
        attention = F.interpolate(tensor[0:1], size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        axis.imshow(rgb)
        axis.imshow(attention, alpha=0.48, cmap="jet", vmin=0.0, vmax=1.0)
        axis.set_title(name)
        axis.axis("off")
    fig.tight_layout()
    folder = output_dir / "attention_samples"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"epoch_{epoch:03d}_{stage}.png", dpi=150)
    plt.close(fig)

    summary = {
        "epoch": epoch,
        "stage": stage,
        "label": float(batch["label"][0].cpu()),
        "prob_fire": float(output["prob_fire"][0].cpu()),
        "global_probability": float(torch.sigmoid(output["global_logit"])[0].cpu()),
        "local_probability": float(torch.sigmoid(output["local_logit"])[0].cpu()),
        "semantic_probability": float(torch.sigmoid(output["semantic_logit"])[0].cpu()),
        "branch_weights": output["branch_weights"][0].cpu().tolist(),
        "semantic_scores": output["semantic_scores"][0].cpu().tolist(),
        "semantic_confidence": float(output["semantic_confidence"][0].cpu()),
        "prior_expert_weights": output["expert_prior_weights"][0].cpu().tolist(),
        "posterior_expert_weights": output["expert_posterior_weights"][0].cpu().tolist(),
        "expert_confidence": output["expert_confidence"][0].cpu().tolist(),
    }
    save_json(folder / f"epoch_{epoch:03d}_{stage}.json", summary)

    # 显式释放单样本可视化输出，并恢复调用前的训练状态。
    del output, batch, maps, summary, rgb
    model.train(was_training)


def selection_score(metrics: Dict[str, float]) -> float:
    def finite(name: str, default: float) -> float:
        value = float(metrics.get(name, default))
        return value if math.isfinite(value) else default

    return (
        0.45 * finite("balanced_acc", 0.0)
        + 0.30 * finite("f1", 0.0)
        + 0.15 * finite("auc", 0.0)
        - 0.05 * finite("ece", 1.0)
        - 0.05 * finite("mix_far_leak", 1.0)
    )


def _checkpoint_stage_epoch(checkpoint: Dict) -> int:
    value = checkpoint.get("stage_epoch")
    if value is not None:
        return int(value)
    metadata = checkpoint.get("metadata", {})
    value = metadata.get("stage_epoch") if isinstance(metadata, dict) else None
    if value is not None:
        return int(value)
    history = checkpoint.get("history", [])
    stage = checkpoint.get("stage")
    for row in reversed(history if isinstance(history, list) else []):
        if row.get("stage") == stage and row.get("stage_epoch") is not None:
            return int(row["stage_epoch"])
    return 0


def _checkpoint_stage_total(checkpoint: Dict) -> int:
    value = checkpoint.get("stage_total_epochs")
    if value is not None:
        return int(value)
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("stage_total_epochs")
        if value is not None:
            return int(value)
        stage_config = metadata.get("stage_config", {})
        if isinstance(stage_config, dict) and stage_config.get("epochs") is not None:
            return int(stage_config["epochs"])
    history = checkpoint.get("history", [])
    stage = checkpoint.get("stage")
    for row in reversed(history if isinstance(history, list) else []):
        if row.get("stage") != stage:
            continue
        if row.get("stage_total_epochs") is not None:
            return int(row["stage_total_epochs"])
    return 0


def _history_best_values(history: List[Dict]) -> Tuple[float, float]:
    best_score = float("-inf")
    best_f1 = float("-inf")
    for row in history:
        val_metrics = {
            key[4:]: value
            for key, value in row.items()
            if isinstance(key, str) and key.startswith("val_")
        }
        if val_metrics:
            best_score = max(best_score, selection_score(val_metrics))
        try:
            f1 = float(row.get("val_f1", float("nan")))
        except (TypeError, ValueError):
            f1 = float("nan")
        if math.isfinite(f1):
            best_f1 = max(best_f1, f1)
    return best_score, best_f1


def _restore_scheduler_without_state(scheduler, optimizer: torch.optim.Optimizer, completed_epochs: int) -> None:
    """兼容旧Checkpoint：保留优化器中的当前学习率，并对齐调度器轮次。"""
    scheduler.last_epoch = int(completed_epochs)
    scheduler._last_lr = [float(group["lr"]) for group in optimizer.param_groups]


def running_train(mode: str, checkpoint_path: Optional[str], config: Dict) -> None:
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["seed"]) + rank)

    root = Path(config["root"])
    train_csv = Path(config["train_csv"])
    val_csv = Path(config["val_csv"])
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Existing split CSV not found: {train_csv} / {val_csv}")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    expert_names = list(config["expert_names"])
    expert_to_index = {name: index for index, name in enumerate(expert_names)}

    train_output_dir = Path(config["train_output_dir"])
    model_output_dir = Path(config["model_output_dir"])
    train_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_config = dict(config["model"])
    model = FireArbiterMoELite(
        num_experts=len(expert_names),
        expert_names=expert_names,
        **model_config,
    ).to(device)

    history: List[Dict] = []
    start_epoch = 1
    checkpoint: Optional[Dict] = None
    resume_stage_name: Optional[str] = None
    resume_stage_epoch = 0
    resume_stage_total = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_scaler_state = None

    if mode in {"resume", "finetune"} and checkpoint_path:
        checkpoint = load_checkpoint(model, Path(checkpoint_path), device)
        saved_expert_names = checkpoint.get("expert_names")
        if saved_expert_names is None:
            saved_expert_names = checkpoint.get("model_config", {}).get("expert_names")
        if saved_expert_names is not None and list(saved_expert_names) != expert_names:
            raise RuntimeError(
                "Checkpoint专家名称与当前训练配置不一致。"
                f" checkpoint={list(saved_expert_names)}, config={expert_names}"
            )

        if mode == "resume":
            history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            resume_stage_name = str(checkpoint.get("stage", "")).strip() or None
            resume_stage_epoch = _checkpoint_stage_epoch(checkpoint)
            resume_stage_total = _checkpoint_stage_total(checkpoint)
            resume_optimizer_state = checkpoint.get("optimizer_state_dict")
            resume_scheduler_state = checkpoint.get("scheduler_state_dict")
            resume_scaler_state = checkpoint.get("scaler_state_dict")
            if resume_stage_name is None:
                raise RuntimeError("Resume checkpoint缺少stage，无法确定应从哪个训练阶段继续。")
            if resume_optimizer_state is None:
                raise RuntimeError(
                    "Resume checkpoint不包含optimizer_state_dict。"
                    "请使用完整训练断点，而不是 *_inference.pth。"
                )
        if is_main_process():
            print(f"Loaded Lite checkpoint: {checkpoint_path} mode={mode}")

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
    core = unwrap_model(model)

    def build_loaders(batch_size: int):
        return make_loaders(
            train_df, val_df, root, expert_to_index, (), config["image_target_height"], batch_size,
            int(config["num_workers"]), bool(config["persistent_workers"]),
            config.get("image_degradation_config"), distributed, rank, world_size,
        )

    train_loader, val_loader, train_sampler = build_loaders(int(config["batch_size"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    if mode == "resume" and resume_scaler_state:
        scaler.load_state_dict(resume_scaler_state)

    history_best_score, history_best_f1 = _history_best_values(history)
    if mode == "resume" and checkpoint is not None:
        best_score = float(checkpoint.get("best_score", history_best_score))
        best_f1 = float(checkpoint.get("best_f1", history_best_f1))
    else:
        best_score = float("-inf")
        best_f1 = float("-inf")
    epoch = start_epoch
    start_time = time.time()

    stage_items = list(config["stages"].items())
    resume_stage_index: Optional[int] = None
    if mode == "resume":
        stage_names = [name for name, _ in stage_items]
        if resume_stage_name not in stage_names:
            raise RuntimeError(
                f"Resume checkpoint中的stage={resume_stage_name!r}不在当前stages中: {stage_names}"
            )
        resume_stage_index = stage_names.index(resume_stage_name)

    trained_any_epoch = False
    for stage_index, (stage_name, stage) in enumerate(stage_items):
        stage_total_epochs = int(stage["epochs"])
        stage_start_epoch = 1
        restore_this_stage = False

        if mode == "resume" and resume_stage_index is not None:
            if stage_index < resume_stage_index:
                continue
            if stage_index == resume_stage_index:
                stage_start_epoch = resume_stage_epoch + 1
                if stage_start_epoch > stage_total_epochs:
                    if is_main_process():
                        print(
                            f"[Resume] stage={stage_name} 已完成 "
                            f"({resume_stage_epoch}/{stage_total_epochs})，跳过该阶段。"
                        )
                    continue
                restore_this_stage = True

        core.set_train_stage(stage_name)
        optimizer = make_optimizer(core, stage, config["optimizer"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, stage_total_epochs),
        )

        if restore_this_stage:
            try:
                optimizer.load_state_dict(resume_optimizer_state)
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "优化器状态恢复失败。请确认当前阶段的可训练参数分组与保存断点时一致。"
                ) from exc
            extending_completed_stage = (
                resume_stage_total > 0
                and resume_stage_epoch >= resume_stage_total
                and stage_total_epochs > resume_stage_epoch
            )
            if extending_completed_stage:
                # 已完成阶段上追加轮次时，保留AdamW动量，但重新启用该阶段学习率，
                # 并让新增轮次使用独立的余弦退火周期，避免从eta_min继续训练。
                for group in optimizer.param_groups:
                    group_name = str(group.get("name", "head"))
                    lr_key = f"lr_{group_name}"
                    if lr_key not in stage:
                        raise RuntimeError(f"阶段配置缺少{lr_key}，无法追加训练轮次。")
                    group["lr"] = float(stage[lr_key])
                    group["initial_lr"] = float(stage[lr_key])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, stage_total_epochs - resume_stage_epoch),
                )
                if is_main_process():
                    print(
                        f"[Resume] 检测到阶段追加训练: {resume_stage_epoch} -> {stage_total_epochs}，"
                        "保留优化器动量并重启新增轮次的学习率调度。"
                    )
            elif resume_scheduler_state:
                scheduler.load_state_dict(resume_scheduler_state)
                scheduler.T_max = max(1, stage_total_epochs)
            else:
                _restore_scheduler_without_state(scheduler, optimizer, resume_stage_epoch)
                if is_main_process():
                    print("[Resume] 旧Checkpoint无scheduler状态，已按阶段轮次对齐调度器。")

        if is_main_process():
            trainable = sum(parameter.numel() for parameter in core.parameters() if parameter.requires_grad)
            print(
                f"\n===== stage={stage_name} epochs={stage_total_epochs} "
                f"start={stage_start_epoch} trainable={trainable / 1e6:.3f}M ====="
            )

        for stage_epoch in range(stage_start_epoch, stage_total_epochs + 1):
            trained_any_epoch = True
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model, train_loader, device, stage["weights"], optimizer, scaler,
                epoch, stage_name, stage_epoch, stage_total_epochs, bool(stage["force_expert"]),
            )
            scheduler.step()
            val_metrics = {}
            if is_main_process():
                val_metrics = run_epoch(
                    core, val_loader, device, stage["weights"], None, scaler,
                    epoch, stage_name, stage_epoch, stage_total_epochs, False,
                )
                row = {
                    "epoch": epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch,
                    "stage_total_epochs": stage_total_epochs,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
                for group in optimizer.param_groups:
                    row[f"lr_{group.get('name', 'group')}"] = float(group["lr"])
                history.append(row)
                save_history(train_output_dir, history)
                save_json(train_output_dir / "epoch_diagnostics" / f"epoch_{epoch:03d}_{stage_name}.json", row)
                save_training_curves(train_output_dir, history, expert_names)
                save_expert_epoch(train_output_dir, epoch, stage_name, expert_names, train_metrics, val_metrics)
                save_expert_diagnostics_csv(
                    train_output_dir, epoch, stage_name, expert_names, train_metrics, val_metrics,
                )
                save_attention_sample(
                    core, val_df, root, expert_to_index, config["image_target_height"], device,
                    train_output_dir, epoch, stage_name, int(config["validation_attention_sample_index"]),
                )

                score = selection_score(val_metrics)
                current_f1 = float(val_metrics.get("f1", 0.0))
                is_best_score = score > best_score
                is_best_f1 = current_f1 > best_f1
                best_score = max(best_score, score)
                best_f1 = max(best_f1, current_f1)

                metadata = {
                    "version": "fire_arbiter_moe_lite_train",
                    "train_csv": str(train_csv),
                    "val_csv": str(val_csv),
                    "stage_config": stage,
                    "validation_best_threshold": val_metrics.get("best_threshold", core.get_threshold()),
                    "selection_score": score,
                    "expert_names": list(expert_names),
                    "num_experts": len(expert_names),
                }
                save_checkpoint(
                    core, optimizer, scheduler, scaler,
                    model_output_dir / "latest.pth",
                    epoch, stage_name, stage_epoch, stage_total_epochs,
                    history, metadata, best_score, best_f1,
                )
                if is_best_score:
                    save_checkpoint(
                        core, optimizer, scheduler, scaler,
                        model_output_dir / "best.pth",
                        epoch, stage_name, stage_epoch, stage_total_epochs,
                        history, metadata, best_score, best_f1,
                    )
                    save_json(train_output_dir / "best_metrics.json", row)
                if is_best_f1:
                    save_checkpoint(
                        core, optimizer, scheduler, scaler,
                        model_output_dir / "best_f1.pth",
                        epoch, stage_name, stage_epoch, stage_total_epochs,
                        history, metadata, best_score, best_f1,
                    )

                print(
                    f"epoch={epoch} ({stage_epoch}/{stage_total_epochs}) stage={stage_name} "
                    f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
                    f"train_acc={train_metrics.get('acc', float('nan')):.4f} "
                    f"train_f1={train_metrics.get('f1', float('nan')):.4f} "
                    f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
                    f"val_acc={val_metrics.get('acc', float('nan')):.4f} "
                    f"val_f1={val_metrics.get('f1', float('nan')):.4f} "
                    f"val_balanced_acc={val_metrics.get('balanced_acc', float('nan')):.4f} "
                    f"val_best_t={val_metrics.get('best_threshold', float('nan')):.3f} "
                    f"best={best_score:.4f}"
                )
            if distributed:
                dist.barrier()
            epoch += 1

    if is_main_process():
        if mode == "resume" and not trained_any_epoch:
            print(
                "[Resume] 当前配置中没有未完成轮次。若要继续当前阶段，请增大该阶段的epochs；"
                "若要重新执行所有阶段，请使用finetune模式。"
            )
        print(f"训练完成，用时 {(time.time() - start_time) / 60.0:.1f} min")
        print("latest:", model_output_dir / "latest.pth")
        print("latest inference:", inference_checkpoint_path(model_output_dir / "latest.pth"))
        print("best:", model_output_dir / "best.pth")
        print("best inference:", inference_checkpoint_path(model_output_dir / "best.pth"))
        print("best_f1:", model_output_dir / "best_f1.pth")
        print("best_f1 inference:", inference_checkpoint_path(model_output_dir / "best_f1.pth"))
        print("训练日志:", train_output_dir / "history.csv")
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    config = build_train_config()
    mode = str(config["run"]["mode"]).strip().lower()
    if mode == "resume":
        running_train("resume", config["run"]["resume_path"], config)
    elif mode == "finetune":
        running_train("finetune", config["run"]["finetune_path"], config)
    else:
        running_train("start", None, config)


if __name__ == "__main__":
    main()
