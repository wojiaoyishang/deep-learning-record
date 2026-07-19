# -*- coding: utf-8 -*-
"""GUI-only model file discovery, download, and strict local loading.

This module deliberately stays outside ``fire_model.py`` so the model remains a
standalone training/inference file with its original public constructor.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import urlparse

import torch
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OWL_MODELS_DIR = MODELS_DIR / "OWL"
CONVNEXT_MODELS_DIR = MODELS_DIR / "ConvNext"
DEFAULT_FIRE_LITE_PATH = MODELS_DIR / "fire-lite.pth"
DEFAULT_OWL_REPO_ID = "google/owlvit-base-patch32"


def _atomic_directory_target(target: Path) -> Tuple[Path, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.download-", dir=str(target.parent)))
    return temp_root, temp_root / target.name


def _commit_directory(downloaded_dir: Path, target: Path, temp_root: Path) -> None:
    if target.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
        return
    downloaded_dir.replace(target)
    shutil.rmtree(temp_root, ignore_errors=True)


def resolve_gui_owl_source(
    model_name: Optional[str],
    models_dir: str | Path = MODELS_DIR,
    local_first: bool = True,
) -> Tuple[Optional[str], bool]:
    """Resolve OWL for the GUI; an existing local folder never triggers network access."""
    if not model_name:
        return None, bool(local_first)
    if not local_first:
        return str(model_name), False

    target = Path(models_dir) / "OWL"
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"OWL 本地路径存在但不是文件夹：{target}")
        return str(target.resolve()), True

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "models/OWL 不存在，需要下载 OWL 权重，但当前环境缺少 huggingface_hub。"
        ) from exc

    repo_id = str(model_name or DEFAULT_OWL_REPO_ID)
    temp_root, download_dir = _atomic_directory_target(target)
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(download_dir))
        _commit_directory(download_dir, target, temp_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return str(target.resolve()), True


def find_gui_convnext_weight(directory: Path) -> Path:
    preferred_names = (
        "convnext_tiny-983f1562.pth",
        "convnext_tiny.pth",
        "pytorch_model.bin",
    )
    for name in preferred_names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(
        path
        for pattern in ("*.pth", "*.pt", "*.bin")
        for path in directory.rglob(pattern)
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"ConvNext 文件夹已存在，因此不会联网；但未找到 .pth/.pt/.bin 权重：{directory}"
        )
    return candidates[0].resolve()


def resolve_gui_convnext_weight(
    models_dir: str | Path = MODELS_DIR,
    local_first: bool = True,
) -> Optional[Path]:
    """Resolve ConvNeXt for the GUI; an existing local folder short-circuits network access."""
    if not local_first:
        return None

    target = Path(models_dir) / "ConvNext"
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"ConvNext 本地路径存在但不是文件夹：{target}")
        return find_gui_convnext_weight(target)

    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name or "convnext_tiny.pth"
    temp_root, download_dir = _atomic_directory_target(target)
    download_dir.mkdir(parents=True, exist_ok=False)
    output = download_dir / filename
    try:
        torch.hub.download_url_to_file(weights.url, str(output), progress=True)
        _commit_directory(download_dir, target, temp_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return find_gui_convnext_weight(target)


def _load_gui_torch_state_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"权重文件不是可用的 state_dict：{path}")
    if payload and all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


@contextmanager
def gui_local_model_loading(
    convnext_weight: Optional[Path],
    force_huggingface_offline: bool,
) -> Iterator[None]:
    """Load local GUI assets without adding GUI parameters to the model constructor."""
    env_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous_env = {key: os.environ.get(key) for key in env_keys}

    import torchvision.models._api as torchvision_models_api

    original_tv_loader = torchvision_models_api.load_state_dict_from_url
    original_torch_loader = torch.hub.load_state_dict_from_url

    def local_convnext_loader(url: str, *args, **kwargs):
        if convnext_weight is None:
            return original_tv_loader(url, *args, **kwargs)
        return _load_gui_torch_state_dict(Path(convnext_weight))

    try:
        if force_huggingface_offline:
            for key in env_keys:
                os.environ[key] = "1"
        if convnext_weight is not None:
            torchvision_models_api.load_state_dict_from_url = local_convnext_loader
            torch.hub.load_state_dict_from_url = local_convnext_loader
        yield
    finally:
        torchvision_models_api.load_state_dict_from_url = original_tv_loader
        torch.hub.load_state_dict_from_url = original_torch_loader
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_fire_lite_checkpoint(
    selected_path: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    local_first: bool = True,
    download_url: str = "",
) -> Path:
    """Resolve the main GUI checkpoint, downloading only when no local file exists."""
    selected = Path(selected_path).expanduser() if selected_path else None
    canonical = Path(models_dir) / "fire-lite.pth"

    if selected is not None and selected.is_file() and selected.resolve() != canonical.resolve():
        return selected.resolve()
    if canonical.is_file():
        return canonical.resolve()
    if selected is not None and selected.is_file():
        return selected.resolve()
    if not local_first and selected is not None:
        return selected.resolve()

    url = str(download_url or os.environ.get("FIRE_LITE_MODEL_URL", "")).strip()
    if not url:
        raise FileNotFoundError(
            f"未找到主模型权重：{canonical}\n"
            "请把 fire-lite.pth 放入 models 文件夹，或在“模型资源设置”中填写下载地址。"
        )

    canonical.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".fire-lite.download-", suffix=".pth", dir=str(canonical.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        urllib.request.urlretrieve(url, temp_path)
        if canonical.exists():
            temp_path.unlink(missing_ok=True)
        else:
            temp_path.replace(canonical)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return canonical.resolve()
