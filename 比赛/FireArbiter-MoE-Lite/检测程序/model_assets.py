# -*- coding: utf-8 -*-
"""Local-first model asset resolution for the GUI and model constructors.

The important rule in this module is intentionally strict:
- if a configured local folder/file exists, it is used without any network access;
- network download is attempted only when that local folder/file does not exist.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple
from urllib.parse import urlparse

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OWL_DIRNAME = "OWL"
DEFAULT_CONVNEXT_DIRNAME = "ConvNext"
DEFAULT_FIRE_LITE_FILENAME = "fire-lite.pth"
DEFAULT_OWL_REPO_ID = "google/owlvit-base-patch32"


def models_dir_path(models_dir: str | Path | None = None) -> Path:
    return Path(models_dir).expanduser().resolve() if models_dir else DEFAULT_MODELS_DIR.resolve()


def owl_dir_path(models_dir: str | Path | None = None) -> Path:
    return models_dir_path(models_dir) / DEFAULT_OWL_DIRNAME


def convnext_dir_path(models_dir: str | Path | None = None) -> Path:
    return models_dir_path(models_dir) / DEFAULT_CONVNEXT_DIRNAME


def fire_lite_path(models_dir: str | Path | None = None) -> Path:
    return models_dir_path(models_dir) / DEFAULT_FIRE_LITE_FILENAME


@contextmanager
def forced_huggingface_offline() -> Iterator[None]:
    """Temporarily force Hugging Face/Transformers into offline mode."""
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _atomic_directory_target(target: Path) -> Tuple[Path, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.download-", dir=str(target.parent)))
    return temp_root, temp_root / target.name


def _commit_directory(downloaded_dir: Path, target: Path, temp_root: Path) -> None:
    if target.exists():
        # Another process may have completed the same download. Existing local data wins.
        shutil.rmtree(temp_root, ignore_errors=True)
        return
    downloaded_dir.replace(target)
    shutil.rmtree(temp_root, ignore_errors=True)


def resolve_owl_source(
    model_name: Optional[str],
    models_dir: str | Path | None = None,
    local_first: bool = True,
) -> Tuple[Optional[str], bool]:
    """Return (source, local_only).

    With local_first=True, an existing models/OWL directory is always used in
    strict offline mode. It is never validated against the internet.
    """
    if not model_name:
        return None, True
    if not local_first:
        return str(model_name), False

    target = owl_dir_path(models_dir)
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"OWL 本地路径存在但不是文件夹：{target}")
        return str(target), True

    repo_id = str(model_name or DEFAULT_OWL_REPO_ID)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "models/OWL 不存在，需要下载 OWL 权重，但当前环境缺少 huggingface_hub。"
        ) from exc

    temp_root, download_dir = _atomic_directory_target(target)
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(download_dir))
        _commit_directory(download_dir, target, temp_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return str(target), True


def find_convnext_weight_file(directory: Path) -> Path:
    preferred_names = (
        "convnext_tiny-983f1562.pth",
        "convnext_tiny.pth",
        "pytorch_model.bin",
    )
    for name in preferred_names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path for pattern in ("*.pth", "*.pt", "*.bin")
        for path in directory.rglob(pattern)
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"ConvNext 文件夹已存在，因此不会联网；但未在其中找到 .pth/.pt/.bin 权重：{directory}"
        )
    return candidates[0]


def resolve_convnext_weight_file(
    weights_url: str,
    models_dir: str | Path | None = None,
    local_first: bool = True,
) -> Optional[Path]:
    """Resolve the ConvNeXt checkpoint under models/ConvNext.

    When the directory already exists, no URL access is performed, even when
    the folder is incomplete; an explicit local-file error is raised instead.
    """
    if not local_first:
        return None

    target = convnext_dir_path(models_dir)
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"ConvNext 本地路径存在但不是文件夹：{target}")
        return find_convnext_weight_file(target)

    temp_root, download_dir = _atomic_directory_target(target)
    download_dir.mkdir(parents=True, exist_ok=False)
    filename = Path(urlparse(weights_url).path).name or "convnext_tiny.pth"
    output = download_dir / filename
    try:
        # Direct download only occurs because models/ConvNext did not exist.
        torch.hub.download_url_to_file(weights_url, str(output), progress=True)
        _commit_directory(download_dir, target, temp_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return find_convnext_weight_file(target)


def load_torch_state_dict(path: Path) -> dict:
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


def resolve_fire_lite_checkpoint(
    selected_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    local_first: bool = True,
    download_url: str = "",
) -> Path:
    """Resolve the main FireArbiter checkpoint.

    A manually selected existing checkpoint remains usable. Otherwise the
    canonical models/fire-lite.pth file is preferred. If it is absent, it can
    be downloaded when a URL has been configured in the GUI or environment.
    """
    selected = Path(selected_path).expanduser() if selected_path else None
    canonical = fire_lite_path(models_dir)

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
    fd, temp_name = tempfile.mkstemp(prefix=".fire-lite.download-", suffix=".pth", dir=str(canonical.parent))
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
