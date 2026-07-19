# -*- coding: utf-8 -*-
"""按原图片路径定向删除 mask/degraded 缓存记录和缓存文件。

不会删除原图片，不会删除或重新划分 train/val 样本。删除索引记录后，下一次运行
全量划分_路径缓存.ipynb 时会重新生成这些路径对应的缓存。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd
from tqdm.auto import tqdm


MASK_INDEX_NAME = "mask_cache_index.csv"
DEGRADED_INDEX_NAME = "degraded_image_cache_index.csv"
SPLIT_CSV_NAMES = ("train_all.csv", "val_all_datasets_real_only.csv")


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def normalize_relative_path(value: str) -> str:
    text = clean_text(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def path_hash(relative_path: str) -> str:
    return hashlib.sha1(normalize_relative_path(relative_path).encode("utf-8")).hexdigest()[:24]


def to_relative_input(value: str, root: Path) -> str:
    text = clean_text(value)
    path = Path(text).expanduser()
    if path.is_absolute():
        try:
            return normalize_relative_path(str(path.resolve().relative_to(root.resolve())))
        except Exception as exc:
            raise ValueError(f"路径不在项目 ROOT 下，拒绝清理: {path}") from exc
    return normalize_relative_path(text)


def target_matches(relative_path: str, targets: Sequence[str]) -> bool:
    value = normalize_relative_path(relative_path)
    return any(value == target or value.startswith(target.rstrip("/") + "/") for target in targets)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        df.to_csv(temp, index=False, encoding="utf-8-sig")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def backup_metadata(paths: Sequence[Path], split_dir: Path) -> Path:
    backup_dir = split_dir / "cache_cleanup_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def collect_targets(args: argparse.Namespace, root: Path) -> List[str]:
    values = list(args.paths)
    if args.paths_file:
        with open(args.paths_file, "r", encoding="utf-8") as handle:
            values.extend(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))
    targets = sorted({to_relative_input(value, root) for value in values if clean_text(value)})
    if not targets:
        raise ValueError("没有提供任何待清理的原图片或文件夹路径")
    return targets


def paths_from_index(index_path: Path, targets: Sequence[str], root: Path, path_column: str) -> tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    if not index_path.exists():
        return pd.DataFrame(), pd.DataFrame(), []
    df = pd.read_csv(index_path)
    if "relative_original_path" not in df.columns:
        raise RuntimeError(f"检测到旧索引格式，请先运行迁移脚本: {index_path}")
    selected = df["relative_original_path"].map(lambda value: target_matches(value, targets))
    removed = df[selected].copy()
    kept = df[~selected].copy()
    files = []
    if path_column in removed.columns:
        for value in removed[path_column]:
            rel = normalize_relative_path(value)
            if rel:
                files.append(root / rel)
    return kept, removed, files


def clear_split_cache_fields(df: pd.DataFrame, targets: Sequence[str]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    selected = out["relative_path"].map(lambda value: target_matches(value, targets))
    count = int(selected.sum())
    string_columns = {"mask_cache_key", "relative_mask_npz_path", "annotation_warning"}
    for column, value in (
        ("mask_cache_key", ""),
        ("relative_mask_npz_path", ""),
        ("has_positive_mask", False),
        ("has_negative_mask", False),
        ("annotation_warning", ""),
    ):
        if column not in out.columns:
            out[column] = value
        if column in string_columns:
            out[column] = out[column].fillna("").astype(str)
        else:
            out[column] = out[column].fillna(False).astype(bool)
        out.loc[selected, column] = value
    return out, count


def deterministic_files(split_dir: Path, targets: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    mask_dir = split_dir / "mask_npz_cache"
    degraded_dir = split_dir / "degraded_image_cache"
    for target in targets:
        # 对文件路径可直接补充确定性文件名；文件夹仍主要依赖索引行枚举。
        digest = path_hash(target)
        files.append(mask_dir / f"{digest}.npz")
        for subdir in ("fire", "no_fire"):
            files.extend((degraded_dir / subdir).glob(f"{digest}_stage*.jpg"))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按原图片/文件夹路径删除新范式缓存。")
    parser.add_argument("paths", nargs="*", help="ROOT 下的原图片或文件夹路径，可传绝对路径或相对路径")
    parser.add_argument("--paths-file", type=Path, help="每行一个原图片或文件夹路径")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FIRE_ROOT", "..")), help="项目根目录")
    parser.add_argument("--apply", action="store_true", help="实际删除；不传时只做 dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    split_dir = root / "fire_splits"
    targets = collect_targets(args, root)
    print("匹配目标:")
    for target in targets:
        print(" -", target)

    mask_path = split_dir / MASK_INDEX_NAME
    degraded_path = split_dir / DEGRADED_INDEX_NAME
    mask_kept, mask_removed, mask_files = paths_from_index(mask_path, targets, root, "relative_mask_npz_path")
    deg_kept, deg_removed, deg_files = paths_from_index(degraded_path, targets, root, "relative_degraded_path")

    split_updates = []
    touched_split_rows = 0
    for name in SPLIT_CSV_NAMES:
        path = split_dir / name
        if path.exists():
            original = pd.read_csv(path)
            updated, touched = clear_split_cache_fields(original, targets)
            split_updates.append((path, updated))
            touched_split_rows += touched

    unique_files = []
    seen: Set[str] = set()
    for path in mask_files + deg_files + deterministic_files(split_dir, targets):
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_files.append(path)

    existing_files = [path for path in unique_files if path.exists() and path.is_file()]
    total_bytes = sum(path.stat().st_size for path in existing_files)
    print(f"\n将移除 mask 索引记录: {len(mask_removed)}")
    print(f"将移除 degraded 索引记录: {len(deg_removed)}")
    print(f"将清空 train/val 缓存字段的样本行: {touched_split_rows}")
    print(f"将删除缓存文件: {len(existing_files)}, {total_bytes / (1024 ** 2):.1f} MiB")

    if not args.apply:
        print("\n当前是 dry-run。确认后追加 --apply 执行实际清理。")
        return

    metadata_paths = [mask_path, degraded_path] + [path for path, _ in split_updates]
    backup_dir = backup_metadata(metadata_paths, split_dir)
    print("元数据备份:", backup_dir)

    for path in tqdm(existing_files, desc="删除缓存文件", unit="file"):
        path.unlink(missing_ok=True)
    if mask_path.exists():
        atomic_write_csv(mask_kept, mask_path)
    if degraded_path.exists():
        atomic_write_csv(deg_kept, degraded_path)
    for path, updated in split_updates:
        atomic_write_csv(updated, path)

    print("\n清理完成。下次运行 全量划分_路径缓存.ipynb 时，这些路径会重新生成缓存。")


if __name__ == "__main__":
    main()
