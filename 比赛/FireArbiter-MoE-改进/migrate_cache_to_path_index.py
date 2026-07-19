# -*- coding: utf-8 -*-
"""将旧缓存迁移为只按原图相对路径判断的新缓存范式。

不会修改 train/val 划分、样本顺序、标签、专家或 sample_id。
迁移内容：
1. mask 缓存改为 relative_original_path 唯一索引；
2. degraded 缓存改为 (relative_original_path, degradation_stage) 唯一索引；
3. 更新 train_all.csv / val CSV 中的 mask 缓存字段；
4. 删除训练不再使用的整图 image_npz_cache；
5. 被成功迁移的旧文件会移动为路径哈希文件名，避免保留重复副本。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


MASK_INDEX_NAME = "mask_cache_index.csv"
DEGRADED_INDEX_NAME = "degraded_image_cache_index.csv"
TRAIN_CSV_NAME = "train_all.csv"
VAL_CSV_NAME = "val_all_datasets_real_only.csv"


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



def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    try:
        return bool(int(float(text)))
    except Exception:
        return False

def normalize_relative_path(value: str) -> str:
    text = clean_text(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def stable_hash(text: str) -> str:
    return hashlib.sha1(normalize_relative_path(text).encode("utf-8")).hexdigest()


def path_hash(relative_path: str) -> str:
    return stable_hash(relative_path)[:24]


def to_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def resolve_from_row(row: Dict, root: Path, relative_column: str, absolute_column: str = "") -> Optional[Path]:
    relative = normalize_relative_path(row.get(relative_column, ""))
    if relative:
        return root / relative
    if absolute_column:
        text = clean_text(row.get(absolute_column, ""))
        if text:
            path = Path(text)
            return path if path.is_absolute() else root / text
    return None


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        df.to_csv(temp, index=False, encoding="utf-8-sig")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def backup_metadata(paths: Sequence[Path], split_dir: Path, dry_run: bool) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing or dry_run:
        return None
    backup_dir = split_dir / "cache_migration_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def move_cache_file(source: Path, target: Path, dry_run: bool) -> Tuple[bool, str]:
    """返回 (可用, 状态)。可用表示迁移后目标缓存存在。"""
    if target.exists():
        if source.exists() and source != target:
            if not dry_run:
                source.unlink(missing_ok=True)
            return True, "target_exists_removed_duplicate"
        return True, "target_exists"
    if not source.exists():
        return False, "source_missing"
    if dry_run:
        return True, "would_move"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return True, "moved"


def inspect_mask_metadata(path: Path, row: Dict) -> Dict:
    result = {
        "has_positive_mask": safe_bool(row.get("has_positive_mask", False)),
        "has_negative_mask": safe_bool(row.get("has_negative_mask", False)),
        "annotation_warning": clean_text(row.get("annotation_warning", "")),
        "original_width": -1,
        "original_height": -1,
    }
    if not path.exists():
        return result
    try:
        with np.load(path) as data:
            pos = np.asarray(data.get("positive_mask", np.zeros((0, 0), dtype=np.uint8)))
            neg = np.asarray(data.get("negative_mask", np.zeros((0, 0), dtype=np.uint8)))
            result["has_positive_mask"] = bool(pos.size > 0 and np.any(pos > 0))
            result["has_negative_mask"] = bool(neg.size > 0 and np.any(neg > 0))
            result["original_width"] = int(data.get("original_width", pos.shape[1] if pos.ndim == 2 else -1))
            result["original_height"] = int(data.get("original_height", pos.shape[0] if pos.ndim == 2 else -1))
            warning = data.get("annotation_warning", result["annotation_warning"])
            if isinstance(warning, np.ndarray) and warning.ndim == 0:
                warning = warning.item()
            result["annotation_warning"] = clean_text(warning)
    except Exception:
        pass
    return result


def load_split_csvs(split_dir: Path) -> Tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    train_path = split_dir / TRAIN_CSV_NAME
    val_path = split_dir / VAL_CSV_NAME
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"缺少划分 CSV: {train_path} / {val_path}")
    return train_path, val_path, pd.read_csv(train_path), pd.read_csv(val_path)


def migrate_masks(root: Path, split_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, dry_run: bool) -> Tuple[pd.DataFrame, Dict[str, int]]:
    mask_dir = split_dir / "mask_npz_cache"
    mask_dir.mkdir(parents=True, exist_ok=True)
    records_df = pd.concat([train_df, val_df], ignore_index=True)
    if "relative_path" not in records_df.columns:
        raise RuntimeError("train/val CSV 缺少 relative_path")
    records_df["_path_key"] = records_df["relative_path"].map(normalize_relative_path)
    records_df = records_df[records_df["_path_key"] != ""].drop_duplicates("_path_key", keep="first")

    rows: List[Dict] = []
    stats = {"moved": 0, "existing": 0, "missing": 0, "duplicates_removed": 0}
    for record in tqdm(records_df.to_dict("records"), desc="迁移 mask 缓存", unit="img"):
        relative_path = record["_path_key"]
        old_path = resolve_from_row(record, root, "relative_mask_npz_path", "mask_npz_path")
        if old_path is None:
            stats["missing"] += 1
            continue
        target = mask_dir / f"{path_hash(relative_path)}.npz"
        usable, status = move_cache_file(old_path, target, dry_run)
        if not usable:
            stats["missing"] += 1
            continue
        if status in {"moved", "would_move"}:
            stats["moved"] += 1
        elif status == "target_exists_removed_duplicate":
            stats["duplicates_removed"] += 1
        else:
            stats["existing"] += 1
        metadata_path = old_path if dry_run and old_path.exists() else target
        metadata = inspect_mask_metadata(metadata_path, record)
        label = int(float(record.get("label", 0))) if clean_text(record.get("label", "")) else 0
        # 训练端对正 mask 仍按图片 label 决定是否有效。
        has_positive = bool(metadata["has_positive_mask"] and label == 1)
        rows.append({
            "cache_key": path_hash(relative_path),
            "relative_original_path": relative_path,
            "relative_mask_npz_path": to_relative(target, root),
            "has_positive_mask": has_positive,
            "has_negative_mask": bool(metadata["has_negative_mask"]),
            "annotation_warning": metadata["annotation_warning"],
            "original_width": int(metadata["original_width"]),
            "original_height": int(metadata["original_height"]),
        })
    index_df = pd.DataFrame(rows, columns=[
        "cache_key", "relative_original_path", "relative_mask_npz_path",
        "has_positive_mask", "has_negative_mask", "annotation_warning",
        "original_width", "original_height",
    ])
    return index_df.drop_duplicates("relative_original_path", keep="first"), stats


def update_split_mask_columns(df: pd.DataFrame, mask_index: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_columns = [
        "relative_image_npz_path", "image_npz_path",
        "relative_mask_npz_path", "mask_npz_path",
        "has_positive_mask", "has_negative_mask", "annotation_warning", "mask_cache_key",
    ]
    out.drop(columns=[column for column in drop_columns if column in out.columns], inplace=True)
    out["_cache_path_key"] = out["relative_path"].map(normalize_relative_path)
    view = mask_index.copy()
    view["_cache_path_key"] = view["relative_original_path"].map(normalize_relative_path)
    view = view.drop_duplicates("_cache_path_key", keep="first")
    view.rename(columns={"cache_key": "mask_cache_key"}, inplace=True)
    view.drop(columns=["relative_original_path", "original_width", "original_height"], inplace=True, errors="ignore")
    out = out.merge(view, on="_cache_path_key", how="left", sort=False, validate="m:1")
    out.drop(columns=["_cache_path_key"], inplace=True)
    out["mask_cache_key"] = out["mask_cache_key"].fillna("")
    out["relative_mask_npz_path"] = out["relative_mask_npz_path"].fillna("")
    out["has_positive_mask"] = out["has_positive_mask"].fillna(False).astype(bool)
    out["has_negative_mask"] = out["has_negative_mask"].fillna(False).astype(bool)
    out["annotation_warning"] = out["annotation_warning"].fillna("")
    return out


def migrate_degraded(root: Path, split_dir: Path, dry_run: bool) -> Tuple[pd.DataFrame, Dict[str, int]]:
    index_path = split_dir / DEGRADED_INDEX_NAME
    columns = [
        "cache_key", "relative_original_path", "relative_degraded_path",
        "label", "degradation_stage", "sample_id", "expert_name", "source_dataset",
    ]
    if not index_path.exists():
        return pd.DataFrame(columns=columns), {"moved": 0, "existing": 0, "missing": 0, "duplicates_removed": 0}
    old = pd.read_csv(index_path)
    for column in ["relative_original_path", "relative_degraded_path", "label", "degradation_stage"]:
        if column not in old.columns:
            old[column] = ""
    old["relative_original_path"] = old["relative_original_path"].map(normalize_relative_path)
    old["degradation_stage"] = pd.to_numeric(old["degradation_stage"], errors="coerce").fillna(-1).astype(int)

    rows: List[Dict] = []
    seen = set()
    stats = {"moved": 0, "existing": 0, "missing": 0, "duplicates_removed": 0}
    for record in tqdm(old.to_dict("records"), desc="迁移 degraded 缓存", unit="variant"):
        relative_path = normalize_relative_path(record.get("relative_original_path", ""))
        stage = int(record.get("degradation_stage", -1))
        if not relative_path or stage < 0:
            stats["missing"] += 1
            continue
        identity = (relative_path, stage)
        old_path = resolve_from_row(record, root, "relative_degraded_path", "degraded_path")
        label = int(float(record.get("label", 0))) if clean_text(record.get("label", "")) else 0
        subdir = "fire" if label == 1 else "no_fire"
        target = split_dir / "degraded_image_cache" / subdir / f"{path_hash(relative_path)}_stage{stage:02d}.jpg"

        if identity in seen:
            if old_path is not None and old_path.exists() and old_path != target and not dry_run:
                old_path.unlink(missing_ok=True)
            stats["duplicates_removed"] += 1
            continue
        seen.add(identity)

        if old_path is None:
            stats["missing"] += 1
            continue
        usable, status = move_cache_file(old_path, target, dry_run)
        if not usable:
            stats["missing"] += 1
            continue
        if status in {"moved", "would_move"}:
            stats["moved"] += 1
        elif status == "target_exists_removed_duplicate":
            stats["duplicates_removed"] += 1
        else:
            stats["existing"] += 1
        rows.append({
            "cache_key": f"{path_hash(relative_path)}:stage={stage:02d}",
            "relative_original_path": relative_path,
            "relative_degraded_path": to_relative(target, root),
            "label": label,
            "degradation_stage": stage,
            "sample_id": clean_text(record.get("sample_id", "")),
            "expert_name": clean_text(record.get("expert_name", "")),
            "source_dataset": clean_text(record.get("source_dataset", "")),
        })
    return pd.DataFrame(rows, columns=columns), stats


def remove_image_npz_cache(split_dir: Path, dry_run: bool) -> Tuple[int, int]:
    directory = split_dir / "image_npz_cache"
    if not directory.exists():
        return 0, 0
    files = [path for path in directory.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    if not dry_run:
        shutil.rmtree(directory)
    return len(files), total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="迁移旧缓存到路径键缓存范式，不改变任何数据划分。")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FIRE_ROOT", "..")), help="项目根目录")
    parser.add_argument("--dry-run", action="store_true", help="只显示将执行的操作，不写入或删除")
    parser.add_argument("--keep-image-npz", action="store_true", help="保留旧 image_npz_cache（默认删除）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    split_dir = root / "fire_splits"
    train_path, val_path, train_df, val_df = load_split_csvs(split_dir)
    degraded_path = split_dir / DEGRADED_INDEX_NAME
    mask_index_path = split_dir / MASK_INDEX_NAME

    backup_dir = backup_metadata([train_path, val_path, degraded_path, mask_index_path], split_dir, args.dry_run)
    if backup_dir is not None:
        print("元数据备份:", backup_dir)

    mask_index, mask_stats = migrate_masks(root, split_dir, train_df, val_df, args.dry_run)
    new_train = update_split_mask_columns(train_df, mask_index)
    new_val = update_split_mask_columns(val_df, mask_index)
    degraded_index, degraded_stats = migrate_degraded(root, split_dir, args.dry_run)

    if not args.dry_run:
        atomic_write_csv(mask_index, mask_index_path)
        atomic_write_csv(degraded_index, degraded_path)
        atomic_write_csv(new_train, train_path)
        atomic_write_csv(new_val, val_path)

    removed_files = removed_bytes = 0
    if not args.keep_image_npz:
        removed_files, removed_bytes = remove_image_npz_cache(split_dir, args.dry_run)

    print("\n迁移完成" if not args.dry_run else "\nDry-run 完成，未修改文件")
    print("ROOT:", root)
    print("train 行数保持:", len(train_df), "->", len(new_train))
    print("val 行数保持:", len(val_df), "->", len(new_val))
    print("mask 索引记录:", len(mask_index), mask_stats)
    print("degraded 索引记录:", len(degraded_index), degraded_stats)
    if not args.keep_image_npz:
        action = "将删除" if args.dry_run else "已删除"
        print(f"旧 image_npz_cache {action}: files={removed_files}, size={removed_bytes / (1024 ** 3):.2f} GiB")
    print("下一步：运行 全量划分_路径缓存.ipynb。索引中已有的路径不会重复生成。")


if __name__ == "__main__":
    main()
