# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import *
from .engine import FireInferenceEngine
from .logging_utils import logger

class WorkerSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(object, object, object)
    error = QtCore.pyqtSignal(str)


class BatchSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str, object, object, object)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)


class InferenceRunnable(QtCore.QRunnable):
    def __init__(
        self,
        engine: FireInferenceEngine,
        rgb: np.ndarray,
        model_settings: List[Dict[str, Any]],
        meta: Dict[str, Any],
        valid_mask_np: np.ndarray | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.rgb = np.ascontiguousarray(rgb.copy())
        self.model_settings = [dict(x) for x in model_settings]
        self.meta = dict(meta)
        self.valid_mask_np = None if valid_mask_np is None else np.ascontiguousarray(valid_mask_np.astype(np.float32))
        self.signals = WorkerSignals()

    def run(self):
        try:
            result, maps = self.engine.predict_rgb(
                self.rgb,
                self.model_settings,
                self.meta,
                self.valid_mask_np,
            )
            self.signals.result.emit(result, maps, self.rgb)
        except Exception as e:
            logger.exception(
                "模型预测失败 | source_type=%s | source_path=%s | frame_index=%s",
                self.meta.get("source_type", "unknown"),
                self.meta.get("source_path", ""),
                self.meta.get("frame_index", 0),
            )
            self.signals.error.emit(f"{type(e).__name__}: {e}")


class ModelLoadThread(QtCore.QThread):
    """Resolve/download and initialize the model outside the GUI thread."""

    loaded = QtCore.pyqtSignal(object, str, str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        selected_path: str | None,
        device: torch.device,
        local_models_first: bool,
        models_dir: str | Path,
        download_url: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.selected_path = selected_path
        self.device = device
        self.local_models_first = bool(local_models_first)
        self.models_dir = Path(models_dir)
        self.download_url = str(download_url or "")

    def run(self):
        try:
            logger.info(
                "开始加载模型 | selected_path=%s | device=%s | local_first=%s",
                self.selected_path or "<default>",
                self.device,
                self.local_models_first,
            )
            resolved_path = resolve_fire_lite_checkpoint(
                selected_path=self.selected_path or None,
                models_dir=self.models_dir,
                local_first=self.local_models_first,
                download_url=self.download_url,
            )
            engine = FireInferenceEngine(
                str(resolved_path),
                self.device,
                local_models_first=self.local_models_first,
                models_dir=self.models_dir,
            )
            logger.info("模型加载完成 | path=%s | device=%s", resolved_path, self.device)
            self.loaded.emit(engine, str(resolved_path), str(self.device))
        except Exception as exc:
            logger.exception(
                "模型加载失败 | selected_path=%s | device=%s",
                self.selected_path or "<default>",
                self.device,
            )
            self.error.emit(f"{type(exc).__name__}: {exc}")


class BatchInferenceRunnable(QtCore.QRunnable):
    def __init__(
        self,
        engine: FireInferenceEngine,
        image_paths: List[str],
        model_settings: List[Dict[str, Any]],
        root_dir: str,
        save_attention: bool = True,
        resize_settings: Dict[str, Any] | None = None,
        tmp_dir: str | Path | None = None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.engine = engine
        self.image_paths = [str(p) for p in image_paths]
        self.model_settings = [dict(x) for x in model_settings]
        self.root_dir = str(root_dir)
        self.save_attention = bool(save_attention)
        self.resize_settings = dict(resize_settings or {})
        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else BATCH_TMP_ROOT / time.strftime("batch_%Y%m%d_%H%M%S")
        self.batch_size = max(1, int(batch_size or 1))
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self.signals = BatchSignals()

    def stop(self):
        self._stop_requested = True
        self._pause_event.set()

    def pause(self):
        if not self._stop_requested:
            self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def _wait_if_paused(self) -> bool:
        while not self._stop_requested and not self._pause_event.wait(0.1):
            pass
        return not self._stop_requested

    @staticmethod
    def _json_key(path: str, root_dir: str) -> str:
        p = Path(path)
        try:
            return p.relative_to(Path(root_dir)).as_posix()
        except Exception:
            return p.name

    @staticmethod
    def _binary_export_label(result: Dict[str, Any]) -> int:
        # 批量导出的 JSON 只能表达 0/1；平票“疑似”按有火处理，降低漏报风险。
        return 0 if str(result.get("result", "no_fire")) == "no_fire" else 1

    def _meta_path_for_key(self, key: str) -> Path:
        return self.tmp_dir / "meta" / f"{safe_artifact_stem(key)}.json"

    def _load_cached_result(self, key: str) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
        meta_path = self._meta_path_for_key(key)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            result = dict(payload.get("result", {}))
            artifacts = dict(payload.get("artifacts", {}))
            artifacts.setdefault("meta_path", str(meta_path))
            result["batch_key"] = key
            result["batch_artifacts"] = artifacts
            return result, artifacts
        except Exception:
            return None

    @staticmethod
    def _compact_result_for_meta(result: Dict[str, Any]) -> Dict[str, Any]:
        keep = [
            "timestamp", "source_type", "source_path", "frame_index", "original_width", "original_height",
            "infer_width", "infer_height", "pred", "result", "result_cn", "prob_fire", "threshold",
            "height_resize_enabled", "resize_target_height", "resize_scale", "resize_output_width", "resize_output_height",
            "model_threshold", "manual_threshold", "threshold_source", "model_name", "model_path",
            "owl_positive_prompt_count", "owl_negative_prompt_count", "owl_suspicious_prompt_count",
            "global_logit", "local_logit", "semantic_logit", "raw_logit",
            "global_probability", "local_probability", "semantic_probability",
            "global_confidence", "local_confidence", "semantic_confidence",
            "branch_w_global", "branch_w_local",
            "semantic_positive", "semantic_negative", "semantic_suspicious",
            "router_change", "expert_data", "model_results",
        ]
        return {k: result.get(k) for k in keep if k in result}

    def _save_batch_artifacts(self, key: str, rgb: np.ndarray, result: Dict[str, Any], maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        meta_dir = self.tmp_dir / "meta"
        attn_dir = self.tmp_dir / "attention"
        meta_dir.mkdir(parents=True, exist_ok=True)
        if self.save_attention:
            attn_dir.mkdir(parents=True, exist_ok=True)

        stem = safe_artifact_stem(key)
        artifacts: Dict[str, Any] = {
            "tmp_dir": str(self.tmp_dir),
            "meta_path": str(meta_dir / f"{stem}.json"),
            "attention_paths": {},
            "overlay_paths": {},
        }
        if self.save_attention:
            for map_key, attn in maps.items():
                if BATCH_ATTENTION_CACHE_KEYS and map_key not in BATCH_ATTENTION_CACHE_KEYS:
                    continue
                map_path = attn_dir / f"{stem}_{map_key}.png"
                overlay_path = attn_dir / f"{stem}_{map_key}_overlay.jpg"
                imwrite_png(map_path, map_to_gray(attn))
                imwrite_jpg(overlay_path, heatmap_overlay(rgb, attn), quality=BATCH_OVERLAY_JPG_QUALITY)
                artifacts["attention_paths"][map_key] = str(map_path)
                artifacts["overlay_paths"][map_key] = str(overlay_path)

        meta_payload = {
            "key": key,
            "prediction_label": int(self._binary_export_label(result)),
            "result": self._compact_result_for_meta(result),
            "artifacts": artifacts,
        }
        with open(artifacts["meta_path"], "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)
        return artifacts

    def run(self):
        total = len(self.image_paths)
        stats = {
            "total": int(total),
            "done": 0,
            "fire": 0,
            "no_fire": 0,
            "suspect": 0,
            "failed": 0,
            "predictions": {},
            "errors": [],
            "root_dir": self.root_dir,
            "tmp_dir": str(self.tmp_dir),
            "stopped": False,
            "cached": 0,
            "batch_size": int(self.batch_size),
        }

        for batch_start in range(0, total, self.batch_size):
            if self._stop_requested:
                stats["stopped"] = True
                break
            batch_paths = self.image_paths[batch_start:batch_start + self.batch_size]
            for offset, path in enumerate(batch_paths, start=1):
                idx = batch_start + offset
                if self._stop_requested:
                    stats["stopped"] = True
                    break
                if not self._wait_if_paused():
                    stats["stopped"] = True
                    break
                key = self._json_key(path, self.root_dir)
                try:
                    raw_rgb = imread_rgb(path)
                    infer_rgb, valid_mask_np, resize_meta = prepare_rgb_for_inference(raw_rgb, **self.resize_settings)
                    cached = self._load_cached_result(key)
                    if cached is not None:
                        result, artifacts = cached
                        result = reclassify_result_by_threshold(
                            result,
                            self.model_settings,
                            float(getattr(self.engine, "model_threshold", DEFAULT_THRESHOLD)),
                            str(getattr(self.engine, "model_threshold_source", "model")),
                        )
                        maps: Dict[str, np.ndarray] = {}
                        stats["cached"] += 1
                    else:
                        meta = {
                            "source_type": "batch",
                            "source_path": path,
                            "batch_root": self.root_dir,
                            "frame_index": idx,
                            "original_width": int(raw_rgb.shape[1]),
                            "original_height": int(raw_rgb.shape[0]),
                            "infer_width": int(infer_rgb.shape[1]),
                            "infer_height": int(infer_rgb.shape[0]),
                            **resize_meta,
                        }
                        result, maps = self.engine.predict_rgb(infer_rgb, self.model_settings, meta, valid_mask_np)
                        artifacts = self._save_batch_artifacts(key, infer_rgb, result, maps)
                        result = dict(result)
                        result["batch_key"] = key
                        result["batch_artifacts"] = artifacts
                    result.setdefault("source_path", path)
                    result.setdefault("source_type", "batch")
                    label = self._binary_export_label(result)
                    stats["predictions"][key] = int(label)
                    stats["done"] += 1
                    if result.get("result") == "fire":
                        stats["fire"] += 1
                    elif result.get("result") == "suspect":
                        stats["suspect"] += 1
                    else:
                        stats["no_fire"] += 1
                    self.signals.progress.emit(idx, total, path, result, maps, infer_rgb)
                except Exception as e:
                    logger.exception("批量预测失败 | %s/%s | %s", idx, total, key)
                    stats["failed"] += 1
                    stats["errors"].append({"path": key, "error": str(e)})
                    self.signals.error.emit(f"{idx}/{total} {key}: {type(e).__name__}: {e}")

        self.signals.finished.emit(stats)


class FrameReaderThread(QtCore.QThread):
    """摄像头/视频读取线程。

    只负责读取原始帧并发给界面显示，不做模型推理。
    这样即使模型推理很慢，摄像头/视频预览也不会卡住 GUI 主线程。
    """

    frame_ready = QtCore.pyqtSignal(object, int)
    error = QtCore.pyqtSignal(str)
    finished_reading = QtCore.pyqtSignal()

    def __init__(self, source, source_type: str, parent=None):
        super().__init__(parent)
        self.source = source
        self.source_type = str(source_type)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                msg = f"无法打开视频源：{self.source}"
                logger.error("%s | source_type=%s", msg, self.source_type)
                self.error.emit(msg)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) if self.source_type == "video" else 0
            delay_ms = int(1000 / fps) if fps and fps > 1 else 33
            frame_index = 0

            while self._running:
                ok, bgr = cap.read()
                if not ok:
                    break
                frame_index += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb, frame_index)
                if delay_ms > 0:
                    self.msleep(delay_ms)
        except Exception as exc:
            logger.exception("视频/摄像头读取失败 | source=%s | type=%s", self.source, self.source_type)
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.finished_reading.emit()


