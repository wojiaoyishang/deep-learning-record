# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import *


class FireInferenceEngine:
    """FireArbiter-MoE-Lite single-model inference engine."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        local_models_first: bool = True,
        models_dir: str | Path = MODELS_DIR,
    ):
        self.model_path = Path(model_path)
        self.model_paths = [self.model_path]
        self.device = device
        self.local_models_first = bool(local_models_first)
        self.models_dir = Path(models_dir)
        self.model_name = self.model_path.name
        self.model_names = [self.model_name]
        self.model = None
        self.model_threshold = float(DEFAULT_THRESHOLD)
        self.model_threshold_source = "default"
        self.model_thresholds = [self.model_threshold]
        self.model_threshold_sources = [self.model_threshold_source]
        self.expert_names: List[str] = []
        self.load_model()

    @staticmethod
    def _model_config_expert_names(config: Dict[str, Any]) -> List[str]:
        expert_names = [str(name) for name in config["expert_names"]]
        num_experts = int(config["num_experts"])
        if len(expert_names) != num_experts:
            raise ValueError(
                f"model_config.expert_names数量为{len(expert_names)}，"
                f"但model_config.num_experts={num_experts}。"
            )
        return expert_names

    def load_model(self):
        checkpoint = safe_torch_load(self.model_path, map_location=self.device)
        if checkpoint.get("format") != FireArbiterMoELite.CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"当前 GUI 只支持 {FireArbiterMoELite.CHECKPOINT_FORMAT}，实际为 {checkpoint.get('format')!r}。"
            )

        config = dict(checkpoint["model_config"])
        expert_names = self._model_config_expert_names(config)

        owl_source, owl_is_local = resolve_gui_owl_source(
            config.get("owlvit_model_name"),
            models_dir=self.models_dir,
            local_first=self.local_models_first,
        )
        convnext_weight = resolve_gui_convnext_weight(
            models_dir=self.models_dir,
            local_first=self.local_models_first,
        )
        config["owlvit_model_name"] = owl_source

        with gui_local_model_loading(
            convnext_weight=convnext_weight,
            force_huggingface_offline=owl_is_local,
        ):
            model = FireArbiterMoELite(**config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        threshold, source = get_threshold_from_checkpoint_dict(checkpoint, DEFAULT_THRESHOLD)
        model.set_threshold(threshold)
        model.to(self.device).eval()
        self.model = model
        self.model_threshold = model.get_threshold()
        self.model_threshold_source = source
        self.model_thresholds = [self.model_threshold]
        self.model_threshold_sources = [source]
        self.expert_names = list(expert_names)

    @torch.no_grad()
    def predict_rgb(
        self,
        rgb: np.ndarray,
        model_settings: List[Dict[str, Any]],
        meta: Dict[str, Any],
        valid_mask_np: np.ndarray | None = None,
        map_keys: Tuple[str, ...] | None = None,
    ):
        if self.model is None:
            raise RuntimeError("模型尚未加载。")
        image = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).convert("RGB")
        raw_w, raw_h = image.size
        setting = dict(model_settings[0]) if model_settings else {}
        manual = bool(setting.get("manual_threshold", False))
        threshold = float(setting.get("threshold", self.model_threshold)) if manual else self.model_threshold

        self.model.set_prompts(
            setting["owl_positive_prompts"],
            setting["owl_negative_prompts"],
            setting["owl_suspicious_prompts"],
        )
        tensor = global_eval_transform(image).unsqueeze(0).to(self.device)
        if valid_mask_np is None:
            valid_mask = torch.ones((1, 1, tensor.shape[-2], tensor.shape[-1]), device=self.device)
        else:
            mask = np.asarray(valid_mask_np, dtype=np.float32)
            if mask.shape != tensor.shape[-2:]:
                mask = cv2.resize(mask, (tensor.shape[-1], tensor.shape[-2]), interpolation=cv2.INTER_NEAREST)
            valid_mask = torch.from_numpy(mask).to(self.device).view(1, 1, *tensor.shape[-2:])
        output = self.model(tensor, valid_mask)

        probability = float(output["prob_fire"].item())
        pred = int(probability >= threshold)
        requested = set(map_keys or ("global", "local", "semantic", "mix"))
        maps = {
            key: attention_to_canvas(output[f"attention_{key}"].squeeze(0), raw_w, raw_h)
            for key in ("global", "local", "semantic", "mix") if key in requested
        }

        prior = output["expert_prior_weights"][0].cpu().numpy()
        posterior = output["expert_posterior_weights"][0].cpu().numpy()
        expert_probs = output["expert_probs"][0].cpu().numpy()
        confidence = output["expert_confidence"][0].cpu().numpy()
        expert_data = []
        for index, name in enumerate(self.expert_names):
            expert_data.append({
                "expert_index": index,
                "expert_name": name,
                "weight": float(posterior[index]),
                "prior_weight": float(prior[index]),
                "posterior_weight": float(posterior[index]),
                "prob_fire": float(expert_probs[index]),
                "confidence": float(confidence[index]),
            })

        semantic_scores = output["semantic_scores"][0].cpu().tolist()
        branch_weights = output["branch_weights"][0].cpu().tolist()
        model_result = {
            "model_index": 0,
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "threshold": threshold,
            "used_threshold": threshold,
            "model_threshold": self.model_threshold,
            "manual_threshold": manual,
            "threshold_source": "manual" if manual else self.model_threshold_source,
            "prob_fire": probability,
            "pred": pred,
            "result": "fire" if pred else "no_fire",
            "result_cn": "有火" if pred else "无火",
        }
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_type": meta.get("source_type", "image"),
            "source_path": meta.get("source_path", ""),
            "frame_index": int(meta.get("frame_index", 0)),
            "original_width": int(meta.get("original_width", raw_w)),
            "original_height": int(meta.get("original_height", raw_h)),
            "infer_width": int(meta.get("infer_width", raw_w)),
            "infer_height": int(meta.get("infer_height", raw_h)),
            "height_resize_enabled": bool(meta.get("height_resize_enabled", meta.get("resize_enabled", False))),
            "resize_target_height": int(meta.get("resize_target_height", raw_h)),
            "resize_scale": float(meta.get("resize_scale", 1.0)),
            "resize_output_width": int(meta.get("resize_output_width", raw_w)),
            "resize_output_height": int(meta.get("resize_output_height", raw_h)),
            "pred": pred,
            "result": "fire" if pred else "no_fire",
            "result_cn": "有火" if pred else "无火",
            "prob_fire": probability,
            "threshold": threshold,
            "model_threshold": self.model_threshold,
            "manual_threshold": manual,
            "threshold_source": "manual" if manual else self.model_threshold_source,
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "owl_positive_prompt_count": len(setting["owl_positive_prompts"]),
            "owl_negative_prompt_count": len(setting["owl_negative_prompts"]),
            "owl_suspicious_prompt_count": len(setting["owl_suspicious_prompts"]),
            "global_logit": float(output["global_logit"].item()),
            "local_logit": float(output["local_logit"].item()),
            "semantic_logit": float(output["semantic_logit"].item()),
            "raw_logit": float(output["raw_logit"].item()),
            "global_probability": float(torch.sigmoid(output["global_logit"]).item()),
            "local_probability": float(torch.sigmoid(output["local_logit"]).item()),
            "semantic_probability": float(torch.sigmoid(output["semantic_logit"]).item()),
            "global_confidence": float(output["global_confidence"].item()),
            "local_confidence": float(output["local_confidence"].item()),
            "semantic_confidence": float(output["semantic_confidence"].item()),
            "branch_w_global": float(branch_weights[0]),
            "branch_w_local": float(branch_weights[1]),
            "semantic_positive": float(semantic_scores[0]),
            "semantic_negative": float(semantic_scores[1]),
            "semantic_suspicious": float(semantic_scores[2]),
            "router_change": int(np.argmax(prior) != np.argmax(posterior)),
            "expert_data": expert_data,
            "model_results": [model_result],
            "model_settings": model_settings,
            "tile_boxes": [],
            "tile_data": [],
        }
        return result, maps
