# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OWLVIT_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWLVIT_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_OWLVIT_POSITIVE_PROMPTS = (
    # 核心火焰
    "visible flame",
    "open flame",
    "active fire",
    "real fire",
    "burning flames",
    "fire",
    "visible combustion flame",
    "burning object with visible flames",
    "burning material with flames",
    "clear visible fire in the image",
    "a scene with visible flames",
    "a fire burning in the scene",
    "an object on fire",
    "a burning scene with clear flames",

    # 火焰尺寸、颜色和状态
    "small visible flame",
    "small flame",
    "tiny flame",
    "faint flame",
    "large visible flame",
    "large open flames",
    "spreading flames",
    "intense flames",
    "dense flames in the scene",
    "flickering flames",
    "orange flame",
    "yellow flame",
    "blue flame",
    "white hot flame",
    "red and orange flames",
    "fire spreading across an object",
    "fire spreading in a room",

    # 常见火焰对象
    "match flame",
    "lighter flame",
    "candle flame",
    "torch flame",
    "blowtorch flame",
    "campfire flames",
    "fireplace flames",
    "flaming debris",
    "burning wood with flames",
    "burning grass",
    "burning trash with visible flames",
    "burning furniture with visible flames",
    "burning electrical equipment with visible flames",
    "burning vegetation with visible flames",

    # 厨房、灶具和厨具下方火焰
    "gas stove flame",
    "gas burner flame",
    "kitchen stove flame",
    "cooking flame on a gas stove",
    "flame under cookware",
    "flame under a cooking pot",
    "flame under a frying pan",
    "flame under a pan",
    "fire under cookware",
    "fire under a cooking pot",
    "fire under a pan",
    "gas stove flame under a pot",
    "gas stove flame under a pan",
    "burner flame under cookware",
    "burner flame under a cooking pot",
    "burner flame below a pan",
    "visible flame under kitchen cookware",
    "visible flame below a pot",
    "visible flame below a pan",
    "visible gas flame beneath cookware",
    "a pot heated by visible flames",
    "a pan heated by visible flames",
    "visible flames around the bottom of a pot",
    "visible flames around the bottom of a pan",
    "kitchen cookware above an open flame",

    # 室内起火场景
    "a room on fire",
    "an indoor fire scene",
    "a kitchen fire",
    "a bedroom fire",
    "a living room fire",
    "a warehouse fire",
    "a shop on fire",
    "a restaurant kitchen fire",
    "a house fire",
    "a building on fire",
    "a building engulfed in flames",
    "flames coming out of a window",
    "flames coming from a doorway",
    "fire inside a building",
    "visible flames inside a room",
    "visible flames in an indoor scene",

    # 室外、车辆、工业和自然火灾
    "an outdoor fire scene",
    "a car on fire",
    "a truck on fire",
    "a bus on fire",
    "a motorcycle on fire",
    "a vehicle engulfed in flames",
    "burning vehicle",
    "an accident scene with visible flames",
    "industrial fire",
    "a factory fire with visible flames",
    "fire in an industrial facility",
    "forest fire",
    "wildfire",
    "a wildfire scene with visible flames",
    "grassland fire",
    "burning trees with visible flames",
    "burning bushes with visible flames",

    # 烟火共同出现
    "smoke and flames together",
    "visible flames surrounded by smoke",
    "fire with thick smoke",
    "fire with black smoke",
    "flames emerging through smoke",
    "smoke rising from visible flames",

    # 夜间、低清晰度和监控画面
    "visible flames in a dark scene",
    "visible flames in a night scene",
    "visible flames in a blurry scene",
    "visible flames in a low quality image",
    "visible flames in a surveillance camera scene",
    "surveillance footage of a fire",
    "cctv footage showing a fire",
    "cctv footage showing visible flames",
    "cctv footage showing flames in a room",
    "security camera footage of visible flames",
    "security camera view of a burning object",
    "monitoring camera capturing a fire",
    "fire visible in security camera footage",

    # 火焰视频、素材、动画和视觉特效
    "video footage of fire",
    "video footage of flames",
    "fire footage",
    "flame footage",
    "fire stock footage",
    "flame stock footage",
    "fire visual material",
    "flame visual material",
    "animated flame effect",
    "animated fire effect",
    "flame animation",
    "fire animation",
    "flame visual effect",
    "fire visual effect",
    "flame special effect",
    "fire special effect",
    "flame overlay",
    "fire overlay",
    "flame overlay effect",
    "fire overlay effect",
    "isolated flame",
    "isolated fire flame",
    "flame element",
    "fire element",
    "digital flame with visible fire",
    "computer generated visible flames",
)

DEFAULT_OWLVIT_NEGATIVE_PROMPTS = (
    # 普通灯具和人工光源
    "electric lamp",
    "ceiling lamp",
    "wall lamp",
    "table lamp",
    "floor lamp",
    "street light",
    "street lamp",
    "led light",
    "neon light",
    "decorative light",
    "stage light",
    "spotlight",
    "orange lamp",
    "red lamp",
    "yellow lamp",
    "bright lamp",
    "warm colored light source",
    "decorative candle-shaped lamp",
    "fake flame light",
    "artificial flame shaped decoration",

    # 车辆灯光
    "car headlight",
    "car tail light",
    "vehicle brake light",
    "motorcycle headlight",
    "truck headlight",
    "traffic light",
    "red traffic light",
    "yellow traffic light",
    "warning beacon light",
    "emergency vehicle light",
    "flashing warning light",

    # 反光、眩光和曝光
    "light reflection",
    "glass reflection",
    "metal reflection",
    "water reflection",
    "reflection on a wet road",
    "bright reflection on metal",
    "bright reflection on glass",
    "specular highlight",
    "headlight glare",
    "lens flare",
    "camera flare",
    "sunlight glare",
    "reflected sunlight",
    "overexposed bright area",
    "overexposed light source",
    "bright illuminated window",
    "glowing window",
    "bright illuminated object",

    # 太阳、天空和云层
    "sunset sky",
    "sunrise sky",
    "orange sky",
    "red sky",
    "yellow sky",
    "sun behind clouds",
    "sunlight through clouds",
    "glowing sunset clouds",
    "bright orange sunset",
    "sun near the horizon",
    "bright sunlight",
    "sunlight shining through trees",
    "red clouds at sunset",
    "orange clouds at sunset",

    # 暖色物体和自然物
    "orange object",
    "red object",
    "yellow object",
    "orange plastic object",
    "yellow plastic object",
    "red plastic object",
    "orange fabric",
    "red fabric",
    "yellow fabric",
    "yellow flower",
    "orange flower",
    "orange leaves",
    "autumn leaves",
    "red leaves",
    "orange safety vest",
    "yellow safety vest",
    "orange clothing",
    "red clothing",

    # 屏幕、字幕、广告和电子显示
    "video subtitle",
    "white subtitle text",
    "yellow subtitle text",
    "red subtitle text",
    "image watermark",
    "bright text overlay",
    "screen graphic overlay",
    "television screen",
    "computer monitor",
    "mobile phone screen",
    "illuminated screen",
    "bright signboard",
    "colorful advertisement",
    "digital billboard",
    "large outdoor led screen",
    "electronic signboard",
    "led display",
    "led display screen",
    "led message board",
    "electronic scrolling sign",
    "red led text",
    "scrolling red led text",
    "red text on an led display",
    "orange glowing sign",
    "red glowing sign",
    "bright stage screen",
    "concert lighting",
    "stage video screen",

    # 道路、施工和警示设施
    "traffic cone",
    "orange traffic cone",
    "road cone",
    "safety cone",
    "construction cone",
    "traffic barrel",
    "orange traffic barrel",
    "road safety barrel",
    "crash barrel",
    "impact attenuator barrel",
    "water-filled traffic barrier",
    "plastic road barrier",
    "orange road barrel",
    "yellow warning sign",
    "road warning light",
    "reflective road marker",
    "construction warning object",
    "orange roadside object",
    "road construction equipment",

    # 不存在明火的烹饪和加热设备
    "electric stove without visible flame",
    "induction cooker",
    "electric hot plate",
    "glowing electric heater",
    "red heating coil",
    "toaster heating element",
    "oven light",
    "microwave oven light",
    "hot cookware without visible flames",
    "steam rising from cookware",
    "boiling water steam",
    "cooking steam without fire",

    # 其他容易被误识别为火焰的场景
    "bright orange reflection",
    "bright red reflection",
    "warm indoor lighting",
    "yellow indoor lighting",
    "orange decorative object",
    "red decorative object",
    "illuminated advertisement",
    "bright colored stage lighting",
    "colorful light installation",
    "glowing decorative ornament",
)

DEFAULT_OWLVIT_SUSPICIOUS_PROMPTS = (
    # 点火设备和潜在火源
    "lighter",
    "cigarette lighter",
    "pocket lighter",
    "gas lighter",
    "disposable lighter",
    "butane lighter",
    "match",
    "matchstick",
    "wooden match",
    "lit match",
    "box of matches",
    "match head",
    "glowing match head",
    "candle",
    "candle wick",
    "torch",
    "blowtorch",
    "gas torch",
    "ignition device",
    "fire starter",
    "fire starter stick",
    "burning fuse",
    "possible ignition source",

    # 灶具、加热和烹饪设备
    "gas stove burner",
    "portable gas stove",
    "camping stove",
    "stove burner",
    "barbecue grill",
    "charcoal grill",
    "fireplace",
    "fire pit",
    "incense stick",
    "hot cooking appliance",
    "overheated cooking appliance",
    "glowing stove burner",
    "small burner flame partly hidden by cookware",
    "possible flame hidden under cookware",

    # 烟雾
    "smoke",
    "thin smoke",
    "heavy smoke",
    "white smoke",
    "gray smoke",
    "black smoke",
    "dark smoke",
    "rising smoke",
    "smoke plume",
    "smoke cloud",
    "smoke coming from an object",
    "smoke coming from electrical equipment",
    "smoke in a room",
    "smoke in an indoor scene",
    "smoke in a surveillance camera scene",
    "surveillance footage showing smoke",
    "cctv footage showing suspicious smoke",
    "security camera view of a smoking object",
    "scene with smoke but no visible flames",

    # 阴燃、余烬和烧焦痕迹
    "smoldering material",
    "smoking object",
    "smoldering object",
    "smoldering wooden stick",
    "burning wooden stick",
    "burning cigarette",
    "cigarette ember",
    "ember",
    "glowing ember",
    "hot coal",
    "glowing charcoal",
    "ash",
    "hot ash",
    "burnt object",
    "charred object",
    "scorched surface",
    "burnt surface",
    "burnt debris",
    "smoldering debris",
    "blackened material",
    "possible combustion trace",

    # 电气异常和过热
    "smoking electrical device",
    "battery smoke",
    "overheated battery",
    "swollen battery",
    "overheated appliance",
    "overheated object",
    "overheated electrical equipment",
    "hot electrical device",
    "damaged electrical wire",
    "overheated wire",
    "sparking electrical wire",
    "burnt electrical outlet",
    "electrical overheating",
    "short circuit",
    "short circuit spark",
    "glowing heating element",
    "red hot heating element",

    # 火花、电焊和电弧
    "electrical spark",
    "small electrical sparks",
    "flying sparks",
    "spark shower",
    "welding sparks",
    "welding arc",
    "electric arc",
    "grinding sparks",
    "metal cutting sparks",
    "scene with sparks but no visible flames",

    # 高温和发光物体
    "hot metal",
    "red hot metal",
    "glowing metal",
    "hot glowing object",
    "glowing hot spot",
    "orange glowing region",
    "yellow glowing region",
    "red glowing region",
    "fire-like bright region",
    "flickering bright spot",
    "small suspicious bright region",
    "localized hot spot",

    # 烟火、爆燃和其他可疑现象
    "firework",
    "signal flare",
    "sparkler",
    "possible early fire sign",
    "possible combustion",
    "possible hidden flame",
    "partially obscured flame",
    "faint possible flame",
    "small fire-like region",
    "smoke and a suspicious bright region",
    "heat damage around an object",
    "recently extinguished fire",
    "residual smoke after a fire",
)


@dataclass(frozen=True)
class PromptBank:
    positive: Tuple[str, ...] = DEFAULT_OWLVIT_POSITIVE_PROMPTS
    negative: Tuple[str, ...] = DEFAULT_OWLVIT_NEGATIVE_PROMPTS
    suspicious: Tuple[str, ...] = DEFAULT_OWLVIT_SUSPICIOUS_PROMPTS


@dataclass
class SemanticEvidence:
    group_maps: torch.Tensor  # 三组语义分数图 (B, 3, S/P, S/P)
    attention: torch.Tensor  # 语义注意力图 (B, 1, H, W)
    feature: torch.Tensor  # 新语义视觉特征 / 语义特征图 (B, D, S/P, S/P)
    vector: torch.Tensor  # 语义特征向量  (B, D)
    scores: torch.Tensor  # 三组提示词语义得分  (B, 3)
    confidence: torch.Tensor  # 语义置信度 (B, 1)
    logit: torch.Tensor  # 语义分类分数 (B, 1)


@dataclass
class ExpertEvidence:
    feature: torch.Tensor
    consensus: torch.Tensor
    prior_weights: torch.Tensor
    posterior_weights: torch.Tensor
    spatial_weights: torch.Tensor
    advice_logits: torch.Tensor
    expert_logits: torch.Tensor
    confidence: torch.Tensor
    prior_routed_logit: torch.Tensor
    posterior_routed_logit: torch.Tensor


@dataclass
class VisualPyramid:
    mid: torch.Tensor
    final: torch.Tensor
    mask: Optional[torch.Tensor]


@dataclass
class FusionContext:
    semantic: SemanticEvidence
    expert: ExpertEvidence


@dataclass
class BranchEvidence:
    logit: torch.Tensor
    confidence: torch.Tensor
    feature: torch.Tensor
    attention: torch.Tensor


@dataclass
class BranchPair:
    global_branch: BranchEvidence
    local_branch: BranchEvidence


@dataclass
class PriorRoute:
    logits: torch.Tensor  # 路由分数 (B, E)
    weights: torch.Tensor  # 路由权重 (B, E)
    advice_logits: torch.Tensor  # 自荐分数 (B, E)
    advice_features: torch.Tensor  # 自荐特征 (B, E, A)


@dataclass
class ExpertEvaluation:
    logits: torch.Tensor  # 专家共识分数
    confidence_logits: torch.Tensor  # 专家共识置信度分数
    reason: torch.Tensor  # 专家共识原因
    agreement: torch.Tensor  # 专家共识相似度


class SwiGLU(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=self.dim)
        return value * F.silu(gate)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = max(16, channels * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1),
            SwiGLU(dim=1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.scale) * self.net(x)


class SoftRegionPool(nn.Module):
    def forward(
            self,
            feature: torch.Tensor,
            attention: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn = F.interpolate(attention.float(), size=feature.shape[-2:], mode="bilinear", align_corners=False)
        if mask is not None:
            valid = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
            attn = attn * valid
        denom = attn.sum(dim=(2, 3)).clamp_min(1e-6)
        return (feature * attn).sum(dim=(2, 3)) / denom


class ConvNeXtFeatureBackbone(nn.Module):
    """ConvNeXt-Tiny：feature[5] 为局部细节，feature[6] 为 MoE 插入点"""

    def __init__(self, pretrained: bool = True, trainable_layers: int = 1):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        self.features = model.features
        self.mid_channels = 384
        self.expert_channels = 768
        self.out_channels = 768
        self.trainable_layers = int(trainable_layers)
        self.set_trainable_layers(self.trainable_layers)

    def set_trainable_layers(self, count: int) -> None:
        self.trainable_layers = int(count)
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        modules = list(self.features) if count < 0 else list(self.features)[-max(0, count):]
        for module in modules:
            module.requires_grad_(True)

    def forward_until(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = images
        mid = None
        for index in range(7):
            out = self.features[index](out)
            if index == 5:
                mid = out
        if mid is None:
            raise RuntimeError("ConvNeXt feature[5] was not produced.")
        return mid, out

    def forward_from(self, feature: torch.Tensor) -> torch.Tensor:
        return self.features[7](feature)


class ThreeGroupPromptBuilder(nn.Module):
    """
    将每个提示词的激活程度处理为一个激活值，输入的维度大小是 (B, N, Q)
    N 是图片 Patch 数量，Q 是每个 Patch 对应提示词的强度
    问题在于，Q 实际上是 Q1 Q2 Q3（分别对应正向、负向、疑似）多个提示词构成的，
    这个模块类的目的是把 Q1 Q2 Q3 三个组各求一个强度出来，来表示 正向、负向、疑似 三个组提示词的强度。
    """

    def __init__(self, temperature: float = 0.7):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))

    def _aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(0.2, 2.0)
        count = logits.shape[-1]
        return temperature * (torch.logsumexp(logits / temperature, dim=-1) - math.log(count))

    def forward(self, logits: torch.Tensor, slices: Tuple[slice, slice, slice]) -> torch.Tensor:
        return torch.stack([self._aggregate(logits[:, :, group_slice]) for group_slice in slices], dim=-1)


class GroupScorePool(nn.Module):
    def __init__(self, ratios: Sequence[float] = (0.08, 0.10, 0.15)):
        super().__init__()
        self.ratios = tuple(float(value) for value in ratios)

    def forward(self, group_logits: torch.Tensor) -> torch.Tensor:
        scores = []
        patch_count = group_logits.shape[1]
        for group_index, ratio in enumerate(self.ratios):
            k = max(1, int(round(patch_count * ratio)))
            scores.append(group_logits[:, :, group_index].topk(k, dim=1).values.mean(dim=1))
        return torch.stack(scores, dim=1)


class GroupScoreCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3))
        self.log_temperature = nn.Parameter(torch.zeros(3))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(0.25, 4.0)
        return (scores - self.bias) / temperature


class SoftBoxField(nn.Module):
    """Convert OWL boxes to a broad Gaussian influence field, never a hard rectangle."""

    def __init__(self, topk: int = 12, spread: float = 0.85, minimum_sigma: float = 0.08):
        super().__init__()
        self.topk = int(topk)
        self.spread = float(spread)
        self.minimum_sigma = float(minimum_sigma)

    def forward(self, boxes: torch.Tensor, patch_scores: torch.Tensor) -> torch.Tensor:
        batch, patch_count, _ = boxes.shape
        grid_size = int(round(math.sqrt(patch_count)))
        if grid_size * grid_size != patch_count:
            raise RuntimeError(f"OWL patch count {patch_count} is not a square grid.")
        k = min(self.topk, patch_count)
        values, indices = patch_scores.topk(k, dim=1)
        selected = boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4)).float()
        weights = torch.softmax(values.float(), dim=1)

        axis = torch.linspace(0.0, 1.0, grid_size, device=boxes.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        xx = xx.view(1, 1, grid_size, grid_size)
        yy = yy.view(1, 1, grid_size, grid_size)
        cx, cy, width, height = selected.unbind(dim=-1)
        sigma_x = (width.abs() * self.spread).clamp_min(self.minimum_sigma).unsqueeze(-1).unsqueeze(-1)
        sigma_y = (height.abs() * self.spread).clamp_min(self.minimum_sigma).unsqueeze(-1).unsqueeze(-1)
        field = torch.exp(
            -0.5 * (((xx - cx.unsqueeze(-1).unsqueeze(-1)) / sigma_x) ** 2
                    + ((yy - cy.unsqueeze(-1).unsqueeze(-1)) / sigma_y) ** 2)
        )
        field = (field * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        return field / field.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)


class SemanticMapDecoder(nn.Module):
    def __init__(self, owl_dim: int, output_dim: int = 128):
        super().__init__()
        self.feature_proj = nn.Conv2d(owl_dim, output_dim, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim + 2, output_dim, 1),
            ConvFFN(output_dim, expansion=2),
            nn.Conv2d(output_dim, 1, 1),
        )
        self.box_scale = nn.Parameter(torch.tensor(-1.5))

    def forward(
            self,
            owl_feature: torch.Tensor,
            seed: torch.Tensor,
            box_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.feature_proj(owl_feature)
        seed = F.interpolate(seed, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        box_prior = F.interpolate(box_prior, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.fusion(torch.cat([feature, seed, box_prior], dim=1))
        seed_logit = torch.logit(seed.clamp(1e-4, 1.0 - 1e-4))
        attention = torch.sigmoid(seed_logit + torch.sigmoid(self.box_scale) * box_prior + residual)
        return attention, feature


class SemanticEvidenceHead(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.pool = SoftRegionPool()
        self.vector_head = MLP(channels + 6, 192, channels, dropout=0.05)
        self.confidence_head = MLP(channels + 6, 96, 1, dropout=0.05)
        self.logit_head = MLP(channels + 6, 128, 1, dropout=0.05)

    @staticmethod
    def _statistics(attention: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        flat = attention.flatten(1).clamp(1e-6, 1.0 - 1e-6)
        coverage = flat.mean(dim=1, keepdim=True)
        peak = flat.amax(dim=1, keepdim=True)
        entropy = -(flat * flat.log() + (1.0 - flat) * (1.0 - flat).log()).mean(dim=1, keepdim=True)
        return torch.cat([scores, coverage, peak, entropy], dim=1)

    def forward(
            self,
            feature: torch.Tensor,
            attention: torch.Tensor,
            scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool(feature, attention)
        descriptor = torch.cat([pooled, self._statistics(attention, scores)], dim=1)
        return (
            self.vector_head(descriptor),
            torch.sigmoid(self.confidence_head(descriptor)),
            self.logit_head(descriptor),
        )


class OWLSemanticEncoder(nn.Module):
    """Single OWL forward producing three prompt maps and a soft semantic attention map."""

    def __init__(
            self,
            model_name: Optional[str],
            prompt_bank: PromptBank,
            image_size: int = 384,
            feature_dim: int = 128,
            vision_tail_layers: int = 2,
            text_tail_layers: int = 0,
            train_class_head: bool = True,
            train_box_head: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.prompt_bank = prompt_bank
        self.image_size = int(image_size)
        self.feature_dim = int(feature_dim)
        self.vision_tail_layers = int(vision_tail_layers)
        self.text_tail_layers = int(text_tail_layers)
        self.train_class_head = bool(train_class_head)
        self.train_box_head = bool(train_box_head)
        self.model = None
        self.processor = None
        self.trainable_owl_modules: List[nn.Module] = []

        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("owl_mean", torch.tensor(OWLVIT_IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("owl_std", torch.tensor(OWLVIT_IMAGE_STD).view(1, 3, 1, 1), persistent=False)

        owl_dim = 512
        self.supports_interpolate = False
        if model_name:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor  # type: ignore
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name)

            owl_dim = int(self.model.config.vision_config.hidden_size)

            self.supports_interpolate = "interpolate_pos_encoding" in inspect.signature(self.model.forward).parameters
            self._configure_tuning()

        self.prompt_builder = ThreeGroupPromptBuilder()
        self.score_pool = GroupScorePool()
        self.calibrator = GroupScoreCalibrator()
        self.box_field = SoftBoxField()
        self.map_decoder = SemanticMapDecoder(owl_dim, feature_dim)
        self.evidence_head = SemanticEvidenceHead(feature_dim)
        self.suspicious_scale = nn.Parameter(torch.tensor(-0.4))
        self.negative_scale = nn.Parameter(torch.tensor(0.0))
        self._cache_prompt_tokens()

    def _cache_prompt_tokens(self) -> None:
        prompts = self.prompt_bank.positive + self.prompt_bank.negative + self.prompt_bank.suspicious
        p0 = len(self.prompt_bank.positive)
        p1 = p0 + len(self.prompt_bank.negative)
        self.group_slices = (slice(0, p0), slice(p0, p1), slice(p1, len(prompts)))
        if self.processor is None:
            self._prompt_tokens = None
            return
        tokens = self.processor.tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt")
        self._prompt_tokens = {key: value for key, value in tokens.items()}

    def set_prompts(self, positive: Sequence[str], negative: Sequence[str], suspicious: Sequence[str]) -> None:
        self.prompt_bank = PromptBank(tuple(positive), tuple(negative), tuple(suspicious))
        self._cache_prompt_tokens()

    @staticmethod
    def _tail_layers(root: nn.Module, count: int) -> List[nn.Module]:
        encoder = getattr(root, "encoder")
        layers = list(getattr(encoder, "layers"))
        return layers[-max(0, count):] if count > 0 else []

    def _configure_tuning(self) -> None:
        if self.model is None:
            return
        self.model.requires_grad_(False)
        core = getattr(self.model, "owlvit", self.model)
        modules: List[nn.Module] = []
        modules.extend(self._tail_layers(core.vision_model, self.vision_tail_layers))
        modules.extend(self._tail_layers(core.text_model, self.text_tail_layers))
        for name, module in self.model.named_children():
            if name == "owlvit":
                continue
            is_box_head = "box" in name.lower()
            if is_box_head and self.train_box_head:
                modules.append(module)
            elif not is_box_head and self.train_class_head:
                modules.append(module)
        for module in modules:
            module.requires_grad_(True)
        self.trainable_owl_modules = modules

    def set_owl_trainable(self, enabled: bool) -> None:
        for module in self.trainable_owl_modules:
            module.requires_grad_(enabled)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.model is not None:
            self.model.eval()
            if mode:
                for module in self.trainable_owl_modules:
                    if any(parameter.requires_grad for parameter in module.parameters()):
                        module.train()
        return self

    def _letterbox(self, images: torch.Tensor, valid_mask: Optional[torch.Tensor]):
        rgb = (images * self.imagenet_std.to(images.device) + self.imagenet_mean.to(images.device)).clamp(0.0, 1.0)
        pixels, metadata = [], []
        canvas_size = self.image_size
        for index in range(images.shape[0]):
            if valid_mask is None:
                height, width = images.shape[-2:]
            else:
                mask = valid_mask[index, 0] > 0.5
                rows = torch.where(mask.any(dim=1))[0]
                cols = torch.where(mask.any(dim=0))[0]
                height = int(rows[-1].item()) + 1
                width = int(cols[-1].item()) + 1
            crop = rgb[index:index + 1, :, :height, :width]
            scale = min(canvas_size / max(width, 1), canvas_size / max(height, 1))
            new_h = max(1, int(round(height * scale)))
            new_w = max(1, int(round(width * scale)))
            resized = F.interpolate(crop, size=(new_h, new_w), mode="bicubic", align_corners=False)
            pad_y = (canvas_size - new_h) // 2
            pad_x = (canvas_size - new_w) // 2
            canvas = resized.new_zeros((1, 3, canvas_size, canvas_size))
            canvas[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            pixels.append((canvas - self.owl_mean.to(images.device)) / self.owl_std.to(images.device))
            metadata.append((height, width, new_h, new_w, pad_y, pad_x))
        return torch.cat(pixels, dim=0), metadata

    def _restore(self, tensor: torch.Tensor, metadata, canvas_hw: Tuple[int, int]) -> torch.Tensor:
        restored = tensor.new_zeros((tensor.shape[0], tensor.shape[1], canvas_hw[0], canvas_hw[1]))
        full = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        for index, (height, width, new_h, new_w, pad_y, pad_x) in enumerate(metadata):
            crop = full[index:index + 1, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w]
            restored[index:index + 1, :, :height, :width] = F.interpolate(
                crop, size=(height, width), mode="bilinear", align_corners=False
            )
        return restored

    def _owl_forward(self, pixels: torch.Tensor):
        if self.model is None or self._prompt_tokens is None:
            raise RuntimeError("OWL-ViT is not configured.")
        device = next(self.model.parameters()).device
        tokens = {key: value.to(device) for key, value in self._prompt_tokens.items()}
        batch = pixels.shape[0]
        query_count = tokens["input_ids"].shape[0]
        input_ids = tokens["input_ids"].unsqueeze(0).expand(batch, -1, -1).reshape(batch * query_count, -1)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).expand(batch, -1, -1).reshape(batch * query_count, -1)
        kwargs = {"pixel_values": pixels.to(device), "input_ids": input_ids, "attention_mask": attention_mask}
        if self.supports_interpolate:
            kwargs["interpolate_pos_encoding"] = True

        # OWL被冻结时不保留其视觉/文本编码器的反向计算图。
        # 新增的SemanticMapDecoder仍在该上下文之外正常训练。
        owl_trainable = any(parameter.requires_grad for parameter in self.model.parameters())
        grad_context = nullcontext() if owl_trainable else torch.no_grad()
        with grad_context:
            return self.model(**kwargs)

    @staticmethod
    def _image_feature(outputs) -> torch.Tensor:
        feature = outputs.image_embeds  # (B, S/P, S/P, D)  D - 内部隐藏维度大小
        if feature.ndim == 4:
            if feature.shape[1] < feature.shape[-1]:
                feature = feature.permute(0, 3, 1, 2)
        elif feature.ndim == 3:
            batch, patches, channels = feature.shape
            size = int(round(math.sqrt(patches)))
            feature = feature.transpose(1, 2).reshape(batch, channels, size, size)
        else:
            raise RuntimeError(f"Unsupported OWL image feature shape: {tuple(feature.shape)}")
        return feature.float()

    def forward(self, images: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> SemanticEvidence:
        if self.model is None:
            batch, _, height, width = images.shape
            zeros_map = images.new_zeros((batch, 1, height, width))
            grid_h = max(1, height // 32)
            grid_w = max(1, width // 32)
            zeros_grid = images.new_zeros((batch, 1, grid_h, grid_w))
            zeros_feature = images.new_zeros((batch, self.feature_dim, grid_h, grid_w))
            return SemanticEvidence(
                group_maps=zeros_grid.expand(-1, 3, -1, -1), attention=zeros_map,
                feature=zeros_feature, vector=images.new_zeros((batch, self.feature_dim)),
                scores=images.new_zeros((batch, 3)), confidence=images.new_zeros((batch, 1)),
                logit=images.new_zeros((batch, 1)),
            )

        pixels, metadata = self._letterbox(images, valid_mask)
        outputs = self._owl_forward(pixels)
        logits = outputs.logits.float().to(images.device)  # (B, N, Q)
        boxes = outputs.pred_boxes.float().to(images.device).detach()  # (B, N, 4)
        owl_feature = self._image_feature(outputs).to(images.device)  # (B, D, S/P, S/P)

        group_logits = self.prompt_builder(logits, self.group_slices)  # (B, N, 3)
        scores = self.calibrator(self.score_pool(group_logits))  # (B, 3)

        patch_count = group_logits.shape[1]
        grid = int(round(math.sqrt(patch_count)))

        if grid * grid != patch_count:
            raise RuntimeError(
                f"OWL patch count {patch_count} cannot form a square grid."
            )

        # 先恢复为二维网格，避免重复 reshape。
        group_logits_grid = group_logits.transpose(1, 2).reshape(
            images.shape[0],
            3,
            grid,
            grid,
        )

        positive_logits = group_logits_grid[:, 0:1]
        negative_logits = group_logits_grid[:, 1:2]
        suspicious_logits = group_logits_grid[:, 2:3]

        negative = torch.sigmoid(negative_logits)
        suspicious = torch.sigmoid(suspicious_logits)

        # 将正向证据和负向干扰直接融合为火焰种子图，
        negative_weight = torch.sigmoid(self.negative_scale)
        fire_seed = torch.sigmoid(
            positive_logits - negative_weight * negative_logits
        )  # 火焰种子图

        # 可疑证据只作为补充线索。
        suspicious_weight = torch.sigmoid(self.suspicious_scale)
        suspicious_seed = suspicious * suspicious_weight  # 疑似种子图

        # 火焰证据与可疑证据采用概率 OR 融合。
        inspection_seed = 1.0 - (
                (1.0 - fire_seed) * (1.0 - suspicious_seed)
        )

        # 使用负向语义进行宽容抑制，最多降低 50%。
        inspection_seed = inspection_seed * (1.0 - 0.5 * negative)

        # 第一通道直接使用融合后的 fire_seed，
        # 使诊断图与后续 inspection_seed 的计算逻辑保持一致。
        group_maps_grid = torch.cat(
            [fire_seed, negative, suspicious],
            dim=1,
        )

        # 使用原始组级 logits 计算候选框的重要程度。
        box_scores = (
                group_logits[:, :, 0]
                + 0.25 * group_logits[:, :, 2]
        )  # 框置信度

        box_prior = self.box_field(
            boxes,
            box_scores,
        )  # 高斯先验图

        attention_grid, semantic_grid = self.map_decoder(
            owl_feature,
            inspection_seed,
            box_prior,
        )  # 语义注意力 新语义特征

        canvas_hw = images.shape[-2:]

        # 三组Prompt图只用于诊断，不参与后续梯度计算；保持OWL网格尺寸即可。
        group_maps = group_maps_grid.detach()

        # Semantic Attention只有1个通道，恢复到输入画布用于宽容Mask监督和GUI显示。
        attention = self._restore(attention_grid, metadata, canvas_hw)  # 注意力
        if valid_mask is not None:
            attention = attention * valid_mask

        # LocalSemanticFusion会在真正需要时直接插值到局部特征尺寸，禁止先放大到原图再缩小。
        vector, confidence, logit = self.evidence_head(semantic_grid, attention_grid, scores)
        return SemanticEvidence(group_maps, attention, semantic_grid, vector, scores, confidence, logit)


class ExpertConvAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = max(32, channels // int(bottleneck_ratio))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1),
            SwiGLU(dim=1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class PriorExpertRouter(nn.Module):
    """
    先验专家路由
    """

    def __init__(self, channels: int, semantic_dim: int, num_experts: int, advice_dim: int = 16):
        super().__init__()
        self.num_experts = num_experts
        self.advice_dim = advice_dim
        base_dim = channels * 2 + semantic_dim + 4
        self.advice_heads = nn.ModuleList([MLP(base_dim, 128, advice_dim + 1) for _ in range(num_experts)])
        self.router = MLP(base_dim + num_experts * (advice_dim + 1), 256, num_experts)

    def forward(self, feature: torch.Tensor, semantic: SemanticEvidence) -> PriorRoute:
        pooled = torch.cat([feature.mean((2, 3)), feature.amax((2, 3))], dim=1)
        base = torch.cat([pooled, semantic.vector, semantic.scores, semantic.confidence], dim=1)
        advice = torch.stack([head(base) for head in self.advice_heads], dim=1)
        advice_logits = advice[:, :, 0]
        advice_features = advice[:, :, 1:]
        logits = self.router(torch.cat([base, advice.flatten(1)], dim=1)) + advice_logits
        return PriorRoute(logits, torch.softmax(logits, dim=1), advice_logits, advice_features)


class SpatialExpertMixer(nn.Module):
    """
    空间专家混合器，用于把若干个专家给出的图片特征混合为一个新的特征，这个特征中包含了所有专家的输出。
    """

    def __init__(self, channels: int, key_dim: int = 64):
        super().__init__()
        self.query = nn.Conv2d(channels, key_dim, 1)
        self.key = nn.Conv2d(channels, key_dim, 1)
        self.scale = nn.Parameter(torch.tensor(-1.0))

    def compatibility(self, feature: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        这个函数相当于是做了一个 空间专家门控机制 ，用于计算每个像素与专家的对应程度。
        """
        batch, experts, channels, height, width = deltas.shape
        query = self.query(feature).unsqueeze(1)
        keys = self.key(deltas.reshape(batch * experts, channels, height, width)).reshape(
            batch, experts, -1, height, width
        )
        return (query * keys).sum(dim=2) / math.sqrt(keys.shape[2])

    def remix(self, feature: torch.Tensor, deltas: torch.Tensor, compatibility: torch.Tensor,
              weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # weights: 路由权重
        logits = compatibility + weights.clamp_min(1e-6).log().unsqueeze(-1).unsqueeze(-1)
        spatial_weights = torch.softmax(logits, dim=1)
        # deltas: 各专家图像特征 (B, E, C, H, W)
        delta = (spatial_weights.unsqueeze(2) * deltas).sum(dim=1)
        # 混合图片特征 空间权重
        return feature + torch.sigmoid(self.scale) * delta, spatial_weights


class ExpertDescriptor(nn.Module):
    """
    专家解释器，用于输出每个专家对目前输入图像的理解
    """

    def __init__(self, channels: int, output_dim: int = 128):
        super().__init__()
        self.project = MLP(channels * 3, 256, output_dim, dropout=0.05)

    def forward(self, feature: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        batch, experts, channels, _, _ = deltas.shape
        base = feature.unsqueeze(1) + deltas
        descriptor = torch.cat([
            deltas.mean((3, 4)), deltas.amax((3, 4)), base.mean((3, 4))
        ], dim=-1)
        return self.project(descriptor.reshape(batch * experts, -1)).reshape(batch, experts, -1)


class ExpertConsensusProjector(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.project = MLP(feature_dim, feature_dim * 2, feature_dim)

    def forward(self, description_feature: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        consensus = (description_feature * weights.unsqueeze(-1)).sum(dim=1)
        return self.project(consensus)


class ExpertSelfEvaluator(nn.Module):
    def __init__(self, feature_dim: int, advice_dim: int, reason_dim: int = 16):
        super().__init__()
        self.reason_dim = reason_dim
        self.head = MLP(feature_dim * 2 + advice_dim + 1, 192, reason_dim + 2, dropout=0.05)

    def forward(
            self,
            description_features: torch.Tensor,
            consensus: torch.Tensor,
            route: PriorRoute,
    ) -> ExpertEvaluation:
        batch, experts, _ = description_features.shape
        expanded_consensus = consensus.unsqueeze(1).expand(-1, experts, -1)
        inputs = torch.cat([
            description_features, expanded_consensus, route.advice_features, route.weights.unsqueeze(-1)
        ], dim=-1)
        # MLP 过的是软限制，是自我学习的，来判断自己的判断自不自信、是什么理由
        output = self.head(inputs.reshape(batch * experts, -1)).reshape(batch, experts, -1)
        # agreement 是硬性限制，直接看出该专家的建议是否和公式一致
        agreement = F.cosine_similarity(description_features, expanded_consensus, dim=-1)
        return ExpertEvaluation(output[:, :, 0], output[:, :, 1], output[:, :, 2:], agreement)


class PosteriorExpertRouter(nn.Module):
    def __init__(self, reason_dim: int = 16):
        super().__init__()
        self.correction = MLP(reason_dim + 5, 64, 1)

    def forward(self, route: PriorRoute, evaluation: ExpertEvaluation) -> torch.Tensor:
        inputs = torch.cat([
            route.logits.unsqueeze(-1), route.advice_logits.unsqueeze(-1),
            evaluation.logits.unsqueeze(-1), evaluation.confidence_logits.unsqueeze(-1),
            evaluation.agreement.unsqueeze(-1), evaluation.reason,
        ], dim=-1)
        batch, experts, _ = inputs.shape
        correction = self.correction(inputs.reshape(batch * experts, -1)).reshape(batch, experts)
        return route.logits + correction


class ReciprocalExpertMoE(nn.Module):
    """
    专家选择模块，
    核心思像是先让路由进行一次投票，该投票结果是路由根据ConvNeXt的图像视觉特征和OWL的语义特征进行判断的（先验特征），
    随后每一个专家都接收图片的特征输入，并获得每一个专家对图片的特征与把握程度，并对自己给出的结果打分，
    最后的参考结合路由给出的先验特征，和专家自评的结果进行综合，并给出后验特征，重新加权得出最终结果。
    """

    def __init__(
            self,
            channels: int,
            semantic_dim: int,
            num_experts: int,
            bottleneck_ratio: int = 4,
            dropout: float = 0.0,
            advice_dim: int = 16,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList([
            ExpertConvAdapter(channels, bottleneck_ratio, dropout) for _ in range(num_experts)
        ])
        self.prior_router = PriorExpertRouter(channels, semantic_dim, num_experts, advice_dim)
        self.mixer = SpatialExpertMixer(channels)
        self.descriptor = ExpertDescriptor(channels)
        self.consensus = ExpertConsensusProjector()
        self.evaluator = ExpertSelfEvaluator(128, advice_dim)
        self.posterior_router = PosteriorExpertRouter()

    def forward(
            self,
            feature: torch.Tensor,
            semantic: SemanticEvidence,
            forced_expert: Optional[torch.Tensor] = None,
    ) -> ExpertEvidence:
        route = self.prior_router(feature, semantic)

        if forced_expert is not None:
            valid = forced_expert >= 0
            forced = F.one_hot(forced_expert.clamp_min(0), self.num_experts).to(route.weights.dtype)
            route.weights = torch.where(valid.unsqueeze(1), forced, route.weights)

        deltas = torch.stack([expert(feature) for expert in self.experts], dim=1)  # 各专家图像特征
        compatibility = self.mixer.compatibility(feature, deltas)  # 兼容分数
        description_features = self.descriptor(feature, deltas)  # 专家解释特征
        prior_consensus = self.consensus(description_features, route.weights)
        evaluation = self.evaluator(description_features, prior_consensus, route)
        posterior_weights = torch.softmax(self.posterior_router(route, evaluation), dim=1)

        if forced_expert is not None:
            posterior_weights = torch.where(valid.unsqueeze(1), route.weights, posterior_weights)  # NOQA

        final_feature, spatial_weights = self.mixer.remix(feature, deltas, compatibility, posterior_weights)
        final_consensus = self.consensus(description_features, posterior_weights)
        return ExpertEvidence(
            feature=final_feature,  # 图片混合特征
            consensus=final_consensus,
            prior_weights=route.weights,
            posterior_weights=posterior_weights,
            spatial_weights=spatial_weights,
            advice_logits=route.advice_logits,
            expert_logits=evaluation.logits,
            confidence=torch.sigmoid(evaluation.confidence_logits),
            prior_routed_logit=(route.weights * evaluation.logits).sum(dim=1, keepdim=True),
            posterior_routed_logit=(posterior_weights * evaluation.logits).sum(dim=1, keepdim=True),
        )


class EvidenceMapHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(32, channels // 4)
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(feature))


class GlobalBranch(nn.Module):
    def __init__(self, channels: int, semantic_dim: int, consensus_dim: int = 128):
        super().__init__()
        self.map_head = EvidenceMapHead(channels)
        input_dim = channels * 2 + semantic_dim + consensus_dim + 4
        self.feature_head = MLP(input_dim, 384, 192, dropout=0.10)
        self.logit_head = MLP(192, 128, 1, dropout=0.05)
        self.confidence_head = MLP(192, 96, 1, dropout=0.05)

    def forward(self, feature: torch.Tensor, context: FusionContext) -> BranchEvidence:
        attention = F.interpolate(
            self.map_head(feature), size=context.semantic.attention.shape[-2:], mode="bilinear", align_corners=False
        )
        descriptor = torch.cat([
            feature.mean((2, 3)), feature.amax((2, 3)), context.semantic.vector,
            context.expert.consensus, context.semantic.scores, context.semantic.confidence,
        ], dim=1)
        branch_feature = self.feature_head(descriptor)
        return BranchEvidence(
            logit=self.logit_head(branch_feature),
            confidence=torch.sigmoid(self.confidence_head(branch_feature)),
            feature=branch_feature,
            attention=attention,
        )


class LocalSemanticFusion(nn.Module):
    def __init__(self, channels: int, semantic_channels: int):
        super().__init__()
        self.semantic_proj = nn.Conv2d(semantic_channels, channels, 1)
        self.refine = ConvFFN(channels, expansion=2)
        self.scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, local: torch.Tensor, semantic: SemanticEvidence) -> torch.Tensor:
        feature = F.interpolate(semantic.feature, size=local.shape[-2:], mode="bilinear", align_corners=False)
        attention = F.interpolate(semantic.attention, size=local.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.semantic_proj(feature) * attention * semantic.confidence.unsqueeze(-1).unsqueeze(-1)
        return self.refine(local + torch.sigmoid(self.scale) * delta)


class LocalBranch(nn.Module):
    def __init__(self, mid_channels: int, final_channels: int, semantic_channels: int, consensus_dim: int = 128):
        super().__init__()
        local_channels = 128
        self.mid_proj = nn.Conv2d(mid_channels, local_channels, 1)
        self.final_proj = nn.Conv2d(final_channels, local_channels, 1)
        self.fusion = LocalSemanticFusion(local_channels, semantic_channels)
        self.map_head = EvidenceMapHead(local_channels)
        self.pool = SoftRegionPool()
        input_dim = local_channels * 3 + semantic_channels + consensus_dim
        self.feature_head = MLP(input_dim, 320, 192, dropout=0.10)
        self.logit_head = MLP(192, 128, 1, dropout=0.05)
        self.confidence_head = MLP(192, 96, 1, dropout=0.05)

    @staticmethod
    def _context_map(attention: torch.Tensor) -> torch.Tensor:
        expanded = F.avg_pool2d(attention, kernel_size=9, stride=1, padding=4)
        return (expanded - 0.5 * attention).clamp(0.0, 1.0)

    def forward(self, pyramid: VisualPyramid, context: FusionContext) -> BranchEvidence:
        final = F.interpolate(self.final_proj(pyramid.final),
                              size=pyramid.mid.shape[-2:],
                              mode="bilinear",
                              align_corners=False)
        local = self.mid_proj(pyramid.mid) + final
        local = self.fusion(local, context.semantic)
        attention = self.map_head(local)
        foreground = self.pool(local, context.semantic.attention, pyramid.mask)
        surrounding = self.pool(local, self._context_map(context.semantic.attention), pyramid.mask)
        topk = local.flatten(2).topk(max(1, local.shape[-2] * local.shape[-1] // 10), dim=2).values.mean(dim=2)
        descriptor = torch.cat([
            foreground, surrounding, topk, context.semantic.vector, context.expert.consensus
        ], dim=1)
        branch_feature = self.feature_head(descriptor)
        attention = F.interpolate(attention,
                                  size=context.semantic.attention.shape[-2:],
                                  mode="bilinear",
                                  align_corners=False)
        return BranchEvidence(
            logit=self.logit_head(branch_feature),
            confidence=torch.sigmoid(self.confidence_head(branch_feature)),
            feature=branch_feature,
            attention=attention,
        )


class LiteArbiter(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()

        input_dim = (
                2  # global/local logits
                + 1  # semantic logit
                + 2  # global/local confidence
                + 1  # semantic confidence
                + 3  # semantic scores
                + 3  # semantic map stats
                + num_experts * 2  # prior/posterior weights
        )

        # 三个输出分别对应：
        # global、local、semantic
        self.weight_head = MLP(
            input_dim,
            96,
            3,
            dropout=0.05,
        )

        # 避免训练开始时 OWL 权重过大。
        # 初始大致为 global 42%、local 42%、semantic 16%。
        self.branch_prior = nn.Parameter(
            torch.tensor([0.0, 0.0, -1.0])
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(1.0)
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(0.0)
        )

    @staticmethod
    def _map_stats(
            attention: torch.Tensor,
    ) -> torch.Tensor:
        flat = attention.flatten(1).clamp(
            1e-6,
            1.0 - 1e-6,
        )

        return torch.cat(
            [
                flat.mean(dim=1, keepdim=True),
                flat.amax(dim=1, keepdim=True),
                (
                    -(flat * flat.log())
                ).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )

    def forward(
            self,
            branches: BranchPair,
            context: FusionContext,
    ):
        global_branch = branches.global_branch
        local_branch = branches.local_branch
        semantic = context.semantic

        descriptor = torch.cat(
            [
                global_branch.logit,
                local_branch.logit,
                semantic.logit,

                global_branch.confidence,
                local_branch.confidence,
                semantic.confidence,

                semantic.scores,
                self._map_stats(semantic.attention),

                context.expert.prior_weights,
                context.expert.posterior_weights,
            ],
            dim=1,
        )

        weight_logits = (
                self.weight_head(descriptor)
                + self.branch_prior
        )

        weights = torch.softmax(
            weight_logits,
            dim=1,
        )

        raw_logit = (
                weights[:, 0:1] * global_branch.logit
                + weights[:, 1:2] * local_branch.logit
                + weights[:, 2:3] * semantic.logit
        )

        logit = (
                self.logit_scale.clamp(0.25, 4.0)
                * raw_logit
                + self.logit_bias
        )

        mix_map = (
                weights[:, 0:1, None, None]
                * global_branch.attention
                + weights[:, 1:2, None, None]
                * local_branch.attention
                + weights[:, 2:3, None, None]
                * semantic.attention
        )

        mix_map = mix_map / mix_map.amax(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1e-6)

        return logit, raw_logit, weights, mix_map


class FireArbiterMoELite(nn.Module):
    CHECKPOINT_FORMAT = "FireArbiter-MoE-Lite.v1"

    def __init__(
            self,
            owlvit_model_name: Optional[str] = "google/owlvit-base-patch32",
            positive_prompts: Sequence[str] = DEFAULT_OWLVIT_POSITIVE_PROMPTS,
            negative_prompts: Sequence[str] = DEFAULT_OWLVIT_NEGATIVE_PROMPTS,
            suspicious_prompts: Sequence[str] = DEFAULT_OWLVIT_SUSPICIOUS_PROMPTS,
            num_experts: int = 7,
            expert_names: Optional[Sequence[str]] = None,
            threshold: float = 0.5,
            pretrained_backbone: bool = True,
            convnext_trainable_layers: int = 1,
            owlvit_image_size: int = 384,
            owlvit_feature_dim: int = 128,
            owlvit_trainable_vision_layers: int = 2,
            owlvit_trainable_text_layers: int = 0,
            owlvit_train_class_head: bool = True,
            owlvit_train_box_head: bool = False,
            expert_adapter_bottleneck_ratio: int = 4,
            expert_adapter_dropout: float = 0.0,
            expert_advice_dim: int = 16,
    ):
        super().__init__()
        self.owlvit_model_name = owlvit_model_name
        self.positive_prompts = tuple(positive_prompts)
        self.negative_prompts = tuple(negative_prompts)
        self.suspicious_prompts = tuple(suspicious_prompts)
        self.num_experts = int(num_experts)
        self.expert_names = list(expert_names or [f"expert_{index}" for index in range(num_experts)])
        if len(self.expert_names) != self.num_experts:
            raise ValueError("expert_names length must equal num_experts.")
        self.pretrained_backbone = bool(pretrained_backbone)
        self.convnext_trainable_layers = int(convnext_trainable_layers)
        self.owlvit_image_size = int(owlvit_image_size)
        self.owlvit_feature_dim = int(owlvit_feature_dim)
        self.owlvit_trainable_vision_layers = int(owlvit_trainable_vision_layers)
        self.owlvit_trainable_text_layers = int(owlvit_trainable_text_layers)
        self.owlvit_train_class_head = bool(owlvit_train_class_head)
        self.owlvit_train_box_head = bool(owlvit_train_box_head)
        self.expert_adapter_bottleneck_ratio = int(expert_adapter_bottleneck_ratio)
        self.expert_adapter_dropout = float(expert_adapter_dropout)
        self.expert_advice_dim = int(expert_advice_dim)
        self.register_buffer("threshold", torch.tensor(float(threshold)), persistent=True)

        self.semantic_encoder = OWLSemanticEncoder(
            owlvit_model_name,
            PromptBank(self.positive_prompts, self.negative_prompts, self.suspicious_prompts),
            image_size=self.owlvit_image_size,
            feature_dim=self.owlvit_feature_dim,
            vision_tail_layers=self.owlvit_trainable_vision_layers,
            text_tail_layers=self.owlvit_trainable_text_layers,
            train_class_head=self.owlvit_train_class_head,
            train_box_head=self.owlvit_train_box_head,
        )
        self.backbone = ConvNeXtFeatureBackbone(pretrained_backbone, convnext_trainable_layers)
        self.expert_moe = ReciprocalExpertMoE(
            self.backbone.expert_channels,
            self.owlvit_feature_dim,
            self.num_experts,
            bottleneck_ratio=self.expert_adapter_bottleneck_ratio,
            dropout=self.expert_adapter_dropout,
            advice_dim=self.expert_advice_dim,
        )
        self.global_branch = GlobalBranch(self.backbone.out_channels, self.owlvit_feature_dim)
        self.local_branch = LocalBranch(
            self.backbone.mid_channels, self.backbone.out_channels, self.owlvit_feature_dim
        )
        self.arbiter = LiteArbiter(self.num_experts)

    def set_prompts(self, positive: Sequence[str], negative: Sequence[str], suspicious: Sequence[str]) -> None:
        self.positive_prompts = tuple(positive)
        self.negative_prompts = tuple(negative)
        self.suspicious_prompts = tuple(suspicious)
        self.semantic_encoder.set_prompts(positive, negative, suspicious)

    def set_threshold(self, threshold: float):
        self.threshold.fill_(float(threshold))
        return self

    def get_threshold(self) -> float:
        return float(self.threshold.detach().cpu())

    def set_train_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        def enable_semantic_new_modules() -> None:
            self.semantic_encoder.requires_grad_(True)
            if self.semantic_encoder.model is not None:
                self.semantic_encoder.model.requires_grad_(False)

        if stage == "semantic_warmup":
            enable_semantic_new_modules()
            self.expert_moe.requires_grad_(True)
            self.global_branch.requires_grad_(True)
            self.local_branch.requires_grad_(True)
            self.arbiter.requires_grad_(True)
        elif stage == "expert_specialization":
            enable_semantic_new_modules()
            self.expert_moe.requires_grad_(True)
            self.global_branch.requires_grad_(True)
            self.local_branch.requires_grad_(True)
        elif stage in {"joint_finetune", "owl_finetune"}:
            enable_semantic_new_modules()
            self.expert_moe.requires_grad_(True)
            self.global_branch.requires_grad_(True)
            self.local_branch.requires_grad_(True)
            self.arbiter.requires_grad_(True)
            self.backbone.set_trainable_layers(self.convnext_trainable_layers)
            self.semantic_encoder.set_owl_trainable(stage == "owl_finetune")
        elif stage == "calibration":
            self.arbiter.weight_head.requires_grad_(True)
            self.arbiter.logit_scale.requires_grad_(True)
            self.arbiter.logit_bias.requires_grad_(True)
            self.global_branch.confidence_head.requires_grad_(True)
            self.local_branch.confidence_head.requires_grad_(True)
            self.semantic_encoder.evidence_head.confidence_head.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def forward(
            self,
            images: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
            forced_expert: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        semantic = self.semantic_encoder(images, valid_mask)

        mid_feature, expert_input = self.backbone.forward_until(images)
        expert = self.expert_moe(expert_input, semantic, forced_expert)
        final_feature = self.backbone.forward_from(expert.feature)

        # 中层特征 最终特征 有效遮罩
        pyramid = VisualPyramid(mid_feature, final_feature, valid_mask)
        context = FusionContext(semantic, expert)

        global_result = self.global_branch(final_feature, context)
        local_result = self.local_branch(pyramid, context)
        branches = BranchPair(global_result, local_result)

        logit, raw_logit, branch_weights, mix_map = self.arbiter(branches, context)
        probability = torch.sigmoid(logit)

        return {
            "logit": logit,
            "raw_logit": raw_logit,
            "prob_fire": probability,
            "pred": (probability >= self.threshold).long(),
            "global_logit": global_result.logit,
            "local_logit": local_result.logit,
            "semantic_logit": semantic.logit,
            "global_confidence": global_result.confidence,
            "local_confidence": local_result.confidence,
            "semantic_confidence": semantic.confidence,
            "branch_weights": branch_weights,
            "attention_global": global_result.attention,
            "attention_local": local_result.attention,
            "attention_semantic": semantic.attention,
            "attention_mix": mix_map,
            "semantic_scores": semantic.scores,
            "expert_prior_weights": expert.prior_weights,
            "expert_posterior_weights": expert.posterior_weights,
            "expert_logits": expert.expert_logits,
            "expert_probs": torch.sigmoid(expert.expert_logits),
            "expert_confidence": expert.confidence,
            "prior_routed_logit": expert.prior_routed_logit,
            "posterior_routed_logit": expert.posterior_routed_logit,
        }

    def _model_config(self) -> Dict:
        return {
            "owlvit_model_name": self.owlvit_model_name,
            "positive_prompts": list(self.positive_prompts),
            "negative_prompts": list(self.negative_prompts),
            "suspicious_prompts": list(self.suspicious_prompts),
            "num_experts": self.num_experts,
            "expert_names": list(self.expert_names),
            "threshold": self.get_threshold(),
            "pretrained_backbone": self.pretrained_backbone,
            "convnext_trainable_layers": self.convnext_trainable_layers,
            "owlvit_image_size": self.owlvit_image_size,
            "owlvit_feature_dim": self.owlvit_feature_dim,
            "owlvit_trainable_vision_layers": self.owlvit_trainable_vision_layers,
            "owlvit_trainable_text_layers": self.owlvit_trainable_text_layers,
            "owlvit_train_class_head": self.owlvit_train_class_head,
            "owlvit_train_box_head": self.owlvit_train_box_head,
            "expert_adapter_bottleneck_ratio": self.expert_adapter_bottleneck_ratio,
            "expert_adapter_dropout": self.expert_adapter_dropout,
            "expert_advice_dim": self.expert_advice_dim,
        }

    def build_checkpoint(self, metadata: Optional[Dict] = None) -> Dict:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "model_config": self._model_config(),
            "state_dict": self.state_dict(),
            "threshold": self.get_threshold(),
            "metadata": dict(metadata or {}),
        }

    def export(self, path: Path | str, metadata: Optional[Dict] = None) -> None:
        torch.save(self.build_checkpoint(metadata), path)

    def load(self, path: Path | str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise RuntimeError(f"Expected {self.CHECKPOINT_FORMAT}, got {checkpoint.get('format')!r}.")
        self.load_state_dict(checkpoint["state_dict"])
        self.set_threshold(float(checkpoint.get("threshold", self.get_threshold())))
        return self


if __name__ == "__main__":
    import gc
    from pathlib import Path

    import torch

    # ---------------------------------------------------------
    # 基础设置
    # ---------------------------------------------------------
    torch.manual_seed(3407)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
            Path(__file__).resolve().parent
            / "fire_arbiter_moe_lite_test.pth"
    )

    print("=" * 80)
    print("FireArbiter-MoE-Lite 保存/读取测试")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. 创建模型
    # 根据你的训练配置修改 num_experts 等参数
    # ---------------------------------------------------------
    print("\n[1/6] 创建原始模型...")

    model = FireArbiterMoELite(
        owlvit_model_name="google/owlvit-base-patch32",
        num_experts=7,
        expert_names=[
            "expert_0",
            "expert_1",
            "expert_2",
            "expert_3",
            "expert_4",
            "expert_5",
            "expert_6",
        ],
        threshold=0.5,
        pretrained_backbone=True,
        convnext_trainable_layers=1,
        owlvit_image_size=768,
        owlvit_feature_dim=128,
        owlvit_trainable_vision_layers=2,
        owlvit_trainable_text_layers=0,
        owlvit_train_class_head=True,
        owlvit_train_box_head=False,
        expert_adapter_bottleneck_ratio=4,
        expert_adapter_dropout=0.0,
        expert_advice_dim=16,
    ).to(device)

    model.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"模型总参数量: "
        f"{total_parameters / 1e6:.3f} M"
    )
    print(
        f"当前可训练参数量: "
        f"{trainable_parameters / 1e6:.3f} M"
    )

    # ---------------------------------------------------------
    # 2. 构造测试输入
    # 模型输入采用 ImageNet 标准化
    # 图片不必是 768×768，OWL 内部会 Letterbox 到 768
    # ---------------------------------------------------------
    print("\n[2/6] 构造测试输入并执行第一次前向传播...")

    batch_size = 1
    input_height = 224
    input_width = 320

    image = torch.rand(
        batch_size,
        3,
        input_height,
        input_width,
        device=device,
    )

    imagenet_mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device,
    ).view(1, 3, 1, 1)

    imagenet_std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device,
    ).view(1, 3, 1, 1)

    image = (
                    image - imagenet_mean
            ) / imagenet_std

    valid_mask = torch.ones(
        batch_size,
        1,
        input_height,
        input_width,
        device=device,
    )

    with torch.inference_mode():
        output_before = model(
            image,
            valid_mask,
            None,
        )

    probability_before = (
        output_before["prob_fire"]
        .detach()
        .cpu()
    )

    branch_weights_before = (
        output_before["branch_weights"]
        .detach()
        .cpu()
    )

    attention_mix_before = (
        output_before["attention_mix"]
        .detach()
        .cpu()
    )

    print(
        "保存前火灾概率:",
        probability_before.flatten().tolist(),
    )
    print(
        "保存前分支权重:",
        branch_weights_before.tolist(),
    )
    print(
        "Mix Attention形状:",
        tuple(attention_mix_before.shape),
    )

    # ---------------------------------------------------------
    # 3. 保存模型
    # 保存内容包括：
    # format、model_config、state_dict、threshold、metadata
    # ---------------------------------------------------------
    print("\n[3/6] 保存模型...")

    model.export(
        checkpoint_path,
        metadata={
            "purpose": "save_load_test",
            "input_shape": list(image.shape),
            "pytorch_version": torch.__version__,
        },
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"模型保存失败：{checkpoint_path}"
        )

    checkpoint_size_mb = (
            checkpoint_path.stat().st_size
            / 1024
            / 1024
    )

    print(
        f"模型已保存: {checkpoint_path}"
    )
    print(
        f"Checkpoint大小: "
        f"{checkpoint_size_mb:.2f} MB"
    )

    # ---------------------------------------------------------
    # 4. 删除原模型，确认不是继续使用内存中的旧模型
    # ---------------------------------------------------------
    print("\n[4/6] 删除原模型并释放显存...")

    del output_before
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 5. 读取checkpoint中的模型配置并重新创建模型
    # ---------------------------------------------------------
    print("\n[5/6] 从Checkpoint重建并读取模型...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_format = checkpoint.get("format")

    if (
            checkpoint_format
            != FireArbiterMoELite.CHECKPOINT_FORMAT
    ):
        raise RuntimeError(
            "Checkpoint格式错误："
            f"期望 {FireArbiterMoELite.CHECKPOINT_FORMAT}，"
            f"实际为 {checkpoint_format!r}"
        )

    model_config = dict(
        checkpoint["model_config"]
    )

    print(
        "Checkpoint格式:",
        checkpoint_format,
    )
    print(
        "Checkpoint阈值:",
        checkpoint.get("threshold"),
    )
    print(
        "Checkpoint元数据:",
        checkpoint.get("metadata", {}),
    )

    # state_dict随后会覆盖全部ConvNeXt权重，
    # 因此重建时不需要再次加载ConvNeXt预训练权重。
    model_config["pretrained_backbone"] = False

    loaded_model = FireArbiterMoELite(
        **model_config
    ).to(device)

    loaded_model.load(
        checkpoint_path,
        map_location=device,
    )

    loaded_model.eval()

    print(
        "读取后的模型阈值:",
        loaded_model.get_threshold(),
    )

    # ---------------------------------------------------------
    # 6. 使用读取后的模型再次推理并比较结果
    # ---------------------------------------------------------
    print("\n[6/6] 执行读取后的模型前向传播...")

    with torch.inference_mode():
        output_after = loaded_model(
            image,
            valid_mask,
            None,
        )

    probability_after = (
        output_after["prob_fire"]
        .detach()
        .cpu()
    )

    branch_weights_after = (
        output_after["branch_weights"]
        .detach()
        .cpu()
    )

    attention_mix_after = (
        output_after["attention_mix"]
        .detach()
        .cpu()
    )

    probability_difference = (
            probability_before
            - probability_after
    ).abs().max().item()

    branch_difference = (
            branch_weights_before
            - branch_weights_after
    ).abs().max().item()

    attention_difference = (
            attention_mix_before
            - attention_mix_after
    ).abs().max().item()

    print(
        "读取后火灾概率:",
        probability_after.flatten().tolist(),
    )
    print(
        "读取后分支权重:",
        branch_weights_after.tolist(),
    )

    print("\n" + "-" * 80)
    print(
        "火灾概率最大误差:",
        f"{probability_difference:.10f}",
    )
    print(
        "分支权重最大误差:",
        f"{branch_difference:.10f}",
    )
    print(
        "Mix Attention最大误差:",
        f"{attention_difference:.10f}",
    )

    probability_ok = torch.allclose(
        probability_before,
        probability_after,
        atol=1e-5,
        rtol=1e-4,
    )

    branch_ok = torch.allclose(
        branch_weights_before,
        branch_weights_after,
        atol=1e-5,
        rtol=1e-4,
    )

    attention_ok = torch.allclose(
        attention_mix_before,
        attention_mix_after,
        atol=1e-5,
        rtol=1e-4,
    )

    if not probability_ok:
        raise RuntimeError(
            "模型读取失败：前后火灾概率不一致。"
        )

    if not branch_ok:
        raise RuntimeError(
            "模型读取失败：前后分支权重不一致。"
        )

    if not attention_ok:
        raise RuntimeError(
            "模型读取失败：前后Attention Map不一致。"
        )

    print("-" * 80)
    print("测试通过：模型保存、读取和前向结果一致。")
    print("=" * 80)
