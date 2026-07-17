import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvNeXtAttentionFireClassifier(nn.Module):
    """
    ConvNeXt-Tiny + Attention Pooling 火焰二分类模型。

    设计思路：
    1. ConvNeXt-Tiny 负责提取整图高级特征。
    2. 不直接使用普通全局平均池化。
    3. 使用 Attention Pooling，让模型学习哪些空间区域更重要。
    4. 同时融合 attention pooling、average pooling、max pooling。
    5. 输出一个 logit，训练时使用 BCEWithLogitsLoss。

    可选：
        return_attention=True 时返回：
        logit, attention_map
    """

    def __init__(
            self,
            dropout=0.35,
            use_pretrained=True,
            max_pool_scale=None
    ):
        super().__init__()

        if use_pretrained:
            # 加载 ImageNet 预训练 ConvNeXt-Tiny。
            convnext = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        else:
            convnext = models.convnext_tiny()

        self.max_pool_scale = 1.0 if max_pool_scale is None else float(max_pool_scale)

        # ConvNeXt 的特征提取部分。
        # 输入 384×384 时，输出大约是 [B, 768, 12, 12]。
        self.backbone = convnext.features

        # ConvNeXt-Tiny 最后一层通道数是 768。
        self.feature_dim = 768

        # 空间 attention 模块。
        # 输入 feature map: [B, 768, H, W]
        # 输出 attention logits: [B, 1, H, W]
        self.attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(256, 1, kernel_size=1)
        )

        # 融合三种全局特征：
        # 1. attention pooling
        # 2. average pooling
        # 3. max pooling
        fusion_dim = self.feature_dim * 3

        # 二分类分类头。
        # 不加 Sigmoid，因为训练时用 BCEWithLogitsLoss。
        self.fc = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(p=dropout),

            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.35),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(p=0.25),

            nn.Linear(128, 1)
        )

    def forward_features(self, x):
        """
        ConvNeXt 特征提取。
        输入: [B, 3, H, W]
        输出: [B, 768, H/32, W/32]
        """
        x = self.backbone(x)
        return x

    def attention_pooling(self, feature_map):
        """
        对 feature map 做 attention pooling。

        feature_map: [B, C, H, W]

        返回：
        attention_feature: [B, C]
        """
        B, C, H, W = feature_map.shape

        # [B, 1, H, W] -> [B, H*W]
        attn_logits = self.attention(feature_map).flatten(1)

        # 在空间维度做 softmax，让模型学习每个位置的重要性。
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)

        # [B, C, H, W] -> [B, H*W, C]
        tokens = feature_map.flatten(2).transpose(1, 2)

        # 加权求和，得到 attention pooling 特征。
        attention_feature = (tokens * attn_weights).sum(dim=1)

        return attention_feature

    def get_attention_map(self, feature_map):
        """
        获取空间注意力图（用于可视化）。

        feature_map: [B, C, H, W]

        返回：
        attention_map: [B, 1, H, W]，值在 [0, 1] 之间
        """
        attn_logits = self.attention(feature_map)
        B, _, H, W = attn_logits.shape

        # 在空间维度做 softmax，得到归一化的注意力权重
        attention_map = torch.softmax(
            attn_logits.flatten(2),
            dim=-1
        ).reshape(B, 1, H, W)

        return attention_map

    def forward(self, x, return_attention=False):
        """
        前向传播。

        参数：
            x: 输入图像 [B, 3, H, W]
            return_attention: 是否返回注意力图

        返回：
            logit: [B, 1]
            如果 return_attention=True，额外返回 attention_map: [B, 1, H, W]
        """
        feature_map = self.forward_features(x)

        # 1. attention pooling：模型学习重点区域。
        attn_feature = self.attention_pooling(feature_map)

        # 2. average pooling：保留整图平均语义。
        avg_feature = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)

        # 3. max pooling：保留最强局部响应，例如小火焰区域。
        max_feature = F.adaptive_max_pool2d(feature_map, output_size=1).flatten(1)
        max_feature = max_feature * self.max_pool_scale

        # 三种特征融合。
        fused_feature = torch.cat(
            [attn_feature, avg_feature, max_feature],
            dim=1
        )

        logit = self.fc(fused_feature)

        if return_attention:
            attention_map = self.get_attention_map(feature_map)
            return logit, attention_map

        return logit


class GlobalLocalFireClassifier(nn.Module):
    """
    Global + Local Tile 火焰分类模型。

    输入：
        global_image: [B, 3, H, W]
            整图缩放图，用于保留全局场景信息。

        tiles: [B, N, 3, H, W]
            从原图高分辨率上裁出的局部 tile。
            小火苗在 tile 中不会被整图缩放压没。

        tile_mask: [B, N]
            有效 tile 标记。padding tile 为 False。

    输出：
        logit:
            整图最终 fire logit。

        global_logit:
            整图分支 fire logit。

        local_logit:
            top-k tile 聚合后的局部 fire logit。

        tile_logits:
            每个 tile 的 fire logit。
    """

    def __init__(
            self,
            dropout=0.35,
            use_pretrained=True,
            max_pool_scale=0.25,
            topk=3,
            global_weight=0.4,
            local_weight=0.6
    ):
        super().__init__()

        self.topk = int(topk)
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)

        # 使用同一个 ConvNeXt attention classifier 处理整图和 tile。
        # 这样参数量较小，也能让 tile 训练反向改善共享特征。
        self.shared_classifier = ConvNeXtAttentionFireClassifier(
            dropout=dropout,
            use_pretrained=use_pretrained,
            max_pool_scale=max_pool_scale
        )

    def _masked_topk_mean(self, tile_logits, tile_mask):
        """
        tile_logits: [B, N]
        tile_mask: [B, N]
        """
        B, N = tile_logits.shape
        local_logits = []

        for b in range(B):
            valid = tile_mask[b].bool()
            vals = tile_logits[b][valid]

            if vals.numel() == 0:
                vals = tile_logits[b][:1]

            k = min(self.topk, vals.numel())
            top_vals = torch.topk(vals, k=k, largest=True).values
            local_logits.append(top_vals.mean())

        return torch.stack(local_logits, dim=0)

    def forward(self, global_image, tiles, tile_mask=None):
        """
        global_image: [B, 3, H, W]
        tiles: [B, N, 3, H, W]
        tile_mask: [B, N]
        """
        B, N, C, H, W = tiles.shape

        if tile_mask is None:
            tile_mask = torch.ones(
                B,
                N,
                dtype=torch.bool,
                device=tiles.device
            )
        else:
            tile_mask = tile_mask.bool()

        # 整图分支
        global_logit = self.shared_classifier(global_image).squeeze(1)

        # tile 分支
        flat_tiles = tiles.reshape(B * N, C, H, W)
        flat_tile_logits = self.shared_classifier(flat_tiles).squeeze(1)
        tile_logits = flat_tile_logits.reshape(B, N)

        # top-k 局部聚合，避免单个亮点一票决定
        local_logit = self._masked_topk_mean(
            tile_logits=tile_logits,
            tile_mask=tile_mask
        )

        # 最终融合
        final_logit = (
                self.global_weight * global_logit +
                self.local_weight * local_logit
        )

        return {
            "logit": final_logit.unsqueeze(1),
            "global_logit": global_logit.unsqueeze(1),
            "local_logit": local_logit.unsqueeze(1),
            "tile_logits": tile_logits
        }


def build_global_local_fire_model(
        finetune_mode="layer4_last",
        use_pretrained=True,
        dropout=0.35,
        max_pool_scale=0.25,
        topk=3,
        global_weight=0.4,
        local_weight=0.6
):
    model = GlobalLocalFireClassifier(
        dropout=dropout,
        use_pretrained=use_pretrained,
        max_pool_scale=max_pool_scale,
        topk=topk,
        global_weight=global_weight,
        local_weight=local_weight
    )

    model.shared_classifier = set_finetune_mode(
        model.shared_classifier,
        finetune_mode=finetune_mode
    )

    return model


def set_finetune_mode(model, finetune_mode="layer4_last"):
    """
    控制 ConvNeXt 微调范围。

    finetune_mode:
        head:
            只训练 attention + fc
        layer4_last:
            训练 attention + fc + ConvNeXt 最后一层 block
        layer4_all:
            训练 attention + fc + ConvNeXt 最后 stage
        all:
            全模型微调
    """

    for param in model.parameters():
        param.requires_grad = False

    for param in model.attention.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    if finetune_mode == "head":
        pass

    elif finetune_mode == "layer4_last":
        for param in model.backbone[7][-1].parameters():
            param.requires_grad = True

    elif finetune_mode == "layer4_all":
        for param in model.backbone[7].parameters():
            param.requires_grad = True

    elif finetune_mode == "all":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    return model


def build_fire_model(
        finetune_mode="layer4_last",
        use_pretrained=True,
        dropout=0.35,
        max_pool_scale=0.5
):
    model = ConvNeXtAttentionFireClassifier(
        dropout=dropout,
        use_pretrained=use_pretrained,
        max_pool_scale=max_pool_scale
    )

    model = set_finetune_mode(
        model=model,
        finetune_mode=finetune_mode
    )

    return model


def load_fire_model_from_checkpoint(
        checkpoint_path,
        device,
        finetune_mode="layer4_last",
        dropout=0.35,
        max_pool_scale=0.5,
        strict=True
):
    """
    推理 / Attention 使用。
    注意：加载已训练 checkpoint 时 use_pretrained=False 即可。
    """

    model = build_fire_model(
        finetune_mode=finetune_mode,
        use_pretrained=False,
        dropout=dropout,
        max_pool_scale=max_pool_scale
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=strict
    )

    model.to(device)
    model.eval()

    return model, checkpoint


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_ratio": trainable_params / max(total_params, 1)
    }
