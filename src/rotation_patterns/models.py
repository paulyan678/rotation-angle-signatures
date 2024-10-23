"""Encoders, contrastive objectives, downstream probe, and U-Net."""

from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = width * 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image).flatten(1)


class TimmEncoder(nn.Module):
    def __init__(self, name: str, in_channels: int, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("model construction requires timm") from exc
        self.model = timm.create_model(
            name, pretrained=pretrained, in_chans=in_channels, num_classes=0, global_pool="avg"
        )
        self.feature_dim = int(self.model.num_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if output.ndim > 2:
            output = output.mean(dim=tuple(range(2, output.ndim)))
        return output


def build_encoder(
    name: str, in_channels: int, imagenet_initialized: bool = True
) -> nn.Module:
    if name == "tiny_cnn":
        return TinyEncoder(in_channels, width=8)
    if name == "tiny_cnn_wide":
        return TinyEncoder(in_channels, width=12)
    return TimmEncoder(name, in_channels, pretrained=imagenet_initialized)


class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, projection_dim: int, temperature: float):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(encoder.feature_dim, hidden_dim, projection_dim)
        self.temperature = temperature

    def encode_project(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(image)), dim=1)

    def loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_projection = self.encode_project(first)
        second_projection = self.encode_project(second)
        projections = torch.cat([first_projection, second_projection], dim=0)
        logits = projections @ projections.T / self.temperature
        batch_size = first.shape[0]
        diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, float("-inf"))
        targets = torch.arange(2 * batch_size, device=logits.device)
        targets = (targets + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, targets)


class MoCoV2(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        projection_dim: int,
        queue_size: int,
        momentum: float,
        temperature: float,
    ):
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.projector_q = ProjectionHead(encoder.feature_dim, hidden_dim, projection_dim)
        self.projector_k = copy.deepcopy(self.projector_q)
        for parameter in (*self.encoder_k.parameters(), *self.projector_k.parameters()):
            parameter.requires_grad = False
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("queue", F.normalize(torch.randn(projection_dim, queue_size), dim=0))
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self) -> None:
        for query, key in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            key.data.mul_(self.momentum).add_(query.data, alpha=1 - self.momentum)
        for query, key in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            key.data.mul_(self.momentum).add_(query.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        keys = keys.detach()
        if keys.shape[0] >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size :].T)
            self.queue_pointer.zero_()
            return
        pointer = int(self.queue_pointer.item())
        first = min(self.queue_size - pointer, keys.shape[0])
        self.queue[:, pointer : pointer + first] = keys[:first].T
        remaining = keys.shape[0] - first
        if remaining:
            self.queue[:, :remaining] = keys[first:].T
        self.queue_pointer[0] = (pointer + keys.shape[0]) % self.queue_size

    def loss(self, query_image: torch.Tensor, key_image: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.projector_q(self.encoder_q(query_image)), dim=1)
        with torch.no_grad():
            self._momentum_update()
            key = F.normalize(self.projector_k(self.encoder_k(key_image)), dim=1)
        positive = torch.einsum("nc,nc->n", query, key).unsqueeze(1)
        # Clone because the queue is updated before backward; a detached view
        # still shares storage and would invalidate autograd's saved operand.
        negative = torch.einsum("nc,ck->nk", query, self.queue.detach().clone())
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self._enqueue(key)
        return loss


def build_contrastive_model(
    method: str,
    encoder_name: str,
    in_channels: int,
    settings: dict,
) -> nn.Module:
    encoder = build_encoder(
        encoder_name, in_channels, bool(settings.get("imagenet_initialized", True))
    )
    common = {
        "hidden_dim": int(settings["projection_hidden_dim"]),
        "projection_dim": int(settings["projection_dim"]),
        "temperature": float(settings["temperature"]),
    }
    if method == "simclr":
        return SimCLR(encoder, **common)
    if method == "mocov2":
        return MoCoV2(
            encoder,
            **common,
            queue_size=int(settings["queue_size"]),
            momentum=float(settings["moco_momentum"]),
        )
    raise ValueError(f"unsupported contrastive method: {method}")


def query_encoder(model: nn.Module) -> nn.Module:
    if isinstance(model, SimCLR):
        return model.encoder
    if isinstance(model, MoCoV2):
        return model.encoder_q
    raise TypeError(f"unsupported contrastive model: {type(model).__name__}")


class SixLayerProbe(nn.Module):
    """Six linear layers with ReLU after the first five, as described in Appendix B."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        hidden = [int(value) for value in hidden_dims]
        if len(hidden) != 5:
            raise ValueError("the paper's six-layer MLP requires exactly five hidden dimensions")
        dimensions = [input_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class TinyFeaturePyramid(nn.Module):
    def __init__(self, in_channels: int, width: int = 8):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 8]
        blocks = []
        source = in_channels
        for target in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(source, target, 3, stride=2, padding=1),
                    nn.BatchNorm2d(target),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(target, target, 3, padding=1),
                    nn.BatchNorm2d(target),
                    nn.ReLU(inplace=True),
                )
            )
            source = target
        self.blocks = nn.ModuleList(blocks)
        self.channels = channels

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for block in self.blocks:
            image = block(image)
            outputs.append(image)
        return outputs


class TimmFeaturePyramid(nn.Module):
    def __init__(self, name: str, in_channels: int, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("model construction requires timm") from exc
        if name.startswith("vit_"):
            # Four evenly spaced transformer blocks. They share a patch-grid
            # resolution; the decoder still fuses shallow-to-deep features.
            out_indices = (2, 5, 8, 11)
        elif name.startswith(("resnet", "wide_resnet")):
            # Skip the stem and retain residual stages 1 through 4.
            out_indices = (1, 2, 3, 4)
        else:
            out_indices = (0, 1, 2, 3)
        try:
            self.model = timm.create_model(
                name,
                pretrained=pretrained,
                in_chans=in_channels,
                features_only=True,
                out_indices=out_indices,
            )
        except Exception as exc:
            raise ValueError(
                f"encoder {name!r} does not expose four timm feature stages required by U-Net"
            ) from exc
        self.channels = list(self.model.feature_info.channels())
        if len(self.channels) != 4:
            raise ValueError(f"expected four feature stages for {name}, got {self.channels}")

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs = list(self.model(image))
        normalized = []
        for value, channels in zip(outputs, self.channels):
            if value.ndim != 4:
                raise ValueError(f"feature tensor must be 4D, got {value.shape}")
            if value.shape[1] != channels and value.shape[-1] == channels:
                value = value.permute(0, 3, 1, 2).contiguous()
            normalized.append(value)
        return normalized


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolutions(torch.cat([value, skip], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        classes: int,
        imagenet_initialized: bool = True,
    ):
        super().__init__()
        if encoder_name == "tiny_cnn":
            self.encoder = TinyFeaturePyramid(in_channels)
        else:
            self.encoder = TimmFeaturePyramid(
                encoder_name, in_channels, pretrained=imagenet_initialized
            )
        channels = self.encoder.channels
        decoder_channels = [min(256, channels[2]), min(128, channels[1]), min(64, channels[0])]
        self.dec3 = DecoderBlock(channels[3], channels[2], decoder_channels[0])
        self.dec2 = DecoderBlock(decoder_channels[0], channels[1], decoder_channels[1])
        self.dec1 = DecoderBlock(decoder_channels[1], channels[0], decoder_channels[2])
        self.head = nn.Conv2d(decoder_channels[2], classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stage1, stage2, stage3, stage4 = self.encoder(image)
        value = self.dec3(stage4, stage3)
        value = self.dec2(value, stage2)
        value = self.dec1(value, stage1)
        value = self.head(value)
        return F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if logits.shape[1] == 1:
        probability = torch.sigmoid(logits)
        target_value = target.float()
        if target_value.ndim == 3:
            target_value = target_value.unsqueeze(1)
    else:
        probability = torch.softmax(logits, dim=1)
        target_value = F.one_hot(target.long(), logits.shape[1]).permute(0, 3, 1, 2).float()
    dimensions = (0, 2, 3)
    intersection = (probability * target_value).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target_value.sum(dim=dimensions)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice.mean()


@torch.no_grad()
def macro_dice(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    if logits.shape[1] == 1:
        prediction = (torch.sigmoid(logits) >= 0.5).float()
        target_value = target.float()
        if target_value.ndim == 3:
            target_value = target_value.unsqueeze(1)
        prediction = torch.cat([1 - prediction, prediction], dim=1)
        target_value = torch.cat([1 - target_value, target_value], dim=1)
    else:
        prediction = F.one_hot(logits.argmax(1), logits.shape[1]).permute(0, 3, 1, 2).float()
        target_value = F.one_hot(target.long(), logits.shape[1]).permute(0, 3, 1, 2).float()
    dimensions = (0, 2, 3)
    intersection = (prediction * target_value).sum(dim=dimensions)
    denominator = prediction.sum(dim=dimensions) + target_value.sum(dim=dimensions)
    return float(((2 * intersection + epsilon) / (denominator + epsilon)).mean().item())
