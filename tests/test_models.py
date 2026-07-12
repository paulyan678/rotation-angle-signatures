import torch

from rotation_patterns.models import (
    MoCoV2,
    SimCLR,
    SixLayerProbe,
    UNet,
    build_contrastive_model,
    macro_dice,
    soft_dice_loss,
)


SETTINGS = {
    "imagenet_initialized": False,
    "projection_hidden_dim": 16,
    "projection_dim": 8,
    "temperature": 0.2,
    "queue_size": 7,
    "moco_momentum": 0.99,
}


def test_contrastive_losses_are_finite_and_moco_wraps_queue() -> None:
    first = torch.randn(3, 3, 16, 16)
    second = torch.randn(3, 3, 16, 16)
    simclr = build_contrastive_model("simclr", "tiny_cnn", 3, SETTINGS)
    moco = build_contrastive_model("mocov2", "tiny_cnn", 3, SETTINGS)
    assert isinstance(simclr, SimCLR)
    assert isinstance(moco, MoCoV2)
    assert torch.isfinite(simclr.loss(first, second))
    first_loss = moco.loss(first, second)
    assert torch.isfinite(first_loss)
    first_loss.backward()
    assert int(moco.queue_pointer.item()) == 3
    assert torch.isfinite(moco.loss(first, second))
    assert int(moco.queue_pointer.item()) == 6


def test_probe_has_six_linear_layers() -> None:
    probe = SixLayerProbe(20, [16, 12, 10, 8, 4])
    assert sum(isinstance(layer, torch.nn.Linear) for layer in probe.modules()) == 6
    assert probe(torch.randn(5, 20)).shape == (5,)


def test_tiny_unet_and_dice() -> None:
    model = UNet("tiny_cnn", in_channels=3, classes=1, imagenet_initialized=False)
    image = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    logits = model(image)
    assert logits.shape == target.shape
    assert torch.isfinite(soft_dice_loss(logits, target))
    assert 0 <= macro_dice(logits, target) <= 1
