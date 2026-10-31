"""Image-only neural networks and document datasets implemented with PyTorch."""

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    """Expose full-page grayscale caches with optional mild training augmentation."""

    def __init__(self, cache_path, indices, labels=None, model_name="small_cnn", augment=False, geometry=None):
        """Keep a memory-mapped cache and explicit original metadata row indices."""
        self.cache_path = str(cache_path)
        self.images = None
        self.indices = np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
        self.model_name = model_name
        self.augment = augment
        self.geometry = geometry

    def __len__(self):
        """Return the number of selected metadata rows."""
        return len(self.indices)

    def __getitem__(self, item):
        """Load one page, optionally perturb contrast, and return its label/index."""
        if self.images is None:
            self.images = np.load(self.cache_path, mmap_mode="r")
        index = self.indices[item]
        pixels = self.images[index].copy()
        if self.augment and self.geometry:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import affine
            size = pixels.shape[0]
            angle = (2 * torch.rand(1).item() - 1) * self.geometry["degrees"]
            translation = [round((2 * torch.rand(1).item() - 1) * self.geometry["translate"] * size)
                           for _ in range(2)]
            scale = 1 + (2 * torch.rand(1).item() - 1) * self.geometry["scale"]
            pixels = np.asarray(affine(Image.fromarray(pixels), angle=angle, translate=translation,
                                       scale=scale, shear=0, interpolation=InterpolationMode.BILINEAR,
                                       fill=255)).copy()
        image = torch.from_numpy(pixels).float().unsqueeze(0) / 255.0
        if self.augment:
            contrast = 0.9 + 0.2 * torch.rand(1).item()
            image = ((image - 0.5) * contrast + 0.5).clamp(0, 1)
        if self.model_name in {"efficientnet_b0", "resnet50", "convnext_tiny"}:
            image = image.repeat(3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
        else:
            image = (image - 0.5) / 0.5
        return image, int(self.labels[index]) if self.labels is not None else int(index)

    def __getstate__(self):
        """Reopen the cache in spawned Windows workers instead of copying arrays."""
        state = self.__dict__.copy()
        state["images"] = None
        return state


class SmallDocumentCNN(nn.Module):
    """Compact CNN preserving a 4x4 spatial grid for document layout recognition."""

    def __init__(self, num_classes=16):
        """Build four convolution blocks and a layout-aware classification head."""
        super().__init__()
        layers = []
        channels = 1
        for output_channels in (16, 32, 64, 96):
            layers.extend([nn.Conv2d(channels, output_channels, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)])
            channels = output_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * 4 * 4, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, images):
        """Return 16 unnormalized logits for each document image."""
        return self.classifier(self.features(images))


def build_model(name="small_cnn", pretrained=False):
    """Create a small CNN or an ImageNet backbone with a 16-class output head."""
    if name == "small_cnn":
        if pretrained:
            raise ValueError("No pretrained weights are available for small_cnn")
        return SmallDocumentCNN()
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 16)
        return model
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 16)
        return model
    if name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 16)
        return model
    raise ValueError(f"Unknown model: {name}")
