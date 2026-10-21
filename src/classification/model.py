import torch
import torchvision.models as models


def load_model(checkpoint_path: str, device: torch.device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"[Classifier] Loaded from {checkpoint_path} on {device}")
    return model