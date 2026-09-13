import torch
import torch.nn as nn
from torchvision.models import resnet18


MODEL_PATH = "checkpoints/resnet18_calm_aggressive_best.pt"
OUTPUT_PATH = "checkpoints/caninecue_resnet18.onnx"


# Recreate the same model architecture used during training
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)


# Load the trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")


# Dummy input matching the model's expected input
dummy_input = torch.randn(1, 3, 224, 224)


# Export as a single self-contained ONNX file
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    external_data=False,
)


print("ONNX model created successfully.")
print(f"Saved to: {OUTPUT_PATH}")