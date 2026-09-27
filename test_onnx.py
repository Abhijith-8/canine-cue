import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from torchvision.models import resnet18


MODEL_PATH = "checkpoints/resnet18_calm_aggressive_best.pt"
ONNX_PATH = "checkpoints/caninecue_resnet18.onnx"


# -----------------------------
# Load PyTorch model
# -----------------------------
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(state_dict)
model.eval()


# -----------------------------
# Create the same test input
# -----------------------------
torch.manual_seed(42)

test_input = torch.randn(1, 3, 224, 224)


# -----------------------------
# PyTorch prediction
# -----------------------------
with torch.no_grad():
    pytorch_output = model(test_input)
    pytorch_probability = torch.sigmoid(pytorch_output).item()


# -----------------------------
# ONNX prediction
# -----------------------------
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"],
)

onnx_output = session.run(
    ["output"],
    {"input": test_input.numpy()},
)[0]

onnx_probability = 1 / (1 + np.exp(-onnx_output[0][0]))


# -----------------------------
# Compare
# -----------------------------
print()
print("===== MODEL COMPARISON =====")
print(f"PyTorch aggression probability : {pytorch_probability:.6f}")
print(f"ONNX aggression probability    : {onnx_probability:.6f}")
print(f"Difference                     : {abs(pytorch_probability - onnx_probability):.6f}")

if abs(pytorch_probability - onnx_probability) < 0.0001:
    print("RESULT: Models match successfully.")
else:
    print("RESULT: Models have a noticeable difference.")