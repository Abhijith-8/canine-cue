import torch
import torchvision.models as models
import argparse
import os

def export_onnx(checkpoint_path: str, output_path: str):
    device = torch.device("cpu")  # export on CPU for compatibility

    # Build same architecture used in training
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # binary: 1 output
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle if checkpoint is wrapped in a dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Exported to: {output_path}")

    # Quick sanity check
    import onnx
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt file")
    parser.add_argument("--output", default="assets/caninecue_resnet18.onnx",
                        help="Output ONNX path")
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)