import argparse
from pathlib import Path
import torch
import torch.nn as nn
import onnx
from torchvision.models import resnet50

class InsightFaceBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        # Using standard resnet50 from torchvision to avoid torch.hub errors
        self.model = resnet50(weights=None)
        # InsightFace models output 512-d embeddings
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.model(x)

def load_weights(model, weights_path):
    if not weights_path.exists():
        print(f"[convert] Warning: No weights found at {weights_path}. Exporting dummy.")
        return
    
    # If the file is actually an ONNX file, we can't 'load' it into a Torch model
    if weights_path.suffix == '.onnx':
        print(f"[convert] ERROR: {weights_path} is an ONNX file. You need a .pth file to use this script.")
        return

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Strip 'module.' prefix if saved with DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"[convert] Weights loaded from {weights_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, default=Path("model_repository/fr_model/1/model.onnx"))
    args = parser.parse_args()

    # 1. Initialize
    model = InsightFaceBackbone(embedding_dim=512)
    load_weights(model, args.weights_path)
    model.eval()

    # 2. Define names and Dynamic Axes
    # This tells ONNX that the 0th dimension (batch) can change at runtime
    input_names = ["input.1"]
    output_names = ["683"]
    dynamic_axes = {
        "input.1": {0: "batch_size"},
        "683": {0: "batch_size"}
    }
    
    # 3. Create dummy input with shape [1, 3, 112, 112]
    dummy_input = torch.randn(1, 3, 112, 112)

    # 4. Export with Opset 15 and Dynamic Axes
    print(f"[convert] Exporting to {args.onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=15,
        do_constant_folding=True
    )

    # 5. FORCE THE IR VERSION DOWN (Crucial for Triton 23.10)
    model_proto = onnx.load(str(args.onnx_path))
    model_proto.ir_version = 8 
    onnx.save(model_proto, str(args.onnx_path))

    print(f"[convert] Successfully exported ONNX (IR v8) with dynamic batching to {args.onnx_path}")
    
if __name__ == "__main__":
    main()