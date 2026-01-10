"""
Convert BiSeNetV2 PyTorch model to ONNX format for optimized inference.
"""

import torch
import numpy as np
from bisenetv2_model import BiSeNetV2


def convert_to_onnx(
    model_path='pretrained_models/bisenetv2_cityscapes.pth',
    output_path='pretrained_models/bisenetv2.onnx',
    input_size=(512, 1024),
    opset_version=12
):
    """Convert PyTorch model to ONNX format."""
    
    print("Loading PyTorch model...")
    model = BiSeNetV2(n_classes=19, aux_mode='eval')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    print(f"Input shape: {dummy_input.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to: {output_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")
    
    # Print model info
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")


if __name__ == '__main__':
    convert_to_onnx()
