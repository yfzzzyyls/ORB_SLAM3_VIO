#!/usr/bin/env python3
"""
Export RT-MonoDepth-S model to TensorRT FP16 format for embedded deployment.
Optimizes the model for inference on NVIDIA GPUs with reduced precision.
"""

import os
import sys
import torch
import onnx
import tensorrt as trt
import numpy as np
from pathlib import Path
import argparse
import time
from typing import Tuple

# Add project to path
sys.path.append(str(Path(__file__).parent))

from model_rtmonodepth import RTMonoDepthS


class TensorRTBuilder:
    """Build TensorRT engine from ONNX model."""
    
    def __init__(self, onnx_path: str, engine_path: str, 
                 fp16: bool = True, max_batch_size: int = 1,
                 input_shape: Tuple[int, int, int, int] = (1, 3, 1408, 1408)):
        """
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            fp16: Use FP16 precision
            max_batch_size: Maximum batch size
            input_shape: Input shape (batch, channels, height, width)
        """
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        
        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def build_engine(self):
        """Build TensorRT engine from ONNX model."""
        print(f"Building TensorRT engine from {self.onnx_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"FP16 mode: {self.fp16}")
        
        # Create builder
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        
        # Set max workspace size (8GB)
        config.max_workspace_size = 8 * (1 << 30)
        
        # Enable FP16 if requested
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        
        # Parse ONNX
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        print("ONNX parsing successful")
        
        # Set optimization profile for dynamic batch size
        profile = builder.create_optimization_profile()
        
        # Input tensor name
        input_name = network.get_input(0).name
        min_shape = (1, self.input_shape[1], self.input_shape[2], self.input_shape[3])
        opt_shape = self.input_shape
        max_shape = (self.max_batch_size, self.input_shape[1], 
                    self.input_shape[2], self.input_shape[3])
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        print("Building TensorRT engine... (this may take a few minutes)")
        start_time = time.time()
        
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("ERROR: Failed to build engine")
            return None
        
        build_time = time.time() - start_time
        print(f"Engine built successfully in {build_time:.1f} seconds")
        
        # Save engine
        with open(self.engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"Engine saved to {self.engine_path}")
        
        return engine
    
    def test_engine(self, engine):
        """Test TensorRT engine with dummy input."""
        print("\nTesting TensorRT engine...")
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        batch_size = self.input_shape[0]
        input_shape = self.input_shape
        output_shape = (batch_size, 1, self.input_shape[2], self.input_shape[3])
        
        # Host buffers
        h_input = np.random.randn(*input_shape).astype(np.float32)
        h_output = np.zeros(output_shape, dtype=np.float32)
        
        # Device buffers
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # Transfer input
        cuda.memcpy_htod(d_input, h_input)
        
        # Set binding shapes
        context.set_binding_shape(0, input_shape)
        
        # Execute
        start_time = time.time()
        context.execute_v2(bindings=[int(d_input), int(d_output)])
        cuda.Context.synchronize()
        inference_time = (time.time() - start_time) * 1000
        
        # Transfer output
        cuda.memcpy_dtoh(h_output, d_output)
        
        print(f"Inference successful!")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {h_output.shape}")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Output range: [{h_output.min():.3f}, {h_output.max():.3f}]")


def export_to_onnx(model, onnx_path: str, input_shape: Tuple[int, int, int, int]):
    """Export PyTorch model to ONNX format."""
    print(f"Exporting model to ONNX: {onnx_path}")
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'depth': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("ONNX export successful")
    
    # Print model info
    print(f"\nONNX Model Info:")
    print(f"Input: {onnx_model.graph.input[0].name} - "
          f"{[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"Output: {onnx_model.graph.output[0].name} - "
          f"{[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")


def main():
    parser = argparse.ArgumentParser(description='Export RT-MonoDepth-S to TensorRT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./deployment',
                        help='Directory to save exported models')
    parser.add_argument('--height', type=int, default=1408,
                        help='Input height')
    parser.add_argument('--width', type=int, default=1408,
                        help='Input width')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--max-batch-size', type=int, default=4,
                        help='Maximum batch size for TensorRT')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth value in meters')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Minimum depth value in meters')
    parser.add_argument('--test', action='store_true',
                        help='Test the exported engine')
    
    args = parser.parse_args()
    
    # Check TensorRT availability
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
    except ImportError:
        print("ERROR: TensorRT not found. Please install TensorRT.")
        print("Visit: https://developer.nvidia.com/tensorrt")
        return
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "rt_monodepth_s.onnx"
    engine_path = output_dir / f"rt_monodepth_s_{'fp16' if args.fp16 else 'fp32'}.engine"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("ERROR: CUDA is required for TensorRT export")
        return
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = RTMonoDepthS(max_depth=args.max_depth, min_depth=args.min_depth)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Input shape
    input_shape = (args.batch_size, 3, args.height, args.width)
    
    # Export to ONNX
    export_to_onnx(model.cpu(), str(onnx_path), input_shape)
    
    # Build TensorRT engine
    builder = TensorRTBuilder(
        str(onnx_path),
        str(engine_path),
        fp16=args.fp16,
        max_batch_size=args.max_batch_size,
        input_shape=input_shape
    )
    
    engine = builder.build_engine()
    
    if engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return
    
    # Test engine
    if args.test:
        builder.test_engine(engine)
    
    # Save deployment info
    deployment_info = {
        'model': 'RT-MonoDepth-S',
        'checkpoint': args.checkpoint,
        'input_shape': list(input_shape),
        'output_shape': [args.batch_size, 1, args.height, args.width],
        'max_batch_size': args.max_batch_size,
        'precision': 'FP16' if args.fp16 else 'FP32',
        'max_depth': args.max_depth,
        'min_depth': args.min_depth,
        'num_parameters': num_params,
        'onnx_path': str(onnx_path),
        'engine_path': str(engine_path)
    }
    
    import json
    info_path = output_dir / 'deployment_info.json'
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\nDeployment files saved to: {output_dir}")
    print(f"  ONNX model: {onnx_path}")
    print(f"  TensorRT engine: {engine_path}")
    print(f"  Deployment info: {info_path}")


if __name__ == "__main__":
    main()