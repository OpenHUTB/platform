#!/usr/bin/env python
"""模型转换脚本"""
import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import onnx

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="转换模型格式")
    
    # 基础参数
    parser.add_argument("--input", type=str, required=True, help="输入模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出模型路径")
    parser.add_argument("--format", type=str, choices=['onnx', 'torchscript'], required=True, help="输出格式")
    
    # 转换参数
    parser.add_argument("--input-shape", type=str, help="输入形状, 如'1,3,224,224'")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset版本")
    parser.add_argument("--dynamic", action="store_true", help="使用动态形状")
    
    return parser.parse_args()

def convert_to_onnx(model, input_shape, output_path, opset_version=11, dynamic=False):
    """转换为ONNX格式"""
    # 创建示例输入
    if isinstance(input_shape, str):
        input_shape = [int(x) for x in input_shape.split(',')]
    dummy_input = torch.randn(*input_shape)
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic else None
    )
    
    # 验证模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
def convert_to_torchscript(model, output_path):
    """转换为TorchScript格式"""
    # 转换模型
    scripted_model = torch.jit.script(model)
    
    # 保存模型
    scripted_model.save(output_path)

def main():
    """主函数"""
    args = parse_args()
    
    # 加载模型
    model = torch.load(args.input)
    model.eval()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换模型
    if args.format == 'onnx':
        if not args.input_shape:
            raise ValueError("转换ONNX格式需要指定input_shape参数")
        convert_to_onnx(
            model,
            args.input_shape,
            str(output_path),
            args.opset,
            args.dynamic
        )
    elif args.format == 'torchscript':
        convert_to_torchscript(model, str(output_path))
        
    logging.info(f"模型已转换为{args.format}格式并保存到{output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 