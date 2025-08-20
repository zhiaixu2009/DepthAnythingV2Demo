#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthAnythingV2 简单测试脚本
用于测试本地环境是否能够正常运行DepthAnythingV2模型
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import argparse


def test_environment():
    """测试环境配置"""
    print("=" * 50)
    print("环境测试开始...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
    print("=" * 50)


def create_test_image():
    """创建一个简单的测试图片"""
    # 创建一个简单的渐变图像用于测试
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建水平渐变
    for i in range(width):
        image[:, i] = [int(255 * i / width), 100, int(255 * (1 - i / width))]

    # 添加一些几何形状
    cv2.rectangle(image, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(image, (400, 300), 80, (0, 255, 0), -1)
    cv2.rectangle(image, (300, 100), (500, 180), (255, 0, 0), -1)

    return image


def load_depth_model():
    """加载DepthAnythingV2模型"""
    try:
        print("正在加载DepthAnythingV2模型...")
        # 使用transformers pipeline加载模型
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1,
        )
        print("模型加载成功!")
        return pipe
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用CPU模式...")
        try:
            pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=-1,
            )
            print("CPU模式加载成功!")
            return pipe
        except Exception as e2:
            print(f"CPU模式也失败: {e2}")
            return None


def process_image(pipe, image_path):
    """处理图片并生成深度图"""
    try:
        print(f"正在处理图片: {image_path}")

        # 读取图片
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            # 如果是numpy数组，转换为PIL图片
            image = Image.fromarray(image_path)

        print(f"图片尺寸: {image.size}")

        # 使用模型进行深度估计
        result = pipe(image)
        depth = result["depth"]

        print("深度估计完成!")
        return np.array(depth)

    except Exception as e:
        print(f"图片处理失败: {e}")
        return None


def save_results(original_image, depth_map, output_dir="output"):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存原图
    if isinstance(original_image, np.ndarray):
        cv2.imwrite(
            os.path.join(output_dir, "original.jpg"),
            cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
        )

    # 保存深度图
    if depth_map is not None:
        # 归一化深度图到0-255
        depth_normalized = (
            (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        ).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "depth.jpg"), depth_normalized)

        # 创建彩色深度图
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        cv2.imwrite(os.path.join(output_dir, "depth_colored.jpg"), depth_colored)

        print(f"结果已保存到 {output_dir} 文件夹")
        print(f"- original.jpg: 原始图片")
        print(f"- depth.jpg: 深度图（灰度）")
        print(f"- depth_colored.jpg: 深度图（彩色）")


def main():
    parser = argparse.ArgumentParser(description="DepthAnythingV2 简单测试")
    parser.add_argument(
        "--image", type=str, help="输入图片路径（可选，不提供则使用生成的测试图片）"
    )
    parser.add_argument("--output", type=str, default="output", help="输出文件夹路径")

    args = parser.parse_args()

    # 测试环境
    test_environment()

    # 准备测试图片
    if args.image and os.path.exists(args.image):
        print(f"使用提供的图片: {args.image}")
        test_image = args.image
    else:
        print("使用生成的测试图片")
        test_image = create_test_image()
        # 保存测试图片
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(
            os.path.join(args.output, "test_input.jpg"),
            cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR),
        )
        print(f"测试图片已保存: {os.path.join(args.output, 'test_input.jpg')}")

    # 加载模型
    pipe = load_depth_model()
    if pipe is None:
        print("模型加载失败，无法继续测试")
        return

    # 处理图片
    depth_map = process_image(pipe, test_image)

    # 保存结果
    if depth_map is not None:
        save_results(test_image, depth_map, args.output)
        print("\n测试完成! ✅")
        print(f"深度图统计信息:")
        print(f"- 最小深度值: {depth_map.min():.4f}")
        print(f"- 最大深度值: {depth_map.max():.4f}")
        print(f"- 平均深度值: {depth_map.mean():.4f}")
    else:
        print("\n测试失败! ❌")


if __name__ == "__main__":
    main()
