# DepthAnythingV2Demo

一个简单的DepthAnythingV2深度估计测试项目，用于验证本地环境是否能够正常运行。

## 功能特性

- 🚀 简单的命令行界面
- 🔧 自动环境检测（CUDA/CPU）
- 🖼️ 支持自定义图片或自动生成测试图片
- 📊 输出多种格式的深度图（灰度图、彩色图）
- 📈 显示深度图统计信息

## 环境要求

- Python 3.8+
- PyTorch 2.4.0
- CUDA（可选，用于GPU加速）

## 快速开始

### Windows用户

1. 运行安装和测试脚本：
```cmd
install_and_run.bat
```

### Linux/Mac用户

1. 给脚本添加执行权限：
```bash
chmod +x run_test.sh
```

2. 运行安装和测试脚本：
```bash
./run_test.sh
```

### 手动安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行测试：
```bash
python test_depth.py
```

## 使用方法

### 基本使用
```bash
python test_depth.py
```

### 使用自定义图片
```bash
python test_depth.py --image path/to/your/image.jpg
```

### 指定输出文件夹
```bash
python test_depth.py --output my_results
```

## 输出文件

运行完成后，会在输出文件夹中生成以下文件：

- `test_input.jpg` - 测试输入图片（如果使用自动生成的图片）
- `original.jpg` - 原始图片
- `depth.jpg` - 深度图（灰度）
- `depth_colored.jpg` - 深度图（彩色，使用plasma色彩映射）

## 故障排除

### 模型下载问题
如果遇到模型下载缓慢或失败，可以尝试：
1. 使用代理
2. 手动下载模型到本地
3. 使用国内镜像源

### CUDA相关问题
- 确保CUDA版本与PyTorch版本兼容
- 如果CUDA不可用，程序会自动切换到CPU模式

### 内存不足
- 使用较小的输入图片
- 确保有足够的系统内存（建议8GB+）

## 技术细节

- 使用HuggingFace Transformers库
- 模型：`depth-anything/Depth-Anything-V2-Small-hf`
- 支持GPU和CPU推理
- 自动处理图片格式转换
