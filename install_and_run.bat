@echo off
chcp 65001
echo DepthAnythingV2 环境安装和测试脚本
echo ================================

echo 正在安装Python依赖包...
pip install -r requirements.txt

echo.
echo 依赖包安装完成！
echo.
echo 开始运行测试...
python test_depth.py

echo.
echo 测试完成！请查看output文件夹中的结果。
pause 