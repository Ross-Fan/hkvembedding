#!/bin/bash

echo "=== 构建HKV Python Wheel包 ==="

# 检查环境
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA编译器(nvcc)未找到"
    exit 1
fi

echo "✅ CUDA编译器可用: $(nvcc --version | grep release)"

# 清理之前的构建
echo "清理之前的构建..."
rm -rf build/ dist/ *.egg-info/
rm -rf python/__pycache__/ python/*/__pycache__/

# 安装构建依赖
echo "安装构建依赖..."
pip install --upgrade build wheel setuptools pybind11

# 方法1：使用setup.py构建
echo "使用setup.py构建..."
python setup.py bdist_wheel

# 方法2：使用现代构建工具（如果有pyproject.toml）
# echo "使用build构建..."
# python -m build

# 检查结果
echo "检查构建结果..."
if [ -f dist/*.whl ]; then
    echo "✅ Wheel包构建成功"
    ls -la dist/
    
    # 显示wheel内容
    echo "Wheel包内容："
    python -m zipfile -l dist/*.whl
    
    # 可选：测试安装
    echo "测试安装..."
    pip install dist/*.whl --force-reinstall
    
    # 测试导入
    python -c "
import sys
print('Python版本:', sys.version)

try:
    # 测试核心模块
    import hkv_core
    print('✅ hkv_core导入成功')
    print('版本:', hkv_core.version())
    
    # 测试Python包装
    import hkv_python_binding
    print('✅ hkv_python_binding导入成功')
    
    # 测试创建HashTable
    table = hkv_python_binding.HashTable(1024, 2048, 64, 1)
    print('✅ HashTable创建成功')
    print('容量:', table.capacity())
    
    # 测试PyTorch封装
    embedding = hkv_python_binding.HierarchicalHashEmbedding(64, 2048, 1024, 1)
    print('✅ HierarchicalHashEmbedding创建成功')
    print(embedding)
    
except Exception as e:
    print('❌ 测试失败:', str(e))
    import traceback
    traceback.print_exc()
    exit(1)
"
    
    echo "✅ 所有测试通过"
else
    echo "❌ Wheel包构建失败"
    echo "检查错误信息："
    ls -la dist/ || echo "dist目录不存在"
    exit 1
fi

echo "✅ 构建完成"
echo "安装命令: pip install dist/$(ls dist/*.whl | head -1 | xargs basename)"
