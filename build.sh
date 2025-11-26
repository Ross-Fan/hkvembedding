#!/bin/bash

echo "=== 构建HKV Python绑定 ==="

# 检查HKV路径
HKV_ROOT="/home/work/data/HKV/HierarchicalKV"
if [ ! -d "$HKV_ROOT/include" ]; then
    echo "❌ HKV路径不存在: $HKV_ROOT"
    exit 1
fi

echo "✅ 找到HKV: $HKV_ROOT"

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA编译器(nvcc)未找到"
    exit 1
fi

echo "✅ CUDA编译器可用: $(nvcc --version | grep release)"

# 安装依赖
echo "安装Python依赖..."
pip3 install pybind11[global] 

# 使用CMake编译（推荐）
echo "使用CMake编译Python扩展..."
mkdir -p build
cd build
cmake .. -DPYBIND11_FINDPYTHON=ON
make -j$(nproc)
cd ..

# 修正：更准确的检查逻辑
echo "检查编译结果..."
if [ -f "build/hkv_core.cpython-"*"-linux-gnu.so" ]; then
    echo "✅ 编译成功"
    ls -la build/hkv_core*.so
    
    # 显示文件信息
    echo "文件详情："
    file build/hkv_core*.so
    echo "文件大小："
    du -h build/hkv_core*.so
    
elif [ -f "hkv_core.cpython-"*"-linux-gnu.so" ]; then
    echo "✅ 编译成功（在当前目录）"
    ls -la hkv_core*.so
else
    echo "❌ 编译失败 - 未找到.so文件"
    echo "build目录内容："
    ls -la build/
    exit 1
fi

# 测试导入
echo "测试导入..."
python3 -c "
import sys
sys.path.append('build')
try:
    import hkv_core
    print('✅ 导入成功')
    print('版本:', hkv_core.version())
    
    # 简单测试
    table = hkv_core.HashTable(1024, 2048, 64, 1)
    print('✅ HashTable创建成功')
    print('容量:', table.capacity())
    print('嵌入维度:', table.embedding_dim)
    
except Exception as e:
    print('❌ 导入失败:', str(e))
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "✅ 构建完成"
