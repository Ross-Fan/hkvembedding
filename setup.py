from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension, find_packages
import os
import sys
import subprocess
from pathlib import Path

# HierarchicalKV路径
HKV_ROOT = "/home/work/data/HKV/HierarchicalKV"

# 检查HKV路径是否存在
if not os.path.exists(f"{HKV_ROOT}/include"):
    raise RuntimeError(f"HierarchicalKV not found at {HKV_ROOT}")

# 检查CUDA是否可用
def check_cuda():
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

if not check_cuda():
    raise RuntimeError("CUDA compiler (nvcc) not found")

# 自定义构建类
class CudaBuildExt(build_ext):
    def build_extensions(self):
        # 为每个扩展设置CUDA编译
        for ext in self.extensions:
            self._setup_cuda_extension(ext)
        super().build_extensions()
    
    def _setup_cuda_extension(self, ext):
        # 设置编译器能识别.cu文件
        if hasattr(self.compiler, 'src_extensions'):
            self.compiler.src_extensions.append('.cu')
        
        # 保存原始编译方法
        original_compile = self.compiler._compile
        
        def _compile_cuda(obj, src, ext_type, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cu'):
                # CUDA文件编译选项
                self.compiler.set_executable('compiler_so', 'nvcc')
                # 修正：移除不支持的选项，使用正确的nvcc选项
                cuda_args = [
                    '-O3',
                    '-std=c++17',
                    '--compiler-options=-fPIC',  # 正确的写法
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-gencode', 'arch=compute_86,code=sm_86',
                    '--use_fast_math',
                    '-c'
                ]
                extra_postargs = cuda_args
            else:
                # C++文件编译选项
                extra_postargs = ['-O3', '-std=c++17', '-fPIC'] + (extra_postargs or [])
            
            return original_compile(obj, src, ext_type, cc_args, extra_postargs, pp_opts)
        
        self.compiler._compile = _compile_cuda

# 读取README
def read_readme():
    readme_path = Path("README.md")
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Python bindings for HierarchicalKV - High-performance GPU hashtable for embeddings"

# 修正：使用正确的CUDA编译选项
ext_modules = [
    Pybind11Extension(
        "hkv_embedding.hkv_core",
        [
            "src/hkv_wrapper.cu",
            "src/python_bindings.cu",
        ],
        include_dirs=[
            pybind11.get_include(),
            "include",
            f"{HKV_ROOT}/include/",
            "/usr/local/cuda/include/",
        ],
        libraries=["cudart", "cuda"],
        library_dirs=["/usr/local/cuda/lib64/"],
        language='c++',
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"1.0.0"')],
        # 移除有问题的编译选项
    ),
]

setup(
    name="hkv-embedding",
    version="1.0.0",
    author="Wei Fan",
    author_email="your.email@example.com",
    description="Python bindings for HierarchicalKV - High-performance GPU hashtable for embeddings",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hkv-python-binding",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CudaBuildExt},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        # "torch>=1.9.0",
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        # "pytorch": [
        #     "torch>=1.9.0,<3.0.0",  # 放宽版本范围
        # ],
        # "pytorch-cu121": [
        #     "torch>=2.0.0,<3.0.0",
            
        # ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="gpu hashtable embeddings cuda machine-learning pytorch",
)
