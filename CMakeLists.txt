cmake_minimum_required(VERSION 3.18)
project(hkv_python_binding LANGUAGES CXX CUDA)

# 设置标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 专门为A10优化 (sm_86)
set(CMAKE_CUDA_ARCHITECTURES "86")

# 查找依赖
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# HKV路径
set(HKV_ROOT "/home/work/data/HKV/HierarchicalKV")
if(NOT EXISTS "${HKV_ROOT}/include")
    message(FATAL_ERROR "HierarchicalKV not found at ${HKV_ROOT}")
endif()

# 源文件
set(SOURCES
    src/hkv_wrapper.cu
    src/python_bindings.cu
)

# 创建模块
pybind11_add_module(hkv_core ${SOURCES})

# 包含目录
target_include_directories(hkv_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HKV_ROOT}/include
)

# CUDA属性
set_target_properties(hkv_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # 改为OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF  # 改为OFF
    POSITION_INDEPENDENT_CODE ON
)

# A10优化的编译选项
target_compile_options(hkv_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -O3
        --compiler-options=-fPIC
        --compiler-options=-fvisibility=hidden
        --compiler-options=-Wno-unknown-pragmas
        -gencode arch=compute_86,code=sm_86
        --ptxas-options=-v
        --use_fast_math
    >
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
)

# 链接库
target_link_libraries(hkv_core PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# 编译定义
target_compile_definitions(hkv_core PRIVATE 
    VERSION_INFO="1.0.0"
    CUDA_ARCH=86
)

# 设置输出目录
set_target_properties(hkv_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 打印配置信息
message(STATUS "Configuring for NVIDIA A10 (sm_86)")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
