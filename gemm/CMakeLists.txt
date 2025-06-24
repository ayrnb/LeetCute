# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置所有可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/gemm)

# 添加调试选项（如果启用 Debug 构建）
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()

# 定义多个 CUDA 可执行文件
add_executable(my_gemm_baseline my_gemm_baseline.cu)


# 分别为每个可执行文件链接 CUDA 运行时库
target_link_libraries(my_gemm_baseline PRIVATE CUDA::cudart)