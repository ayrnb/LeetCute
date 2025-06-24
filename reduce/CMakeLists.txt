# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置所有可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/reduce)

# 添加调试选项（如果启用 Debug 构建）
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()

# 定义多个 CUDA 可执行文件
add_executable(my_reduce_baseline my_reduce_baseline.cu)
add_executable(my_reduce_share_memory my_reduce_share_memory.cu)
add_executable(my_reduce_warp_divergence my_reduce_warp_divergence.cu)
add_executable(my_reduce_no_warp_conflict my_reduce_no_warp_conflict.cu)
add_executable(my_reduce_use_idle_thread my_reduce_use_idle_thread.cu)
add_executable(my_reduce_warp_shulf my_reduce_warp_shulf.cu)

# 分别为每个可执行文件链接 CUDA 运行时库
target_link_libraries(my_reduce_baseline PRIVATE CUDA::cudart)
target_link_libraries(my_reduce_share_memory PRIVATE CUDA::cudart)
target_link_libraries(my_reduce_warp_divergence PRIVATE CUDA::cudart)
target_link_libraries(my_reduce_no_warp_conflict PRIVATE CUDA::cudart)
target_link_libraries(my_reduce_use_idle_thread PRIVATE CUDA::cudart)
target_link_libraries(my_reduce_warp_shulf PRIVATE CUDA::cudart)