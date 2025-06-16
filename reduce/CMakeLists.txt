find_package(CUDAToolkit REQUIRED)
add_executable(my_reduce_baseline my_reduce_baseline.cu)
target_link_libraries(my_reduce_baseline PRIVATE CUDA::cudart )

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(my_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
endif()

add_executable(my_reduce_share_memory my_reduce_share_memory.cu)
target_link_libraries(my_reduce_share_memory PRIVATE CUDA::cudart )

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(my_reduce_share_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
endif()

add_executable(my_reduce_warp_divergence my_reduce_warp_divergence.cu)
target_link_libraries(my_reduce_warp_divergence PRIVATE CUDA::cudart )

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(my_reduce_warp_divergence PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
endif()