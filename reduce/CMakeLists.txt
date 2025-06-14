find_package(CUDAToolkit REQUIRED)
add_executable(my_reduce my_reduce.cu)
target_link_libraries(my_reduce PRIVATE CUDA::cudart )

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(my_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
endif()

