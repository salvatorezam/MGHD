cmake_minimum_required(VERSION 3.20)
project(fastpath_capi LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Build shared library
add_library(fastpath SHARED
    fastpath_capi.cpp
    persist_lut_capi.cu
)

# Set up include directories
target_include_directories(fastpath PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compiler definitions
target_compile_definitions(fastpath PRIVATE 
    FASTPATH_BUILD_SO=1
    FASTPATH_NO_TORCH=1
)

# Compiler options
target_compile_options(fastpath PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xptxas=-v;-arch=sm_90a;--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-fPIC>
)

# Link CUDA runtime
target_link_libraries(fastpath PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Set output name
set_target_properties(fastpath PROPERTIES
    OUTPUT_NAME "fastpath"
    PREFIX "lib"
    SUFFIX ".so"
)

# Install rules (optional)
install(TARGETS fastpath
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES fastpath_capi.h
    DESTINATION include
)
