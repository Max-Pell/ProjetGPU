cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME "PlaneSweep")
project(${PROJECT_NAME} LANGUAGES CUDA CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source build detected!")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)

if(${CMAKE_CUDA_COMPILER} STREQUAL "")
    message(FATAL_ERROR "CUDA COMPILER not found!")
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    set(CMAKE_CXX_EXTENSIONS OFF) 
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

# Enable relocatable device code
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --relocatable-device-code=true")

file(GLOB kernel_plane_sweep_headers
    ./kernels/*.cuh
)

file(GLOB kernel_plane_sweep_src
    ./kernels/*.cu
)

source_group(kernel_headers FILES ${kernel_plane_sweep_headers})
source_group(kernel_sources FILES ${kernel_plane_sweep_src})

add_library(${PROJECT_NAME}CUDA ${kernel_plane_sweep_headers} ${kernel_plane_sweep_src})
set_property(TARGET ${PROJECT_NAME}CUDA PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
set_target_properties(${PROJECT_NAME}CUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(${PROJECT_NAME}CUDA PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;>")

target_include_directories(${PROJECT_NAME}CUDA PUBLIC
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}CUDA PRIVATE ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${CUVID_LIB} ${OpenCV_LIBS})

file(GLOB source_files_planesweep
    ./src/*.cpp
    ./src/*.hpp
    ./src/*.h
    ${kernel_plane_sweep_headers}
)

add_executable(${PROJECT_NAME} ${source_files_planesweep})
set_property(TARGET ${PROJECT_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-lineinfo;--source-in-ptx;>")

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}CUDA ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${CUVID_LIB} ${OpenCV_LIBS})

file(COPY "data" DESTINATION ${CMAKE_BINARY_DIR})
