cmake_minimum_required(VERSION 3.18)
project(TT_Trajectory_Engine LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89 75)
set_source_files_properties(src/cuda_preprocess.cu PROPERTIES COMPILE_FLAGS "-gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_75,code=sm_75")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch Git Commit SHA at CMake Configure / Build time
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_SHA)
    set(GIT_COMMIT_SHA "UNKNOWN_BUILD_COMMIT")
endif()
add_compile_definitions(COMPILE_GIT_COMMIT_SHA="${GIT_COMMIT_SHA}")
message(STATUS "Compile-Time Git Commit SHA: ${GIT_COMMIT_SHA}")

# Set CXX standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optional CUDA and TensorRT Package Lookup
find_package(CUDA QUIET)
find_package(OpenCV QUIET)

if(CUDA_FOUND)
    message(STATUS "CUDA Found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(EXISTS "C:/TensorRT/include/NvInfer.h")
    message(STATUS "TensorRT Found at C:/TensorRT")
    include_directories("C:/TensorRT/include")
    link_directories("C:/TensorRT/lib")
    add_compile_definitions(HAS_TENSORRT)
    set(TENSORRT_LIBS "C:/TensorRT/lib/nvinfer_11.lib")
else()
    set(TENSORRT_LIBS "")
endif()

if(OpenCV_FOUND)
    message(STATUS "OpenCV Found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Target Source Files
set(SOURCES
    src/main.cpp
    src/tensorrt_engine.cpp
    src/cuda_preprocess.cu
    src/bytetrack.cpp
    src/telemetry_server.cpp
    src/homography_calibrator.cpp
    src/physics_engine.cpp
)


add_executable(tt_trajectory_engine ${SOURCES})

# Link OpenCV and CUDA if available
if(OpenCV_FOUND)
    target_link_libraries(tt_trajectory_engine PRIVATE ${OpenCV_LIBS})
endif()

if(CUDA_FOUND)
    target_link_libraries(tt_trajectory_engine PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

# Physics Engine Offline Unit Test & Latency Benchmark Executables
add_executable(test_physics_engine tests/test_physics_engine.cpp src/physics_engine.cpp)
add_executable(test_bounce_matcher tests/test_bounce_matcher.cpp)
add_executable(eval_bounce tests/eval_bounce.cpp src/physics_engine.cpp)
add_executable(benchmark_latency tests/benchmark_latency.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(benchmark_realtime_loop tests/benchmark_realtime_loop.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(eval_real_trt_system tests/eval_real_trt_system.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(dump_frame_1672_cpp tests/dump_frame_1672_cpp.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(test_gpu_burst_clocks tests/test_gpu_burst_clocks.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(test_gpu_pipeline_latency tests/test_gpu_pipeline_latency.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(test_cudacodec tests/test_cudacodec.cpp)

if(OpenCV_FOUND)
    target_link_libraries(benchmark_latency PRIVATE ${OpenCV_LIBS})
    target_link_libraries(benchmark_realtime_loop PRIVATE ${OpenCV_LIBS})
    target_link_libraries(eval_real_trt_system PRIVATE ${OpenCV_LIBS})
    target_link_libraries(dump_frame_1672_cpp PRIVATE ${OpenCV_LIBS})
    target_link_libraries(test_gpu_burst_clocks PRIVATE ${OpenCV_LIBS})
    target_link_libraries(test_gpu_pipeline_latency PRIVATE ${OpenCV_LIBS})
    target_link_libraries(test_cudacodec PRIVATE ${OpenCV_LIBS})
endif()
if(CUDA_FOUND)
    target_link_libraries(benchmark_latency PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(benchmark_realtime_loop PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(eval_real_trt_system PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(dump_frame_1672_cpp PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(test_gpu_burst_clocks PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(test_gpu_pipeline_latency PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

if(TENSORRT_LIBS)
    target_link_libraries(tt_trajectory_engine PRIVATE ${TENSORRT_LIBS})
    target_link_libraries(benchmark_latency PRIVATE ${TENSORRT_LIBS})
    target_link_libraries(benchmark_realtime_loop PRIVATE ${TENSORRT_LIBS})
    target_link_libraries(eval_real_trt_system PRIVATE ${TENSORRT_LIBS})
    target_link_libraries(dump_frame_1672_cpp PRIVATE ${TENSORRT_LIBS})
    target_link_libraries(test_gpu_burst_clocks PRIVATE ${TENSORRT_LIBS})
    target_link_libraries(test_gpu_pipeline_latency PRIVATE ${TENSORRT_LIBS})
endif()



