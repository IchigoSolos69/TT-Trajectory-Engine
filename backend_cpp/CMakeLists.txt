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
    src/blurball_engine.cpp
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
add_executable(test_analysis_journal tests/test_analysis_journal.cpp)
add_executable(test_bounce_matcher tests/test_bounce_matcher.cpp)
add_executable(test_tracking tests/test_tracking.cpp src/bytetrack.cpp)
add_executable(test_training_recorder tests/test_training_recorder.cpp)
add_executable(test_observed_spin tests/test_observed_spin.cpp)
add_executable(test_imageplane_flight tests/test_imageplane_flight.cpp)
target_link_libraries(test_imageplane_flight PRIVATE ${OpenCV_LIBS})
target_link_libraries(test_observed_spin PRIVATE ${OpenCV_LIBS})
add_executable(test_trajectory_intelligence tests/test_trajectory_intelligence.cpp)
target_link_libraries(test_trajectory_intelligence PRIVATE ${OpenCV_LIBS})
add_executable(eval_trajectory_uncertainty tests/eval_trajectory_uncertainty.cpp)
add_executable(eval_temporal_recovery tests/eval_temporal_recovery.cpp src/temporal_recovery.cpp)
add_executable(eval_blurball_native tests/eval_blurball_native.cpp src/blurball_engine.cpp)
add_executable(test_blurball_stream tests/test_blurball_stream.cpp src/blurball_engine.cpp)
add_executable(eval_imageplane_forecast tests/eval_imageplane_forecast.cpp src/blurball_engine.cpp)
target_link_libraries(eval_blurball_native PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
target_link_libraries(test_blurball_stream PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
target_link_libraries(eval_imageplane_forecast PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
if(WIN32)
    target_link_libraries(eval_blurball_native PRIVATE advapi32)
    target_link_libraries(test_blurball_stream PRIVATE advapi32)
    target_link_libraries(eval_imageplane_forecast PRIVATE advapi32 ws2_32 crypt32 ole32 uuid)
endif()
target_link_libraries(eval_temporal_recovery PRIVATE ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
add_executable(eval_temporal_pipeline tests/eval_temporal_pipeline.cpp src/temporal_recovery.cpp)
target_link_libraries(eval_temporal_pipeline PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
add_executable(test_temporal_window tests/test_temporal_window.cpp)
target_link_libraries(test_temporal_window PRIVATE ${OpenCV_LIBS})
add_executable(test_rally_learning tests/test_rally_learning.cpp)
target_link_libraries(test_rally_learning PRIVATE ${OpenCV_LIBS})
target_link_libraries(eval_trajectory_uncertainty PRIVATE ${OpenCV_LIBS})
target_link_libraries(test_training_recorder PRIVATE ${OpenCV_LIBS})
target_link_libraries(test_tracking PRIVATE ${OpenCV_LIBS})
add_executable(eval_bounce tests/eval_bounce.cpp src/physics_engine.cpp)
add_executable(benchmark_latency tests/benchmark_latency.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(benchmark_realtime_loop tests/benchmark_realtime_loop.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(eval_real_trt_system tests/eval_real_trt_system.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(dump_frame_1672_cpp tests/dump_frame_1672_cpp.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(test_gpu_burst_clocks tests/test_gpu_burst_clocks.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(test_gpu_pipeline_latency tests/test_gpu_pipeline_latency.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu src/bytetrack.cpp src/telemetry_server.cpp src/physics_engine.cpp src/homography_calibrator.cpp)
add_executable(test_cudacodec tests/test_cudacodec.cpp)

foreach(detector_target dump_detector_predictions dump_scene_predictions test_letterbox test_scene_geometry)
    add_executable(${detector_target} tests/${detector_target}.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu)
    target_link_libraries(${detector_target} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${TENSORRT_LIBS})
endforeach()

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

if(WIN32)
    target_link_libraries(tt_trajectory_engine PRIVATE ws2_32 advapi32 crypt32 ole32 uuid)
    target_link_libraries(benchmark_latency PRIVATE ws2_32 advapi32 crypt32)
    target_link_libraries(benchmark_realtime_loop PRIVATE ws2_32 advapi32 crypt32)
    target_link_libraries(eval_real_trt_system PRIVATE ws2_32 advapi32 crypt32)
    target_link_libraries(dump_frame_1672_cpp PRIVATE ws2_32 advapi32 crypt32)
    target_link_libraries(test_gpu_burst_clocks PRIVATE ws2_32 advapi32 crypt32)
    target_link_libraries(test_gpu_pipeline_latency PRIVATE ws2_32 advapi32 crypt32)
endif()

# Validate a candidate without replacing an executable that is in use.
add_executable(tt_intelligence_candidate EXCLUDE_FROM_ALL ${SOURCES})
target_link_libraries(tt_intelligence_candidate PRIVATE $<TARGET_PROPERTY:tt_trajectory_engine,LINK_LIBRARIES>)

# Offline paired-model research. Production source/model selection is unchanged.
add_executable(test_ball_candidate_selection tests/test_ball_candidate_selection.cpp)
target_link_libraries(test_ball_candidate_selection PRIVATE ${OpenCV_LIBS})
add_executable(test_ball_candidate_selection_v2 tests/test_ball_candidate_selection_v2.cpp)
target_link_libraries(test_ball_candidate_selection_v2 PRIVATE ${OpenCV_LIBS})
add_executable(eval_paired_ball_candidates EXCLUDE_FROM_ALL tests/eval_paired_ball_candidates.cpp src/tensorrt_engine.cpp src/blurball_engine.cpp src/cuda_preprocess.cu)
target_link_libraries(eval_paired_ball_candidates PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
if(WIN32)
    target_link_libraries(eval_paired_ball_candidates PRIVATE advapi32 ws2_32 crypt32 ole32 uuid)
endif()

# Shape-aware model inspection and exact raw tensor parity diagnostics.
foreach(TARGET_NAME test_detector_tensor_shapes dump_detector_raw)
    add_executable(${TARGET_NAME} EXCLUDE_FROM_ALL tests/${TARGET_NAME}.cpp src/tensorrt_engine.cpp src/cuda_preprocess.cu)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    if(WIN32)
        target_link_libraries(${TARGET_NAME} PRIVATE advapi32)
    endif()
endforeach()
