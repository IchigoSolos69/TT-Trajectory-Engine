#include "tensorrt_engine.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main() {
    std::string videoPath = "python_tools/dataset/raw_videos/test_1.mp4";
    std::string modelPath = "models/yolov11_ttgames.engine";

    cv::VideoCapture cap(videoPath);
    cv::Mat frame;
    cap.read(frame);
    cap.release();

    TensorRTEngine trtEngine(modelPath);
    if (!trtEngine.init()) {
        std::cerr << "Engine init failed" << std::endl;
        return 1;
    }

    std::cout << "=========================================================================" << std::endl;
    std::cout << "  CPU PREPROCESSING VS CUDA GPU PREPROCESSING BENCHMARK" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    // Warm up
    std::vector<float> inputBufferCPU = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);
    for (int i = 0; i < 10; ++i) {
        trtEngine.inferFrame(inputBufferCPU, frame.cols, frame.rows);
        trtEngine.inferFrameGPU(frame);
    }

    // Benchmark Path A: CPU Preprocess + GPU Inference
    std::vector<float> cpuPrepTimes, cpuInferTimes, cpuTotalTimes;
    for (int i = 0; i < 50; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> inputData = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);
        auto t1 = std::chrono::high_resolution_clock::now();

        InferenceTimingDetailed timing;
        trtEngine.inferFrameDetailed(inputData, timing, frame.cols, frame.rows);
        auto t2 = std::chrono::high_resolution_clock::now();

        float prepMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float totalMs = std::chrono::duration<float, std::milli>(t2 - t0).count();

        cpuPrepTimes.push_back(prepMs);
        cpuInferTimes.push_back(timing.gpuComputeMs);
        cpuTotalTimes.push_back(totalMs);
    }

    // Benchmark Path B: CUDA Preprocess Kernel + GPU Inference
    std::vector<float> gpuH2DTimes, gpuComputeTimes, gpuTotalTimes;
    for (int i = 0; i < 50; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        InferenceTimingDetailed timing;
        trtEngine.inferFrameGPUDetailed(frame, timing);
        auto t1 = std::chrono::high_resolution_clock::now();

        float totalMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        gpuH2DTimes.push_back(timing.h2dMs);
        gpuComputeTimes.push_back(timing.gpuComputeMs);
        gpuTotalTimes.push_back(totalMs);
    }

    std::sort(cpuPrepTimes.begin(), cpuPrepTimes.end());
    std::sort(cpuTotalTimes.begin(), cpuTotalTimes.end());
    std::sort(gpuH2DTimes.begin(), gpuH2DTimes.end());
    std::sort(gpuTotalTimes.begin(), gpuTotalTimes.end());

    float medCpuPrep = cpuPrepTimes[cpuPrepTimes.size() / 2];
    float medCpuTotal = cpuTotalTimes[cpuTotalTimes.size() / 2];

    float medGpuH2D = gpuH2DTimes[gpuH2DTimes.size() / 2];
    float medGpuTotal = gpuTotalTimes[gpuTotalTimes.size() / 2];

    std::cout << "  PATH A: Standard OpenCV CPU Preprocess + TRT GPU Execution:" << std::endl;
    std::cout << "    - CPU Preprocessing Time (p50) : " << medCpuPrep << " ms" << std::endl;
    std::cout << "    - Total Path A Execution Time  : " << medCpuTotal << " ms" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "  PATH B: Fused CUDA Preprocessing Kernel + TRT GPU Execution:" << std::endl;
    std::cout << "    - CUDA H2D + Preprocess Time (p50): " << medGpuH2D << " ms" << std::endl;
    std::cout << "    - Total Path B Execution Time     : " << medGpuTotal << " ms" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "  Empirical Latency Reduction (CPU Work Offloaded): -" << (medCpuTotal - medGpuTotal) << " ms (" << ((medCpuTotal - medGpuTotal) / medCpuTotal * 100.0f) << "% faster)" << std::endl;
    std::cout << "=========================================================================\n" << std::endl;

    return 0;
}
