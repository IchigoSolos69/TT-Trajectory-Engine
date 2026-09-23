#include "tensorrt_engine.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdlib>
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

    std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);

    // Warm up (10 runs)
    for (int i = 0; i < 10; ++i) {
        trtEngine.inferFrame(inputBuffer);
    }

    // Test A: Single isolated inference with 10ms CPU sleep in between (simulating real-time 11ms frame loop)
    std::vector<float> singleTimes;
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        InferenceTimingDetailed timing;
        trtEngine.inferFrameDetailed(inputBuffer, timing);
        singleTimes.push_back(timing.gpuComputeMs);
    }

    // Test B: 100 consecutive back-to-back inferences on warm GPU (simulating trtexec continuous load)
    std::vector<float> burstTimes;
    for (int i = 0; i < 100; ++i) {
        InferenceTimingDetailed timing;
        trtEngine.inferFrameDetailed(inputBuffer, timing);
        burstTimes.push_back(timing.gpuComputeMs);
    }

    std::sort(singleTimes.begin(), singleTimes.end());
    std::sort(burstTimes.begin(), burstTimes.end());

    float medSingle = singleTimes[singleTimes.size() / 2];
    float medBurst = burstTimes[burstTimes.size() / 2];

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "  GPU CLOCK-RAMP & POWER STATE VERIFICATION TEST" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  Isolated Burst GPU Compute (10ms CPU sleep idle window) : " << medSingle << " ms" << std::endl;
    std::cout << "  100 Back-to-Back Warm Inferences (Boosted GPU Load)      : " << medBurst << " ms" << std::endl;
    std::cout << "  Standalone trtexec GPU Compute Reference                   : 0.66 ms" << std::endl;
    std::cout << "  Delta (Isolated Idle Window Penalty)                       : +" << (medSingle - medBurst) << " ms" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "  Querying active NVIDIA GPU SM Clock via nvidia-smi:" << std::endl;
    std::cout << "=========================================================================\n" << std::endl;
    std::fflush(stdout);

    // Run nvidia-smi query
    std::system("nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv");

    return 0;
}
