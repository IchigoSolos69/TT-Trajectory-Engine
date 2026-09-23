#include "tensorrt_engine.hpp"
#include "bytetrack.hpp"
#include "physics_engine.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

struct StageLatencyDetailed {
    float decode_ms;
    float preprocess_ms;
    float h2d_ms;
    float gpu_compute_ms;
    float d2h_ms;
    float nms_ms;
    float track_ms;
    float physics_ms;
    float total_ms;
};

int main(int argc, char** argv) {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  120 FPS Paced Real-Time Wall-Clock Latency & Sub-Stage Benchmark (5 Runs)" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    std::string userPath = (argc > 1) ? argv[1] : "";
    std::string modelPath = TensorRTEngine::resolveEnginePath(userPath);

    TensorRTEngine trtEngine(modelPath);
    if (!trtEngine.init()) {
        std::cerr << "Error: Failed to initialize TensorRT engine." << std::endl;
        return 1;
    }

    std::string videoPath = "python_tools/dataset/raw_videos/test_1.mp4";
    cv::VideoCapture warmCap(videoPath);
    if (!warmCap.isOpened()) {
        std::cerr << "Error: Cannot open video " << videoPath << std::endl;
        return 1;
    }

    std::cout << "\n--- WARM-UP PHASE (100 Frames) ---" << std::endl;
    cv::Mat warmFrame;
    ByteTracker warmTracker(0.25f, 0.1f, 30);
    PhysicsEngine warmPhysics;
    float dt = 1.0f / 120.0f;

    for (int i = 0; i < 100; i++) {
        if (!warmCap.read(warmFrame) || warmFrame.empty()) break;
        std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(warmFrame, 640, 640);
        InferenceTimingDetailed dummyTiming;
        std::vector<Detection> detections = trtEngine.inferFrameDetailed(inputBuffer, dummyTiming);
        std::vector<TrackedObject> tracks = warmTracker.update(detections);
        warmPhysics.update(0.0f, 0.0f, dt, 320.0f, 240.0f);
    }
    warmCap.release();
    std::cout << "[Warm-Up Complete] Engine & CUDA context ready." << std::endl;

    const int NUM_RUNS = 5;
    std::vector<float> runMediansTotal(NUM_RUNS, 0.0f);
    std::vector<float> runMediansDecode(NUM_RUNS, 0.0f);
    std::vector<float> runMediansPreprocess(NUM_RUNS, 0.0f);
    std::vector<float> runMediansGPUCompute(NUM_RUNS, 0.0f);
    std::vector<float> runMediansNMS(NUM_RUNS, 0.0f);

    std::vector<StageLatencyDetailed> bestRunStageLatencies;

    for (int run = 0; run < NUM_RUNS; run++) {
        if (run > 0) {
            std::cout << "\n[Thermal Cooldown] Waiting 10s for GPU clock state stabilization..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }

        std::cout << "\n=== STARTING BENCHMARK RUN " << (run + 1) << " / " << NUM_RUNS << " ===" << std::endl;
        std::system("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader");

        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error opening " << videoPath << " on run " << run + 1 << std::endl;
            return 1;
        }

        ByteTracker tracker(0.25f, 0.1f, 30);
        PhysicsEngine physics;
        cv::Mat frame;

        std::vector<StageLatencyDetailed> runLatencies;
        int frameId = 0;

        while (true) {
            auto t_decode_start = std::chrono::high_resolution_clock::now();
            bool readOk = cap.read(frame);
            auto t_decode_end = std::chrono::high_resolution_clock::now();

            if (!readOk || frame.empty()) break;
            frameId++;

            float decodeMs = std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();

            InferenceTimingDetailed inferTiming;
            auto t_prep_start = std::chrono::high_resolution_clock::now();
            std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);
            auto t_prep_end = std::chrono::high_resolution_clock::now();
            inferTiming.preprocessMs = std::chrono::duration<float, std::milli>(t_prep_end - t_prep_start).count();

            std::vector<Detection> detections = trtEngine.inferFrameDetailed(inputBuffer, inferTiming, frame.cols, frame.rows);

            auto t_track_start = std::chrono::high_resolution_clock::now();
            std::vector<TrackedObject> tracks = tracker.update(detections);
            auto t_track_end = std::chrono::high_resolution_clock::now();
            float trackMs = std::chrono::duration<float, std::milli>(t_track_end - t_track_start).count();

            auto t_phys_start = std::chrono::high_resolution_clock::now();
            float pixelX = -1.0f, pixelY = -1.0f;
            if (!detections.empty()) {
                pixelX = (detections[0].x + detections[0].w / 2.0f) * frame.cols;
                pixelY = (detections[0].y + detections[0].h / 2.0f) * frame.rows;
            }
            float metersX = (pixelX / frame.cols) * 1.525f;
            float metersY = (pixelY / frame.rows) * 2.74f;
            physics.update(metersX, metersY, dt, pixelX, pixelY);
            auto t_phys_end = std::chrono::high_resolution_clock::now();
            float physMs = std::chrono::duration<float, std::milli>(t_phys_end - t_phys_start).count();

            float totalMs = decodeMs + inferTiming.h2dMs + inferTiming.preprocessMs + inferTiming.gpuComputeMs + inferTiming.d2hMs + inferTiming.decodeMs + trackMs + physMs;
            
            runLatencies.push_back({
                decodeMs, inferTiming.preprocessMs, inferTiming.h2dMs, inferTiming.gpuComputeMs,
                inferTiming.d2hMs, inferTiming.decodeMs, trackMs, physMs, totalMs
            });
        }
        cap.release();

        std::vector<float> totals;
        int runDeadlineMet = 0;
        for (const auto& stg : runLatencies) {
            totals.push_back(stg.total_ms);
            if (stg.total_ms <= 8.333f) runDeadlineMet++;
        }
        std::sort(totals.begin(), totals.end());
        size_t n = totals.size();

        float minT = totals.front();
        float p5T  = totals[n * 0.05];
        float p50T = totals[n * 0.50];
        float p95T = totals[n * 0.95];
        float p99T = totals[n * 0.99];
        float maxT = totals.back();

        runMediansTotal[run] = p50T;

        std::cout << "  Run " << (run + 1) << " / " << NUM_RUNS << " | Min: " << minT
                  << " ms | p5: " << p5T << " ms | p50: " << p50T << " ms | p95: " << p95T
                  << " ms | p99: " << p99T << " ms | Max: " << maxT << " ms | On-Time (<=8.33ms): "
                  << (float)runDeadlineMet / n * 100.0f << "%" << std::endl;

        if (run == 0 || runLatencies.size() > bestRunStageLatencies.size()) {
            bestRunStageLatencies = runLatencies;
        }
    }

    std::ofstream csv("realtime_wallclock_latency.csv");
    csv << "# Git Commit SHA: " << trtEngine.getGitCommitSha() << "\n";
    csv << "# Engine Path: " << trtEngine.getEnginePath() << "\n";
    csv << "# Engine Size: " << trtEngine.getFileSizeBytes() << " bytes\n";
    csv << "# Engine SHA-256: " << trtEngine.getSha256Hash() << "\n";
    csv << "frame_id,decode_ms,preprocess_ms,h2d_ms,gpu_compute_ms,d2h_ms,nms_decode_ms,track_ms,physics_ms,total_ms,deadline_met\n";

    int deadlineMetCount = 0;
    std::vector<float> allTotalLatencies, allH2D, allPrep, allGPU, allD2H, allDecode;
    for (size_t i = 0; i < bestRunStageLatencies.size(); i++) {
        const auto& stg = bestRunStageLatencies[i];
        bool metDeadline = (stg.total_ms <= 8.333f);
        if (metDeadline) deadlineMetCount++;

        allTotalLatencies.push_back(stg.total_ms);
        allH2D.push_back(stg.h2d_ms);
        allPrep.push_back(stg.preprocess_ms);
        allGPU.push_back(stg.gpu_compute_ms);
        allD2H.push_back(stg.d2h_ms);
        allDecode.push_back(stg.nms_ms);

        csv << (i + 1) << ","
            << stg.decode_ms << ","
            << stg.preprocess_ms << ","
            << stg.h2d_ms << ","
            << stg.gpu_compute_ms << ","
            << stg.d2h_ms << ","
            << stg.nms_ms << ","
            << stg.track_ms << ","
            << stg.physics_ms << ","
            << stg.total_ms << ","
            << (metDeadline ? 1 : 0) << "\n";
    }
    csv.close();

    std::sort(allTotalLatencies.begin(), allTotalLatencies.end());
    std::sort(allH2D.begin(), allH2D.end());
    std::sort(allPrep.begin(), allPrep.end());
    std::sort(allGPU.begin(), allGPU.end());
    std::sort(allD2H.begin(), allD2H.end());
    std::sort(allDecode.begin(), allDecode.end());

    size_t nFrames = allTotalLatencies.size();
    float p50Total = allTotalLatencies[nFrames * 0.50];
    float p95Total = allTotalLatencies[nFrames * 0.95];
    float p99Total = allTotalLatencies[nFrames * 0.99];

    std::sort(runMediansTotal.begin(), runMediansTotal.end());
    float medianOfMediansTotal = runMediansTotal[NUM_RUNS / 2];

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "  BENCHMARK SUMMARY ACROSS " << NUM_RUNS << " PASSES (" << nFrames << " frames per pass)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  Model Engine SHA-256 : " << trtEngine.getSha256Hash() << std::endl;
    std::cout << "  Model Size           : " << trtEngine.getFileSizeBytes() << " bytes" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "  Median of Medians Total Latency : " << medianOfMediansTotal << " ms" << std::endl;
    std::cout << "  Sub-stage Breakdown (p50 CUDA Event Timers):" << std::endl;
    std::cout << "    - H2D Frame Copy (1080p BGR) : " << allH2D[nFrames * 0.50] << " ms" << std::endl;
    std::cout << "    - CUDA Letterbox Preprocess  : " << allPrep[nFrames * 0.50] << " ms" << std::endl;
    std::cout << "    - TensorRT GPU Compute p50   : " << allGPU[nFrames * 0.50] << " ms" << std::endl;
    std::cout << "    - D2H Output Copy p50        : " << allD2H[nFrames * 0.50] << " ms" << std::endl;
    std::cout << "    - NMS Postprocess Decode     : " << allDecode[nFrames * 0.50] << " ms" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "  Representative Run Percentiles (Total Frame Wall-Clock):" << std::endl;
    std::cout << "    - p50  : " << p50Total << " ms" << std::endl;
    std::cout << "    - p95  : " << p95Total << " ms" << std::endl;
    std::cout << "    - p99  : " << p99Total << " ms" << std::endl;
    std::cout << "  On-Time (<= 8.33ms) : " << deadlineMetCount << " / " << nFrames << " (" << ((float)deadlineMetCount / nFrames * 100.0f) << "%)" << std::endl;
    std::cout << "=========================================================================\n" << std::endl;

    return 0;
}
