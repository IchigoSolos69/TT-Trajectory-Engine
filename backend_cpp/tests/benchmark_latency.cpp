#include "tensorrt_engine.hpp"
#include "bytetrack.hpp"
#include "physics_engine.hpp"
#include "telemetry_server.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "[Benchmark] Running 2,000-Frame Full System Latency Benchmark (incl. Video Decode)..." << std::endl;

    TensorRTEngine trtEngine("models/yolov11_ttgames.engine");
    trtEngine.init();

    ByteTracker tracker(0.25f, 0.1f, 30);
    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;

    std::ofstream csv("latency_benchmark.csv");
    csv << "frame_id,decode_ms,preprocess_ms,inference_ms,bytetrack_ms,physics_ms,websocket_ms,total_ms\n";

    cv::VideoCapture cap("python_tools/dataset/raw_videos/game_1.mp4");
    if (!cap.isOpened()) {
        cap.open("python_tools/dataset/raw_videos/test_1.mp4");
    }

    cv::Mat frame;
    int frameId = 0;

    while (frameId < 2000) {
        frameId++;
        auto t0 = std::chrono::high_resolution_clock::now();

        bool readOk = false;
        if (cap.isOpened()) {
            readOk = cap.read(frame);
            if (!readOk || frame.empty()) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                readOk = cap.read(frame);
            }
        }
        if (!readOk || frame.empty()) {
            frame = cv::Mat::zeros(640, 640, CV_8UC3);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);
        auto t2 = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections = trtEngine.inferFrame(inputBuffer);
        auto t3 = std::chrono::high_resolution_clock::now();

        std::vector<TrackedObject> tracks = tracker.update(detections);
        auto t4 = std::chrono::high_resolution_clock::now();

        float metersX = 0.7625f;
        float metersY = 1.37f;
        float pixelX = 320.0f;
        float pixelY = 240.0f;

        for (auto& track : tracks) {
            if (track.class_id == 0) {
                pixelX = (track.x + track.w / 2.0f) * 640.0f;
                pixelY = (track.y + track.h / 2.0f) * 640.0f;
            }
        }

        PhysicsState pState = physics.update(metersX, metersY, dt, pixelX, pixelY);
        auto t5 = std::chrono::high_resolution_clock::now();

        std::string jsonStr = TelemetryServer::serializeToJSON(frameId, tracks, 120.0f, "benchmark", &pState);
        auto t6 = std::chrono::high_resolution_clock::now();

        float decodeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float prepMs = std::chrono::duration<float, std::milli>(t2 - t1).count();
        float inferMs = std::chrono::duration<float, std::milli>(t3 - t2).count();
        float trackMs = std::chrono::duration<float, std::milli>(t4 - t3).count();
        float physMs = std::chrono::duration<float, std::milli>(t5 - t4).count();
        float sendMs = std::chrono::duration<float, std::milli>(t6 - t5).count();
        float totalMs = std::chrono::duration<float, std::milli>(t6 - t0).count();

        csv << frameId << "," << decodeMs << "," << prepMs << "," << inferMs << "," << trackMs << "," << physMs << "," << sendMs << "," << totalMs << "\n";
    }

    csv.close();
    std::cout << "[Benchmark] Successfully logged 2,000 frames (incl. decode) to latency_benchmark.csv!" << std::endl;
    return 0;
}
