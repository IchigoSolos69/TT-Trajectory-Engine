#include "tensorrt_engine.hpp"
#include "bytetrack.hpp"
#include "telemetry_server.hpp"
#include "homography_calibrator.hpp"
#include "physics_engine.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>

int main(int argc, char** argv) {
    std::cout << "============================================================" << std::endl;
    std::cout << "   TABLE TENNIS TRAJECTORY ENGINE (C++ TensorRT Backend)   " << std::endl;
    std::cout << "============================================================" << std::endl;

    bool isLiveMode = false;
    std::string enginePath = "models/yolov11_ttgames.engine";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--live") {
            isLiveMode = true;
        } else if (arg == "--sim") {
            isLiveMode = false;
        } else if (arg.rfind(".engine") != std::string::npos) {
            enginePath = arg;
        }
    }

    if (isLiveMode) {
        std::cout << "[Main] LIVE MODE: Connecting to External Webcam..." << std::endl;
    } else {
        std::cout << "[Main] SIMULATION MODE: Loading dataset video..." << std::endl;
    }

    // 1. Initialize TensorRT Model Engine
    TensorRTEngine trtEngine(enginePath, 640, 640);
    bool initialized = trtEngine.init();

    if (!initialized) {
        std::cout << "[Main] Running in Standalone Simulation Mode (No TensorRT model loaded)." << std::endl;
    } else {
        std::cout << "[Main] TensorRT Engine initialized successfully!" << std::endl;
    }

    // 2. Initialize ByteTrack Object Tracker
    ByteTracker tracker(0.25f, 0.1f, 30);
    std::cout << "[Main] ByteTrack Object Tracker initialized (conf >= 0.25)." << std::endl;

    // 3. Initialize WebSocket Telemetry Server & Aerodynamic Physics Engine
    TelemetryServer server(9002);
    server.start();

    PhysicsEngine physics;
    std::cout << "[Main] Aerodynamic Physics Engine initialized (120 FPS gravity tick rate)." << std::endl;

    // 4. Video & Calibration Setup
    cv::VideoCapture cap;
    bool isCameraCurrentlyOpen = false;
    std::vector<cv::Point2f> tableCorners;
    cv::Mat H;
    HomographyCalibrator calibrator;

    if (!isLiveMode) {
        std::cout << "[Main] Loading dataset video file..." << std::endl;
        std::string videoPath = "../../python_tools/dataset/raw_videos/game_1.mp4";
        cap.open(videoPath);
        if (!cap.isOpened()) {
            videoPath = "../python_tools/dataset/raw_videos/game_1.mp4";
            cap.open(videoPath);
        }
        if (!cap.isOpened()) {
            videoPath = "python_tools/dataset/raw_videos/game_1.mp4";
            cap.open(videoPath);
        }

        if (cap.isOpened()) {
            std::cout << "[Main] OpenCV Video dataset loaded successfully." << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 2500); // Fast-forward dead time
        } else {
            std::cout << "[Main] Notice: Could not open video dataset." << std::endl;
            std::cout << "[Main] Falling back to high-speed simulation stream..." << std::endl;
        }
        calibrator.calibrateTable(100.0f, 100.0f, 540.0f, 100.0f, 540.0f, 540.0f, 100.0f, 540.0f);
    } else {
        std::cout << "[Main] LIVE MODE: C++ Engine Standby. Awaiting dashboard webcam authorization..." << std::endl;
        calibrator.calibrateTable(100.0f, 100.0f, 540.0f, 100.0f, 540.0f, 540.0f, 100.0f, 540.0f);
    }

    // 5. Main Inference Loop (120 FPS high-speed video capture & CUDA stream loop)
    std::cout << "[Main] Starting GPU Telemetry Loop..." << std::endl;

    int frameId = 0;
    const float targetFPS = 120.0f;
    const auto frameDuration = std::chrono::microseconds(static_cast<int>(1000000.0f / targetFPS));
    cv::Mat frame;
    std::string modeString = isLiveMode ? "live" : "simulation";
    auto lastTickTime = std::chrono::high_resolution_clock::now();

    while (true) {
        auto startTime = std::chrono::high_resolution_clock::now();
        float measuredDt = std::chrono::duration<float>(startTime - lastTickTime).count();
        lastTickTime = startTime;
        if (measuredDt <= 0.001f || measuredDt > 0.1f) {
            measuredDt = 1.0f / 120.0f;
        }

        if (isLiveMode) {
            // STATE 1: Dashboard authorized camera, but hardware is currently closed -> Open webcam & calibrate
            if (server.cameraAuthorized && !isCameraCurrentlyOpen) {
                std::cout << "[Hardware] Dashboard authorized. Booting webcam & searching for table bounds..." << std::endl;
                cap.open(1, cv::CAP_DSHOW);
                if (!cap.isOpened()) {
                    std::cout << "[Hardware] Index 1 webcam not found. Trying index 0..." << std::endl;
                    cap.open(0, cv::CAP_DSHOW);
                }

                if (cap.isOpened()) {
                    isCameraCurrentlyOpen = true;
                    std::cout << "[Hardware] SUCCESS: External webcam connected & hardware locked!" << std::endl;

                    // Sensor Warmup & Automatic HSV Table Calibration
                    cv::Mat calibFrame, hsv, mask;
                    for (int i = 0; i < 30; i++) {
                        cap.read(calibFrame);
                    }

                    if (!calibFrame.empty()) {
                        cv::cvtColor(calibFrame, hsv, cv::COLOR_BGR2HSV);
                        cv::inRange(hsv, cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255), mask);

                        std::vector<std::vector<cv::Point>> contours;
                        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                        double maxArea = 0;
                        std::vector<cv::Point> largestContour;
                        for (const auto& contour : contours) {
                            double area = cv::contourArea(contour);
                            if (area > maxArea) {
                                maxArea = area;
                                largestContour = contour;
                            }
                        }

                        std::vector<cv::Point> approx;
                        if (!largestContour.empty()) {
                            cv::approxPolyDP(largestContour, approx, 0.02 * cv::arcLength(largestContour, true), true);
                        }

                        if (approx.size() == 4) {
                            std::cout << "[Calibration] SUCCESS: 4 table corners automatically detected!" << std::endl;
                            tableCorners.clear();
                            for (int i = 0; i < 4; i++) {
                                tableCorners.push_back(cv::Point2f(static_cast<float>(approx[i].x), static_cast<float>(approx[i].y)));
                            }

                            std::sort(tableCorners.begin(), tableCorners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                                return a.y < b.y;
                            });
                            if (tableCorners[0].x > tableCorners[1].x) std::swap(tableCorners[0], tableCorners[1]);
                            if (tableCorners[2].x < tableCorners[3].x) std::swap(tableCorners[2], tableCorners[3]);

                            std::vector<cv::Point2f> realWorldCorners = {
                                cv::Point2f(0.0f, 0.0f),
                                cv::Point2f(1.525f, 0.0f),
                                cv::Point2f(1.525f, 2.74f),
                                cv::Point2f(0.0f, 2.74f)
                            };

                            H = cv::findHomography(tableCorners, realWorldCorners, cv::RANSAC);
                            calibrator.calibrateTable(
                                tableCorners[0].x, tableCorners[0].y,
                                tableCorners[1].x, tableCorners[1].y,
                                tableCorners[2].x, tableCorners[2].y,
                                tableCorners[3].x, tableCorners[3].y
                            );
                            std::cout << "[Calibration] Homography matrix established." << std::endl;
                        } else {
                            std::cout << "[Calibration] Using default ITTF perspective bounds." << std::endl;
                        }
                    }
                } else {
                    std::cout << "[Hardware] Error: Failed to open webcam device." << std::endl;
                    server.cameraAuthorized = false;
                }
            }
            // STATE 2: Dashboard revoked camera access (or closed tab) -> Release hardware lock instantly
            else if (!server.cameraAuthorized && isCameraCurrentlyOpen) {
                std::cout << "[Hardware] Access revoked. Releasing OpenCV webcam hardware lock." << std::endl;
                cap.release();
                isCameraCurrentlyOpen = false;
                physics.reset();
            }

            // STATE 3: Hardware active & frame acquired
            if (isCameraCurrentlyOpen && cap.read(frame) && !frame.empty()) {
                frameId++;
                auto t0 = std::chrono::high_resolution_clock::now();
                auto t1 = std::chrono::high_resolution_clock::now();
                std::vector<Detection> detections = trtEngine.inferFrameGPU(frame);

                auto t2 = std::chrono::high_resolution_clock::now();
                std::vector<TrackedObject> tracks = tracker.update(detections);

                auto t3 = std::chrono::high_resolution_clock::now();
                float realTimeX_meters = 0.7625f;
                float realTimeY_meters = 1.37f;
                float ballPixelX = -1.0f;
                float ballPixelY = -1.0f;
                bool hasBallDet = false;

                if (!detections.empty() && detections[0].class_id == 0 && detections[0].confidence >= 0.25f) {
                    ballPixelX = (detections[0].x + detections[0].w / 2.0f) * frame.cols;
                    ballPixelY = (detections[0].y + detections[0].h / 2.0f) * frame.rows;
                    auto realWorldMeters = calibrator.pixelToMeters(ballPixelX, ballPixelY);
                    realTimeX_meters = realWorldMeters.first;
                    realTimeY_meters = realWorldMeters.second;
                    hasBallDet = true;
                }

                if (!hasBallDet) {
                    physics.reset();
                }

                PhysicsState pState = physics.update(realTimeX_meters, realTimeY_meters, measuredDt, ballPixelX, ballPixelY);
                auto t4 = std::chrono::high_resolution_clock::now();
                server.broadcastFrame(frameId, tracks, targetFPS, modeString, &pState);
                auto t5 = std::chrono::high_resolution_clock::now();

                if (frameId % 60 == 0) {
                    float prepMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    float inferMs = std::chrono::duration<float, std::milli>(t2 - t1).count();
                    float trackMs = std::chrono::duration<float, std::milli>(t3 - t2).count();
                    float physMs = std::chrono::duration<float, std::milli>(t4 - t3).count();
                    float sendMs = std::chrono::duration<float, std::milli>(t5 - t4).count();
                    float totalMs = std::chrono::duration<float, std::milli>(t5 - t0).count();

                    std::cout << "[Main] LIVE Frame #" << frameId 
                              << " | Latency: " << totalMs << "ms (Prep: " << prepMs 
                              << "ms, Infer: " << inferMs << "ms, Track: " << trackMs 
                              << "ms, Phys: " << physMs << "ms, WS: " << sendMs << "ms)"
                              << " | Speed: " << pState.velocityKmh << " km/h | Z: " << pState.zHeight << "m" << std::endl;
                }
            } else {
                // Engine is idle or standby: Broadcast idle telemetry & sleep 50ms to keep CPU at 0%
                physics.reset();
                std::vector<TrackedObject> emptyTracks;
                server.broadcastFrame(frameId, emptyTracks, targetFPS, modeString);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
        } 
        else {
            // Simulation Mode (Dataset Video Replay)
            frameId++;
            std::vector<Detection> detections;
            if (cap.isOpened()) {
                if (!cap.read(frame) || frame.empty()) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    if (!cap.read(frame)) {
                        std::cout << "[Main] End of video stream." << std::endl;
                        break;
                    }
                }
                detections = trtEngine.inferFrameGPU(frame);
            } else {
                cv::Mat mockFrame(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
                detections = trtEngine.inferFrameGPU(mockFrame);
            }

            std::vector<TrackedObject> tracks = tracker.update(detections);
            float realTimeX_meters = 0.7625f;
            float realTimeY_meters = 1.37f;
            float ballPixelX = -1.0f;
            float ballPixelY = -1.0f;
            bool hasBallDet = false;

            if (!detections.empty() && detections[0].class_id == 0 && detections[0].confidence >= 0.25f) {
                ballPixelX = (detections[0].x + detections[0].w / 2.0f) * frame.cols;
                ballPixelY = (detections[0].y + detections[0].h / 2.0f) * frame.rows;
                auto realWorldMeters = calibrator.pixelToMeters(ballPixelX, ballPixelY);
                realTimeX_meters = realWorldMeters.first;
                realTimeY_meters = realWorldMeters.second;
                hasBallDet = true;
            }

            if (!hasBallDet) {
                physics.reset();
            }

            float simDt = measuredDt;
            if (cap.isOpened()) {
                double videoFps = cap.get(cv::CAP_PROP_FPS);
                if (videoFps > 10.0 && videoFps <= 240.0) {
                    simDt = 1.0f / static_cast<float>(videoFps);
                }
            }

            PhysicsState pState = physics.update(realTimeX_meters, realTimeY_meters, simDt, ballPixelX, ballPixelY);
            server.broadcastFrame(frameId, tracks, targetFPS, modeString, &pState);

            if (frameId % 60 == 0) {
                std::cout << "[Main] SIM Frame #" << frameId 
                          << " | Tracks: " << tracks.size() 
                          << " | Speed: " << pState.velocityKmh << " km/h | Z: " << pState.zHeight << "m"
                          << " | Telemetry (" << modeString << ") broadcasted at " << (1.0f / simDt) << " FPS." << std::endl;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        if (elapsed < frameDuration) {
            std::this_thread::sleep_for(frameDuration - elapsed);
        }
    }

    return 0;
}
