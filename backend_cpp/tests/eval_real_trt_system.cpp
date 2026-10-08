#include "tensorrt_engine.hpp"
#include "bytetrack.hpp"
#include "physics_engine.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>

int main() {
    std::string videoPath = "python_tools/dataset/raw_videos/test_1.mp4";
    std::string ballJsonPath = "python_tools/dataset/raw_markup/test_1/ball_markup.json";
    std::string eventsJsonPath = "python_tools/dataset/raw_markup/test_1/events_markup.json";
    std::string modelPath = "models/yolov11_ttgames.engine";

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open dataset video: " << videoPath << std::endl;
        return 1;
    }

    float imgW = static_cast<float>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    float imgH = static_cast<float>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "=========================================================================" << std::endl;
    std::cout << "  Authoritative C++ TensorRT System Evaluation (Full " << totalFrames << " Frames)" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    // Load GT Ball Markup (880 annotated ball frames)
    std::map<int, std::pair<float, float>> gtBallMap;
    std::ifstream bFile(ballJsonPath);
    if (bFile.is_open()) {
        std::string content((std::istreambuf_iterator<char>(bFile)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos) {
            size_t endKey = content.find("\"", pos + 1);
            if (endKey == std::string::npos) break;
            std::string key = content.substr(pos + 1, endKey - pos - 1);
            pos = endKey + 1;

            if (std::all_of(key.begin(), key.end(), ::isdigit)) {
                size_t xPos = content.find("\"x\":", pos);
                if (xPos != std::string::npos && xPos < pos + 50) {
                    size_t valStart = content.find_first_of("0123456789-.", xPos + 4);
                    size_t valEnd = content.find_first_not_of("0123456789-.", valStart);
                    float xVal = std::stof(content.substr(valStart, valEnd - valStart));

                    size_t yPos = content.find("\"y\":", valEnd);
                    size_t yValStart = content.find_first_of("0123456789-.", yPos + 4);
                    size_t yValEnd = content.find_first_not_of("0123456789-.", yValStart);
                    float yVal = std::stof(content.substr(yValStart, yValEnd - yValStart));

                    if (xVal >= 0 && yVal >= 0) {
                        gtBallMap[std::stoi(key)] = {xVal, yVal};
                    }
                }
            }
        }
    }

    // Load GT Bounce Events (31 bounces)
    std::vector<int> gtBounces;
    std::ifstream eFile(eventsJsonPath);
    if (eFile.is_open()) {
        std::string content((std::istreambuf_iterator<char>(eFile)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos) {
            size_t endKey = content.find("\"", pos + 1);
            if (endKey == std::string::npos) break;
            std::string key = content.substr(pos + 1, endKey - pos - 1);
            pos = endKey + 1;

            if (std::all_of(key.begin(), key.end(), ::isdigit)) {
                size_t valStart = content.find("\"", pos);
                if (valStart != std::string::npos && valStart < pos + 30) {
                    size_t valEnd = content.find("\"", valStart + 1);
                    std::string valStr = content.substr(valStart + 1, valEnd - valStart - 1);
                    pos = valEnd + 1;

                    if (valStr == "bounce") {
                        gtBounces.push_back(std::stoi(key));
                    }
                }
            }
        }
        std::sort(gtBounces.begin(), gtBounces.end());
    }

    std::cout << "Loaded " << totalFrames << " frames | GT Ball Frames: " << gtBallMap.size() << " | GT Bounces (Strict): " << gtBounces.size() << std::endl;

    TensorRTEngine trtEngine(modelPath);
    bool trtOk = trtEngine.init();

    ByteTracker trackerA_025(0.25f, 0.10f, 30);
    ByteTracker trackerA_015(0.15f, 0.08f, 30);

    PhysicsEngine physicsA_025;
    PhysicsEngine physicsA_015;
    PhysicsEngine physicsB_025;
    PhysicsEngine physicsB_015;

    float dt = 1.0f / 120.0f;

    std::vector<int> detectedBouncesA_025;
    std::vector<int> detectedBouncesA_015;
    std::vector<int> detectedBouncesB_025;
    std::vector<int> detectedBouncesB_015;

    std::vector<float> locErrorsPx;
    int ballTP_5 = 0, ballTP_10 = 0, ballTP_15 = 0;
    int ballFN_5 = 0, ballFN_10 = 0, ballFN_15 = 0;
    int totalDetectionsOnAnnotated = 0;

    cv::Mat frame;
    int frameId = 0;

    struct FrameDets {
        float topConfX = -1.0f, topConfY = -1.0f, topConf = 0.0f;
        float closestX = -1.0f, closestY = -1.0f, closestErr = 9999.0f;
        float trackX = -1.0f, trackY = -1.0f;
        std::vector<Detection> allDets;
    };
    std::map<int, FrameDets> frameDetsMap;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frameId++;

        FrameDets fDets;
        float detPxX = -1.0f, detPxY = -1.0f;

        if (trtOk) {
            InferenceTimingDetailed inferTiming;
            std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);

            // Infer candidates down to 0.15f threshold
            std::vector<Detection> detections015 = trtEngine.inferFrameDetailed(inputBuffer, inferTiming, static_cast<int>(imgW), static_cast<int>(imgH), 0.15f);
            std::vector<Detection> detections025;
            for (const auto& d : detections015) {
                if (d.confidence >= 0.25f) detections025.push_back(d);
            }

            fDets.allDets = detections015;

            if (!detections015.empty()) {
                const auto& bestDet = detections015[0];
                detPxX = (bestDet.x + bestDet.w / 2.0f) * imgW;
                detPxY = (bestDet.y + bestDet.h / 2.0f) * imgH;
                fDets.topConfX = detPxX;
                fDets.topConfY = detPxY;
                fDets.topConf = bestDet.confidence;
            }

            // 1. Mode A @ 0.25 Tracker Update
            std::vector<TrackedObject> tracksA_025 = trackerA_025.update(detections025);
            bool hasTrkA_025 = !tracksA_025.empty() && tracksA_025[0].is_active;
            float trkX_025 = hasTrkA_025 ? (tracksA_025[0].x + tracksA_025[0].w / 2.0f) * imgW : -1.0f;
            float trkY_025 = hasTrkA_025 ? (tracksA_025[0].y + tracksA_025[0].h / 2.0f) * imgH : -1.0f;
            fDets.trackX = trkX_025;
            fDets.trackY = trkY_025;

            float mxA_025 = (trkX_025 / imgW) * 1.525f;
            float myA_025 = (trkY_025 / imgH) * 2.74f;
            PhysicsState pStateA_025 = physicsA_025.update(mxA_025, myA_025, dt, trkX_025, trkY_025);
            if (pStateA_025.isTableBounce) detectedBouncesA_025.push_back(frameId);

            // 2. Mode A @ 0.15 Tracker Update
            std::vector<TrackedObject> tracksA_015 = trackerA_015.update(detections015);
            bool hasTrkA_015 = !tracksA_015.empty() && tracksA_015[0].is_active;
            float trkX_015 = hasTrkA_015 ? (tracksA_015[0].x + tracksA_015[0].w / 2.0f) * imgW : -1.0f;
            float trkY_015 = hasTrkA_015 ? (tracksA_015[0].y + tracksA_015[0].h / 2.0f) * imgH : -1.0f;

            float mxA_015 = (trkX_015 / imgW) * 1.525f;
            float myA_015 = (trkY_015 / imgH) * 2.74f;
            PhysicsState pStateA_015 = physicsA_015.update(mxA_015, myA_015, dt, trkX_015, trkY_015);
            if (pStateA_015.isTableBounce) detectedBouncesA_015.push_back(frameId);

            // 3. Mode B @ 0.25 Raw Feed Update
            bool isRawValid025 = (!detections025.empty());
            float rx025 = isRawValid025 ? (detections025[0].x + detections025[0].w / 2.0f) * imgW : -1.0f;
            float ry025 = isRawValid025 ? (detections025[0].y + detections025[0].h / 2.0f) * imgH : -1.0f;
            float mxB_025 = (rx025 / imgW) * 1.525f;
            float myB_025 = (ry025 / imgH) * 2.74f;
            PhysicsState pStateB_025 = physicsB_025.update(mxB_025, myB_025, dt, rx025, ry025);
            if (pStateB_025.isTableBounce) detectedBouncesB_025.push_back(frameId);

            // 4. Mode B @ 0.15 Raw Feed Update
            bool isRawValid015 = (!detections015.empty());
            float rx015 = isRawValid015 ? (detections015[0].x + detections015[0].w / 2.0f) * imgW : -1.0f;
            float ry015 = isRawValid015 ? (detections015[0].y + detections015[0].h / 2.0f) * imgH : -1.0f;
            float mxB_015 = (rx015 / imgW) * 1.525f;
            float myB_015 = (ry015 / imgH) * 2.74f;
            PhysicsState pStateB_015 = physicsB_015.update(mxB_015, myB_015, dt, rx015, ry015);
            if (pStateB_015.isTableBounce) detectedBouncesB_015.push_back(frameId);
        }
    }

    // Frame-Shift Sweep for offsets -3 to +3
    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "--- FRAME-SHIFT SWEEP (-3 to +3 Offsets on C++ TensorRT Path) ---" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    for (int off = -3; off <= 3; ++off) {
        int h5 = 0, h10 = 0, h15 = 0, total = 0;
        std::vector<float> errs;
        for (auto const& [gtF, gtPt] : gtBallMap) {
            int targetF = gtF + off;
            total++;
            auto it = frameDetsMap.find(targetF);
            if (it != frameDetsMap.end() && it->second.topConfX >= 0) {
                float d = std::sqrt((it->second.topConfX - gtPt.first)*(it->second.topConfX - gtPt.first) + (it->second.topConfY - gtPt.second)*(it->second.topConfY - gtPt.second));
                errs.push_back(d);
                if (d <= 5.0f) h5++;
                if (d <= 10.0f) h10++;
                if (d <= 15.0f) h15++;
            }
        }
        std::sort(errs.begin(), errs.end());
        float med = errs.empty() ? 0.0f : errs[errs.size() / 2];
        std::cout << "  Offset " << (off >= 0 ? "+" : "") << off
                  << " | Hit <=5px: " << (float)h5/total*100.0f << "%"
                  << " | Hit <=10px: " << (float)h10/total*100.0f << "%"
                  << " | Hit <=15px: " << (float)h15/total*100.0f << "%"
                  << " | Median Err: " << med << " px" << std::endl;
    }

    auto evalBounces = [&](const std::vector<int>& detBounces, const std::string& modeName) {
        int bounceTP = 0, bounceFP = 0;
        std::set<int> matchedGt;

        for (int det : detBounces) {
            bool match = false;
            for (size_t i = 0; i < gtBounces.size(); ++i) {
                int realG = gtBounces[i];
                if (std::abs(det - realG) <= 3 && matchedGt.find(realG) == matchedGt.end()) {
                    bounceTP++;
                    matchedGt.insert(realG);
                    match = true;
                    break;
                }
            }
            if (!match) bounceFP++;
        }

        int bounceFN = static_cast<int>(gtBounces.size()) - static_cast<int>(matchedGt.size());
        float bouncePrec = (bounceTP + bounceFP > 0) ? (float)bounceTP / (bounceTP + bounceFP) : 0.0f;
        float bounceRec  = (bounceTP + bounceFN > 0) ? (float)bounceTP / (bounceTP + bounceFN) : 0.0f;
        float bounceF1   = (bouncePrec + bounceRec > 0) ? 2.0f * bouncePrec * bounceRec / (bouncePrec + bounceRec) : 0.0f;

        std::cout << "--- " << modeName << " ---" << std::endl;
        std::cout << "  Total Detected Bounces Clip-Wide : " << detBounces.size() << std::endl;
        std::cout << "  True Positive Bounces (TP)       : " << bounceTP << std::endl;
        std::cout << "  False Positive Bounces (FP)      : " << bounceFP << std::endl;
        std::cout << "  False Negative Bounces (FN)      : " << bounceFN << std::endl;
        std::cout << "  System Bounce Precision          : " << bouncePrec * 100.0f << "%" << std::endl;
        std::cout << "  System Bounce Recall             : " << bounceRec * 100.0f << "%" << std::endl;
        std::cout << "  System Bounce F1 Score            : " << bounceF1 * 100.0f << "%\n" << std::endl;
    };

    std::sort(locErrorsPx.begin(), locErrorsPx.end());
    float minLoc = locErrorsPx.empty() ? 0.0f : locErrorsPx.front();
    float p50Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 50 / 100];
    float p90Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 90 / 100];
    float p95Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 95 / 100];
    float p99Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 99 / 100];
    float maxLoc = locErrorsPx.empty() ? 0.0f : locErrorsPx.back();

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "--- DEPLOYED TENSORRT ENGINE DETECTOR METRICS (test_1: " << gtBallMap.size() << " GT Ball Frames) ---" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  Total Annotated GT Ball Frames   : " << gtBallMap.size() << std::endl;
    std::cout << "  Detections on Annotated Frames   : " << totalDetectionsOnAnnotated << " / " << gtBallMap.size() << " (" << (float)totalDetectionsOnAnnotated / gtBallMap.size() * 100.0f << "%)" << std::endl;
    std::cout << "  - Threshold <=  5.0 px : TP = " << ballTP_5 << " | FN = " << ballFN_5 << " | Hit Rate = " << (gtBallMap.size() > 0 ? (float)ballTP_5 / gtBallMap.size() * 100.0f : 0.0f) << "%" << std::endl;
    std::cout << "  - Threshold <= 10.0 px : TP = " << ballTP_10 << " | FN = " << ballFN_10 << " | Hit Rate = " << (gtBallMap.size() > 0 ? (float)ballTP_10 / gtBallMap.size() * 100.0f : 0.0f) << "%" << std::endl;
    std::cout << "  - Threshold <= 15.0 px : TP = " << ballTP_15 << " | FN = " << ballFN_15 << " | Hit Rate = " << (gtBallMap.size() > 0 ? (float)ballTP_15 / gtBallMap.size() * 100.0f : 0.0f) << "%" << std::endl;
    std::cout << "  Localization Error Distribution (pixels):" << std::endl;
    std::cout << "    Min    : " << minLoc << " px" << std::endl;
    std::cout << "    p50    : " << p50Loc << " px (Median)" << std::endl;
    std::cout << "    p90    : " << p90Loc << " px" << std::endl;
    std::cout << "    p95    : " << p95Loc << " px" << std::endl;
    std::cout << "    p99    : " << p99Loc << " px" << std::endl;
    std::cout << "    Max    : " << maxLoc << " px" << std::endl;

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "--- 4-WAY SYSTEM BOUNCE METRICS COMPARISON (Full 16,800 Frames) ---" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    evalBounces(detectedBouncesA_025, "MODE A @ 0.25 (ByteTrack Tracked Feed @ Conf >= 0.25)");
    evalBounces(detectedBouncesA_015, "MODE A @ 0.15 (ByteTrack Tracked Feed @ Conf >= 0.15)");
    evalBounces(detectedBouncesB_025, "MODE B @ 0.25 (Raw Gated Feed @ Conf >= 0.25)");
    evalBounces(detectedBouncesB_015, "MODE B @ 0.15 (Raw Gated Feed @ Conf >= 0.15)");

    // Export list of detected FP bounce frames for instant Python overlay rendering
    std::ofstream fpListOut("python_tools/fp_bounces_list.json");
    fpListOut << "{\n  \"raw_bounces\": [";
    for (size_t i = 0; i < detectedBouncesB_025.size(); ++i) {
        fpListOut << detectedBouncesB_025[i] << (i + 1 < detectedBouncesB_025.size() ? ", " : "");
    }
    fpListOut << "],\n  \"track_bounces\": [";
    for (size_t i = 0; i < detectedBouncesA_025.size(); ++i) {
        fpListOut << detectedBouncesA_025[i] << (i + 1 < detectedBouncesA_025.size() ? ", " : "");
    }
    fpListOut << "]\n}\n";
    fpListOut.close();

    return 0;
}
