#include "tensorrt_engine.hpp"
#include "bytetrack.hpp"
#include "physics_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

struct BallGT {
    int frameId;
    float x;
    float y;
};

int main(int argc, char** argv) {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  Authoritative C++ TensorRT FP16 System Evaluation (Full 16,800 Frames)" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    std::string videoPath = "python_tools/dataset/raw_videos/test_1.mp4";
    std::string ballJsonPath = "python_tools/dataset/raw_markup/test_1/ball_markup.json";
    std::string eventsJsonPath = "python_tools/dataset/raw_markup/test_1/events_markup.json";
    std::string modelPath = "models/yolov11_ttgames.engine";

    if (argc > 1) videoPath = argv[1];

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video " << videoPath << std::endl;
        return 1;
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    float imgW = static_cast<float>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    float imgH = static_cast<float>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Load GT Ball Markups
    std::map<int, std::pair<float, float>> gtBallMap;
    std::ifstream ballFile(ballJsonPath);
    if (ballFile.is_open()) {
        std::stringstream buffer;
        buffer << ballFile.rdbuf();
        std::string content = buffer.str();
        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos) {
            size_t endFramePos = content.find("\"", pos + 1);
            if (endFramePos == std::string::npos) break;
            std::string frameStr = content.substr(pos + 1, endFramePos - pos - 1);
            pos = endFramePos + 1;
            if (frameStr == "x" || frameStr == "y" || frameStr == "status") continue;
            try {
                int frameIdx = std::stoi(frameStr);
                size_t xPos = content.find("\"x\"", pos);
                if (xPos == std::string::npos || xPos > pos + 100) continue;
                size_t xValStart = content.find(":", xPos) + 1;
                size_t xValEnd = content.find_first_of(",}", xValStart);
                float xVal = std::stof(content.substr(xValStart, xValEnd - xValStart));

                size_t yPos = content.find("\"y\"", pos);
                if (yPos == std::string::npos || yPos > pos + 100) continue;
                size_t yValStart = content.find(":", yPos) + 1;
                size_t yValEnd = content.find_first_of(",}", yValStart);
                float yVal = std::stof(content.substr(yValStart, yValEnd - yValStart));

                if (xVal >= 0 && yVal >= 0) {
                    gtBallMap[frameIdx] = {xVal, yVal};
                }
            } catch (...) {
                continue;
            }
        }
    }

    // Load GT Bounce Events (Filter strictly to "bounce" events)
    std::vector<int> gtBounces;
    std::ifstream eventsFile(eventsJsonPath);
    if (eventsFile.is_open()) {
        std::stringstream buffer;
        buffer << eventsFile.rdbuf();
        std::string content = buffer.str();
        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos) {
            size_t endPos = content.find("\"", pos + 1);
            if (endPos == std::string::npos) break;
            std::string key = content.substr(pos + 1, endPos - pos - 1);
            pos = endPos + 1;
            
            // Look for value matching "bounce" strictly
            size_t valStart = content.find("\"", pos);
            if (valStart == std::string::npos || valStart > pos + 30) continue;
            size_t valEnd = content.find("\"", valStart + 1);
            std::string valStr = content.substr(valStart + 1, valEnd - valStart - 1);
            pos = valEnd + 1;

            if (valStr == "bounce") {
                try {
                    gtBounces.push_back(std::stoi(key));
                } catch (...) {}
            }
        }
        std::sort(gtBounces.begin(), gtBounces.end());
    }

    std::cout << "Loaded " << totalFrames << " frames | GT Ball Frames: " << gtBallMap.size() << " | GT Bounces (Strict): " << gtBounces.size() << std::endl;

    TensorRTEngine trtEngine(modelPath);
    bool trtOk = trtEngine.init();

    ByteTracker tracker(0.25f, 0.1f, 30);
    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;

    std::vector<int> detectedBounceFrames;
    int ballTP = 0, ballFN = 0;
    std::vector<float> locErrorsPx;

    cv::Mat frame;
    int frameId = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frameId++;

        float detPxX = -1.0f, detPxY = -1.0f;

        if (trtOk) {
            std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);
            std::vector<Detection> detections = trtEngine.inferFrame(inputBuffer);
            std::vector<TrackedObject> tracks = tracker.update(detections);

            for (const auto& trk : tracks) {
                if (trk.class_id == 0) {
                    detPxX = (trk.x + trk.w / 2.0f) * imgW;
                    detPxY = (trk.y + trk.h / 2.0f) * imgH;
                    break;
                }
            }
        }

        // Ball Detector Level Evaluation on Annotated Frames ONLY
        auto gtIt = gtBallMap.find(frameId);
        if (gtIt != gtBallMap.end()) {
            float gtX = gtIt->second.first;
            float gtY = gtIt->second.second;

            if (detPxX >= 0 && detPxY >= 0) {
                float dist = std::sqrt((detPxX - gtX) * (detPxX - gtX) + (detPxY - gtY) * (detPxY - gtY));
                locErrorsPx.push_back(dist);

                if (dist <= 15.0f) {
                    ballTP++;
                }
            } else {
                ballFN++;
            }
        }

        // System Physics Engine Update across ALL 16,800 frames
        float metersX = (detPxX / imgW) * 1.525f;
        float metersY = (detPxY / imgH) * 2.74f;
        PhysicsState pState = physics.update(metersX, metersY, dt, detPxX, detPxY);

        if (pState.isTableBounce) {
            detectedBounceFrames.push_back(frameId);
        }
    }

    // System Bounce Scoring across ALL 16,800 frames against 31 GT Bounces (+/- 3 frames tolerance)
    int bounceTP = 0, bounceFP = 0;
    std::set<int> matchedGt;

    for (int det : detectedBounceFrames) {
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
        if (!match) {
            bounceFP++;
        }
    }

    int bounceFN = static_cast<int>(gtBounces.size()) - static_cast<int>(matchedGt.size());
    float bouncePrec = (bounceTP + bounceFP > 0) ? (float)bounceTP / (bounceTP + bounceFP) : 0.0f;
    float bounceRec  = (bounceTP + bounceFN > 0) ? (float)bounceTP / (bounceTP + bounceFN) : 0.0f;
    float bounceF1   = (bouncePrec + bounceRec > 0) ? 2.0f * bouncePrec * bounceRec / (bouncePrec + bounceRec) : 0.0f;

    std::sort(locErrorsPx.begin(), locErrorsPx.end());
    float p50Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 50 / 100];
    float p90Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 90 / 100];
    float p95Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 95 / 100];
    float p99Loc = locErrorsPx.empty() ? 0.0f : locErrorsPx[locErrorsPx.size() * 99 / 100];

    std::cout << "\n--- DEPLOYED TENSORRT FP16 ENGINE DETECTOR METRICS (test_1: 880 GT Ball Frames) ---" << std::endl;
    std::cout << "  Center Distance Match Threshold : <= 15.0 pixels" << std::endl;
    std::cout << "  Annotated Frames Detected (TP)  : " << ballTP << " / " << gtBallMap.size() << std::endl;
    std::cout << "  Missed Ball Frames (FN)         : " << ballFN << std::endl;
    std::cout << "  Detector Raw Recall             : " << (gtBallMap.size() > 0 ? (float)ballTP / gtBallMap.size() * 100.0f : 0.0f) << "%" << std::endl;
    std::cout << "  Localization Error p50 (Median) : " << p50Loc << " pixels" << std::endl;
    std::cout << "  Localization Error p90          : " << p90Loc << " pixels" << std::endl;
    std::cout << "  Localization Error p95          : " << p95Loc << " pixels" << std::endl;
    std::cout << "  Localization Error p99          : " << p99Loc << " pixels" << std::endl;

    std::cout << "\n--- DEPLOYED TENSORRT FP16 SYSTEM BOUNCE METRICS (Full 16,800 Frames) ---" << std::endl;
    std::cout << "  Ground Truth Bounces (GT)        : " << gtBounces.size() << std::endl;
    std::cout << "  Total Detected Bounces Clip-Wide : " << detectedBounceFrames.size() << std::endl;
    std::cout << "  True Positive Bounces (TP)       : " << bounceTP << std::endl;
    std::cout << "  False Positive Bounces (FP)      : " << bounceFP << " (all non-GT detected bounces)" << std::endl;
    std::cout << "  False Negative Bounces (FN)      : " << bounceFN << std::endl;
    std::cout << "  System Bounce Precision          : " << bouncePrec * 100.0f << "%" << std::endl;
    std::cout << "  System Bounce Recall             : " << bounceRec * 100.0f << "%" << std::endl;
    std::cout << "  System Bounce F1 Score            : " << bounceF1 * 100.0f << "%" << std::endl;
    std::cout << "=========================================================================\n" << std::endl;

    return 0;
}
