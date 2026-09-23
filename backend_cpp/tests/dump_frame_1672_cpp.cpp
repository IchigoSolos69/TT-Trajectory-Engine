#include "tensorrt_engine.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

int main() {
    std::string videoPath = "python_tools/dataset/raw_videos/test_1.mp4";
    std::string modelPath = "models/yolov11_ttgames.engine";

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video" << std::endl;
        return 1;
    }

    cv::Mat frame;
    int frameId = 0;
    while (cap.read(frame)) {
        frameId++;
        if (frameId == 1672) break;
    }

    TensorRTEngine trtEngine(modelPath);
    if (!trtEngine.init()) {
        std::cerr << "Cannot init engine" << std::endl;
        return 1;
    }

    std::vector<float> inputBuffer = TensorRTEngine::preprocessOpenCVFrame(frame, 640, 640);
    
    // Perform detailed inference to inspect raw output
    std::vector<float> rawOutput(84 * 8400, 0.0f);
    trtEngine.executeInference(inputBuffer, rawOutput);

    // Sort all 8400 anchors by confidence
    struct AnchorInfo {
        int idx;
        float cx, cy, w, h, conf;
    };
    std::vector<AnchorInfo> anchors;
    for (int i = 0; i < 8400; ++i) {
        float conf = rawOutput[4 * 8400 + i];
        if (conf > 0.15f) {
            float cx = rawOutput[0 * 8400 + i];
            float cy = rawOutput[1 * 8400 + i];
            float w  = rawOutput[2 * 8400 + i];
            float h  = rawOutput[3 * 8400 + i];
            anchors.push_back({i, cx, cy, w, h, conf});
        }
    }

    std::sort(anchors.begin(), anchors.end(), [](const AnchorInfo& a, const AnchorInfo& b) {
        return a.conf > b.conf;
    });

    float scaleR = 1.0f / 3.0f;
    float padH = 140.0f;
    float gt_x = 797.0f, gt_y = 504.0f;

    std::cout << "=========================================================================" << std::endl;
    std::cout << "  C++ TENSORRT RAW TOP ANCHORS FOR FRAME 1672 (Letterboxed Input)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Rank | AnchorIdx | Raw 640x640 (cx, cy, w, h)     | Conf   | Unletterboxed 1920x1080 (X, Y) | Err to GT (797, 504)" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------------------" << std::endl;

    for (size_t rank = 0; rank < std::min((size_t)5, anchors.size()); ++rank) {
        const auto& a = anchors[rank];
        float pxX = (a.cx - 0.0f) / scaleR;
        float pxY = (a.cy - padH) / scaleR;
        float err = std::sqrt((pxX - gt_x) * (pxX - gt_x) + (pxY - gt_y) * (pxY - gt_y));
        std::cout << "  " << (rank + 1) << "  | " << a.idx << " | (" << a.cx << ", " << a.cy << ", " << a.w << ", " << a.h << ") | " << a.conf << " | (" << pxX << ", " << pxY << ") | " << err << " px" << std::endl;
    }

    return 0;
}
