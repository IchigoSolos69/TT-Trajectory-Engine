#include "tensorrt_engine.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

// One row per decoded source frame; evaluation labels are loaded separately.
int main(int argc, char** argv) try {
    if (argc < 4 || argc > 6) throw std::runtime_error("Usage: dump_detector_predictions ENGINE VIDEO OUTPUT.csv [FIRST_FRAME [LAST_FRAME]]");
    if (!std::ifstream(argv[1]).good()) throw std::runtime_error("Engine file missing");
    cv::VideoCapture cap(argv[2], cv::CAP_FFMPEG);
    if (!cap.isOpened()) throw std::runtime_error("Video unavailable");
    TensorRTEngine engine(argv[1]);
    if (!engine.init()) throw std::runtime_error("Detector unavailable");
    std::ofstream out(argv[3]);
    if (!out) throw std::runtime_error("Cannot write output CSV");
    out << "frame,x,y,confidence,detections,width,height\n" << std::setprecision(9);
    cv::Mat frame;
    int count = argc >= 5 ? std::stoi(argv[4]) - 1 : 0;
    const int finalFrame = argc == 6 ? std::stoi(argv[5]) : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int firstOffset = count;
    if (count < 0 || finalFrame <= count || finalFrame > cap.get(cv::CAP_PROP_FRAME_COUNT) ||
        (count > 0 && !cap.set(cv::CAP_PROP_POS_FRAMES, count)))
        throw std::runtime_error("Invalid starting frame");
    while (count < finalFrame && cap.read(frame)) {
        ++count;
        auto input = TensorRTEngine::preprocessOpenCVFrame(frame);
        auto detections = engine.inferFrame(input, frame.cols, frame.rows, 0.001f);
        if (detections.empty()) out << count << ",-1,-1,0,0,0,0\n";
        else {
            const auto& d = detections.front();
            out << count << ',' << (d.x + d.w / 2) * frame.cols << ','
                << (d.y + d.h / 2) * frame.rows << ',' << d.confidence << ',' << detections.size() << ',' << d.w * frame.cols << ',' << d.h * frame.rows << '\n';
        }
        if (count % 2000 == 0) std::cout << "Decoded/inferred " << count << " frames\n";
    }
    if (count != finalFrame)
        throw std::runtime_error("Incomplete video decode");
    if (!out.good()) throw std::runtime_error("CSV write failed");
    std::cout << "COMPLETE: " << count - firstOffset << " decoded frames; final source frame " << count
              << ". Engine SHA256=" << engine.getSha256Hash() << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
