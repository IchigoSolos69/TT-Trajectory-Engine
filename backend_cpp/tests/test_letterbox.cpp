#include "tensorrt_engine.hpp"
#include <cmath>
#include <stdexcept>

int main() try {
    for (const auto& size : {cv::Size(1920,1080), cv::Size(1280,721), cv::Size(721,1280), cv::Size(641,479)}) {
        cv::Mat frame(size, CV_8UC3, cv::Scalar(0, 0, 255));
        auto input = TensorRTEngine::preprocessOpenCVFrame(frame);
        if (input.size() != 3 * 640 * 640 || input[320 * 640 + 320] != 1.0f)
            throw std::runtime_error("Letterbox lost CHW data for odd input geometry");
        float scale = std::min(640.0f / size.width, 640.0f / size.height);
        int left = (640 - static_cast<int>(std::round(size.width * scale))) / 2;
        int top = (640 - static_cast<int>(std::round(size.height * scale))) / 2;
        std::vector<float> raw(5 * 8400, 0);
        raw[0] = 123 * scale + left;
        raw[8400] = 234 * scale + top;
        raw[2 * 8400] = raw[3 * 8400] = 10;
        raw[4 * 8400] = .9f;
        auto dets = TensorRTEngine::postProcess(raw, size.width, size.height);
        if (dets.size() != 1 || std::abs((dets[0].x + dets[0].w/2)*size.width - 123) > .001f ||
            std::abs((dets[0].y + dets[0].h/2)*size.height - 234) > .001f)
            throw std::runtime_error("Letterbox coordinate round trip failed");
        std::cout << "PASS " << size.width << 'x' << size.height << " CHW and coordinate round trip\n";
    }
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
