#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  OPENCV CUDA / NVDEC MODULE CAPABILITY CHECK" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  OpenCV Version : " << CV_VERSION << std::endl;
    std::cout << "  CUDA Device Count: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

#ifdef HAVE_OPENCV_CUDACODEC
    std::cout << "  HAVE_OPENCV_CUDACODEC macro: ENABLED" << std::endl;
#else
    std::cout << "  HAVE_OPENCV_CUDACODEC macro: DISABLED / NOT COMPILED" << std::endl;
#endif

    std::string buildInfo = cv::getBuildInformation();
    size_t nvidiaPos = buildInfo.find("NVIDIA CUDA");
    if (nvidiaPos != std::string::npos) {
        std::cout << "\n--- OpenCV Build Info (CUDA Section) ---" << std::endl;
        std::cout << buildInfo.substr(nvidiaPos, 400) << std::endl;
    } else {
        std::cout << "\n--- OpenCV Build Info ---" << std::endl;
        std::cout << "NVIDIA CUDA section not found in OpenCV build info." << std::endl;
    }

    return 0;
}
