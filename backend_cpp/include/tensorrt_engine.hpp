#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>



// Detection Box Structure for YOLO output
struct Detection {
    int class_id;        // 0: Ball, 1: Player Near, 2: Player Far, 3: Table
    float confidence;
    float x, y, w, h;    // Normalized coordinates (0..1)
};

class TensorRTEngine {
public:
    TensorRTEngine(const std::string& enginePath = "", int inputWidth = 640, int inputHeight = 640);
    ~TensorRTEngine();

    bool init();
    std::vector<Detection> inferFrame(const std::vector<float>& rgbNormalizedBuffer);
    
    // Resolve canonical engine path (CLI arg -> TT_ENGINE_PATH env var -> "models/yolov11_ttgames.engine")
    static std::string resolveEnginePath(const std::string& userPath = "");

    // Metadata getters
    std::string getEnginePath() const { return enginePath_; }
    size_t getFileSizeBytes() const { return fileSizeBytes_; }
    std::string getLastModifiedIso() const { return lastModifiedIso_; }
    std::string getSha256Hash() const { return sha256Hash_; }
    std::string getGitCommitSha() const { return gitCommitSha_; }

    // Helper to preprocess raw interleaved BGR (HWC) to planar RGB (CHW) float 0..1
    static std::vector<float> preprocessBGRtoCHW(const unsigned char* bgrData, int width, int height);

    // Vectorized zero-copy OpenCV preprocess via cv::split pointer mapping
    static std::vector<float> preprocessOpenCVFrame(const cv::Mat& inputFrame, int inputWidth = 640, int inputHeight = 640);

    // Phase 3: Async CUDA stream execution
    void executeInference(const std::vector<float>& inputCPU, std::vector<float>& outputCPU);

    // Phase 4: Output tensor decoding and NMS post-processing
    static std::vector<Detection> postProcess(const std::vector<float>& rawOutput, int imgWidth = 640, int imgHeight = 640, float confThreshold = 0.25f, float nmsThreshold = 0.45f);

private:
    std::string enginePath_;
    int inputW_;
    int inputH_;
    bool isInitialized_;
    size_t fileSizeBytes_{0};
    std::string lastModifiedIso_{"N/A"};
    std::string sha256Hash_{"N/A"};
    std::string gitCommitSha_{"N/A"};

    void* context_;
    void* engine_;

    // Host & Device GPU buffer pointers
    void* hostInputBuffer_;
    void* hostOutputBuffer_;
    void* deviceInputBuffer_;
    void* deviceOutputBuffer_;
    
    size_t inputSize_;
    size_t outputSize_;
};
