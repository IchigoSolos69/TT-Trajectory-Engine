#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Detection Box Structure for YOLO output
struct Detection {
    int class_id;        // Head-specific: ball head 0=ball; scene head 0=person, 1=table.
    float confidence;
    float x, y, w, h;    // Normalized coordinates (0..1)
};

struct InferenceTimingDetailed {
    float h2dMs{0.0f};
    float preprocessMs{0.0f};
    float gpuComputeMs{0.0f};
    float d2hMs{0.0f};
    float decodeMs{0.0f};
};

class TensorRTEngine {
public:
    TensorRTEngine(const std::string& enginePath = "", int inputWidth = 640, int inputHeight = 640, int classes = 1);
    ~TensorRTEngine();

    bool init();
    std::vector<Detection> inferFrame(const std::vector<float>& rgbNormalizedBuffer, int origImgWidth = 1920, int origImgHeight = 1080, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    std::vector<Detection> inferFrameDetailed(const std::vector<float>& rgbNormalizedBuffer, InferenceTimingDetailed& timing, int origImgWidth = 1920, int origImgHeight = 1080, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    
    // High-performance CUDA Preprocessed GPU Pipeline (bypasses CPU resize/copy)
    std::vector<Detection> inferFrameGPU(const cv::Mat& frame, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    std::vector<Detection> inferFrameGPUDetailed(const cv::Mat& frame, InferenceTimingDetailed& timing, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    
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
    void executeInferenceDetailed(const std::vector<float>& inputCPU, std::vector<float>& outputCPU, float& h2dMs, float& gpuComputeMs, float& d2hMs);

    // Phase 4: Output tensor decoding and NMS post-processing
    static std::vector<Detection> postProcess(const std::vector<float>& rawOutput, int imgWidth = 640, int imgHeight = 640, float confThreshold = 0.15f, float nmsThreshold = 0.45f);

private:
    std::string enginePath_;
    int classes_;
    int inputW_;
    int inputH_;
    bool isInitialized_;
    size_t fileSizeBytes_{0};
    std::string lastModifiedIso_{"N/A"};
    std::string sha256Hash_{"N/A"};
    std::string gitCommitSha_{"N/A"};

    void* context_{nullptr};
    void* engine_{nullptr};
    void* stream_{nullptr};

    // Host & Device GPU buffer pointers (Pinned Host Memory via cudaMallocHost)
    void* hostInputBuffer_{nullptr};
    void* hostOutputBuffer_{nullptr};
    void* deviceInputBuffer_{nullptr};
    void* deviceOutputBuffer_{nullptr};
    void* deviceBGRBuffer_{nullptr};
    size_t deviceBGRCapacity_{0};

    // Persistent CUDA Timing Events (allocated once, reused per frame)
    void* evStartH2D_{nullptr};
    void* evStopH2D_{nullptr};
    void* evStartPrep_{nullptr};
    void* evStopPrep_{nullptr};
    void* evStartExec_{nullptr};
    void* evStopExec_{nullptr};
    void* evStartD2H_{nullptr};
    void* evStopD2H_{nullptr};
    
    size_t inputSize_{0};
    size_t outputSize_{0};
};
