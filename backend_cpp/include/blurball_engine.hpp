#pragma once
#include "tensorrt_engine.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

// A thresholded heatmap component, not a calibrated probability or physical
// motion measurement. The undirected blur axis has a 180-degree ambiguity.
struct BlurBallObservation {
    Detection detection{}; // Normalized top-left box; confidence = heatmapPeak.
    cv::Point2f pixelCenter{};
    float componentMass = 0;
    float heatmapPeak = 0;
    float blurExtentSourcePx = 0;
    float blurAxisDeg = 0;
    std::array<cv::Point2f, 2> blurEndpoints{};
};

// Fixed-shape BlurBall export. The caller owns frame IDs and decides whether
// output channel 0/1/2 is eligible for recording or causal live presentation.
// There is no camera access, source substitution, or temporal padding here.
class BlurBallEngine {
public:
    static constexpr int inputWidth = 512, inputHeight = 288;
    static constexpr int inputElements = 9 * inputWidth * inputHeight;
    static constexpr int outputElements = 3 * inputWidth * inputHeight;
    static constexpr float heatmapThreshold = .7f;
    struct Result {
        std::array<std::vector<BlurBallObservation>, 3> observations;
        std::vector<float> logits; // Populated only for an explicit parity dump.
        std::vector<float> inputTensor; // Same opt-in diagnostic; RGB affine CHW.
        float preprocessMs = 0, h2dMs = 0, gpuMs = 0, d2hMs = 0;
        float decodeMs = 0, wallMs = 0;
    };
    explicit BlurBallEngine(const std::string& enginePath);
    ~BlurBallEngine();
    BlurBallEngine(const BlurBallEngine&) = delete;
    BlurBallEngine& operator=(const BlurBallEngine&) = delete;
    Result infer(const std::array<cv::Mat, 3>& bgrFrames, bool includeLogits = false);
    const std::string& getEnginePath() const;
    const std::string& getSha256Hash() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
