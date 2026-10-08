#pragma once
#include <memory>
#include <string>
#include <vector>

// A separately exported research candidate. Scores are not calibrated
// probabilities; loading this class does not enable recovery in the live loop.
class TemporalRecoveryEngine {
public:
    static constexpr int inputElements=15*128*128,outputElements=32*32+1;
    struct Result {std::vector<float> logits;float gpuMs=0,wallMs=0;};
    explicit TemporalRecoveryEngine(const std::string& enginePath);
    ~TemporalRecoveryEngine();
    TemporalRecoveryEngine(const TemporalRecoveryEngine&)=delete;
    TemporalRecoveryEngine& operator=(const TemporalRecoveryEngine&)=delete;
    Result infer(const std::vector<float>& rgbTemporalCHW);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
