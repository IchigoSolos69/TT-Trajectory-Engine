#pragma once
#include "blurball_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Offline research interface. Raw YOLO scores, heatmap peaks and component mass
// remain separate evidence; this selector does not calibrate confidence or
// manufacture a coordinate between two detector proposals.
namespace ball_candidates {
inline constexpr const char* selectorRevision = "conservative-source-candidates-v1";
enum class Model { Yolo, BlurBall };
inline const char* name(Model model) { return model == Model::Yolo ? "yolo" : "blurball"; }
inline bool sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
struct FrameIdentity {
    std::string sourceSha256, decodedImageSha256;
    uint64_t frameId = 0;
    double timestamp = 0;
    cv::Size imageSize;
};
inline void validate(const FrameIdentity& frame) {
    if (!sha256(frame.sourceSha256) || !sha256(frame.decodedImageSha256) || frame.frameId == 0 ||
        !std::isfinite(frame.timestamp) || frame.timestamp < 0 || frame.imageSize.width < 1 || frame.imageSize.height < 1)
        throw std::invalid_argument("Invalid source-frame candidate identity");
}
inline bool sameFrame(const FrameIdentity& first, const FrameIdentity& second) {
    return first.sourceSha256 == second.sourceSha256 && first.decodedImageSha256 == second.decodedImageSha256 &&
        first.frameId == second.frameId && first.timestamp == second.timestamp && first.imageSize == second.imageSize;
}
struct Candidate {
    FrameIdentity frame;
    Model model = Model::Yolo;
    std::string modelSha256;
    size_t index = 0; // Index within this model's original postprocessed output.
    Detection raw{}; // Unmodified source-normalized top-left box and raw score.
    cv::Point2f centerPx{};
    std::optional<BlurBallObservation> blur;
    bool valid = false;
    std::string invalidReason;
};
inline std::string invalidGeometry(const Candidate& candidate) {
    const auto& box = candidate.raw;
    if (box.class_id != 0) return "NON_BALL_CLASS";
    if (!std::isfinite(box.confidence) || box.confidence < 0 || box.confidence > 1) return "INVALID_RAW_SCORE";
    if (!std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.w) || !std::isfinite(box.h) || box.w <= 0 || box.h <= 0)
        return "INVALID_BOX";
    if (!std::isfinite(candidate.centerPx.x) || !std::isfinite(candidate.centerPx.y)) return "NONFINITE_CENTER";
    if (candidate.centerPx.x < 0 || candidate.centerPx.y < 0 || candidate.centerPx.x >= candidate.frame.imageSize.width ||
        candidate.centerPx.y >= candidate.frame.imageSize.height) return "CENTER_OUTSIDE_SOURCE";
    if (candidate.model == Model::Yolo && candidate.centerPx != cv::Point2f(
        (box.x + box.w * .5f) * candidate.frame.imageSize.width,
        (box.y + box.h * .5f) * candidate.frame.imageSize.height)) return "YOLO_CENTER_MISMATCH";
    if (candidate.model == Model::BlurBall) {
        if (!candidate.blur) return "MISSING_BLURBALL_EVIDENCE";
        const auto& blur = *candidate.blur;
        if (!std::isfinite(blur.componentMass) || blur.componentMass <= 0 || !std::isfinite(blur.heatmapPeak) ||
            blur.heatmapPeak < 0 || blur.heatmapPeak > 1 || blur.heatmapPeak != box.confidence)
            return "INVALID_BLURBALL_SCORE_OR_MASS";
        if (candidate.centerPx != blur.pixelCenter) return "BLURBALL_CENTER_MISMATCH";
        if (!std::isfinite(blur.blurExtentSourcePx) || blur.blurExtentSourcePx < 0 || !std::isfinite(blur.blurAxisDeg) ||
            blur.blurAxisDeg < 0 || blur.blurAxisDeg >= 180) return "INVALID_BLUR_EVIDENCE";
        for (const auto& endpoint : blur.blurEndpoints)
            if (!std::isfinite(endpoint.x) || !std::isfinite(endpoint.y)) return "NONFINITE_BLUR_ENDPOINT";
    } else if (candidate.blur) return "UNEXPECTED_BLUR_EVIDENCE";
    return "";
}
inline std::vector<Candidate> normalizeYolo(const FrameIdentity& frame, const std::string& modelSha256,
                                           const std::vector<Detection>& observations) {
    validate(frame);
    if (!sha256(modelSha256)) throw std::invalid_argument("Invalid YOLO model SHA-256");
    std::vector<Candidate> result;
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto& box = observations[index];
        Candidate candidate;
        candidate.frame = frame; candidate.model = Model::Yolo; candidate.modelSha256 = modelSha256;
        candidate.index = index; candidate.raw = box;
        candidate.centerPx = {(box.x + box.w * .5f) * frame.imageSize.width, (box.y + box.h * .5f) * frame.imageSize.height};
        candidate.invalidReason = invalidGeometry(candidate); candidate.valid = candidate.invalidReason.empty();
        result.push_back(std::move(candidate));
    }
    return result;
}
inline std::vector<Candidate> normalizeBlurBall(const FrameIdentity& frame, const std::string& modelSha256,
                                               const std::vector<BlurBallObservation>& observations) {
    validate(frame);
    if (!sha256(modelSha256)) throw std::invalid_argument("Invalid BlurBall model SHA-256");
    std::vector<Candidate> result;
    for (size_t index = 0; index < observations.size(); ++index) {
        Candidate candidate;
        candidate.frame = frame; candidate.model = Model::BlurBall; candidate.modelSha256 = modelSha256;
        candidate.index = index; candidate.raw = observations[index].detection;
        candidate.centerPx = observations[index].pixelCenter; candidate.blur = observations[index];
        candidate.invalidReason = invalidGeometry(candidate); candidate.valid = candidate.invalidReason.empty();
        result.push_back(std::move(candidate));
    }
    return result;
}
struct Config {
    // Research defaults, not fitted validation thresholds. Every evaluation must
    // record these values and keep the production models unchanged.
    float yoloProposalScore = .1f, yoloAgreementScore = .25f, yoloSoloScore = .95f;
    float blurballAgreementPeak = BlurBallEngine::heatmapThreshold, blurballSoloPeak = .95f;
    double agreementRadiusPx = 12;
    double ambiguityRatio = .8; // Within-model score/mass ratio only.
};
inline void validate(const Config& config) {
    const auto score = [](float value) { return std::isfinite(value) && value > 0 && value <= 1; };
    if (!score(config.yoloProposalScore) || !score(config.yoloAgreementScore) || !score(config.yoloSoloScore) ||
        config.yoloProposalScore > config.yoloAgreementScore || config.yoloAgreementScore > config.yoloSoloScore ||
        !score(config.blurballAgreementPeak) || !score(config.blurballSoloPeak) ||
        config.blurballAgreementPeak < BlurBallEngine::heatmapThreshold || config.blurballAgreementPeak > config.blurballSoloPeak ||
        !std::isfinite(config.agreementRadiusPx) || config.agreementRadiusPx <= 0 ||
        !std::isfinite(config.ambiguityRatio) || config.ambiguityRatio <= 0 || config.ambiguityRatio > 1)
        throw std::invalid_argument("Invalid conservative candidate selector configuration");
}
struct Selection {
    std::optional<Candidate> selected;
    std::optional<size_t> yoloSupportIndex, blurballSupportIndex;
    std::optional<double> agreementDistancePx;
    std::string reason;
    size_t invalidCandidates = 0, eligibleYolo = 0, eligibleBlurBall = 0;
};
inline Selection select(const FrameIdentity& frame, const std::string& yoloSha256, const std::string& blurballSha256,
                        const std::vector<Candidate>& yolo, const std::vector<Candidate>& blurball,
                        bool causalWindowReady, const Config& config = {}) {
    validate(frame); validate(config);
    if (!sha256(yoloSha256) || !sha256(blurballSha256)) throw std::invalid_argument("Invalid selection model identity");
    Selection result;
    std::vector<const Candidate*> eligibleYolo, eligibleBlurBall;
    const auto collect = [&](const std::vector<Candidate>& candidates, Model model, const std::string& modelHash,
                             std::vector<const Candidate*>& eligible, float minimum) {
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            if (!sameFrame(frame, candidate.frame)) { result.reason = "FRAME_IDENTITY_MISMATCH"; return false; }
            if (candidate.model != model || candidate.modelSha256 != modelHash || candidate.index != index) {
                result.reason = "MODEL_OR_CANDIDATE_IDENTITY_MISMATCH"; return false;
            }
            // Recheck geometry instead of trusting an externally edited valid flag.
            if (!candidate.valid || !invalidGeometry(candidate).empty()) { ++result.invalidCandidates; continue; }
            if (candidate.raw.confidence >= minimum) eligible.push_back(&candidate);
        }
        return true;
    };
    if (!collect(yolo, Model::Yolo, yoloSha256, eligibleYolo, config.yoloAgreementScore) ||
        !collect(blurball, Model::BlurBall, blurballSha256, eligibleBlurBall, config.blurballAgreementPeak)) return result;
    result.eligibleYolo = eligibleYolo.size(); result.eligibleBlurBall = eligibleBlurBall.size();
    if (!causalWindowReady) { result.reason = "BLURBALL_CAUSAL_WARMUP"; return result; }
    const auto best = [](const std::vector<const Candidate*>& candidates, bool mass) -> const Candidate* {
        const Candidate* winner = nullptr;
        for (const auto* candidate : candidates) {
            const double value = mass ? candidate->blur->componentMass : candidate->raw.confidence;
            const double previous = !winner ? -1 : mass ? winner->blur->componentMass : winner->raw.confidence;
            if (value > previous) winner = candidate;
        }
        return winner;
    };
    const auto* yoloBest = best(eligibleYolo, false);
    const auto* blurBest = best(eligibleBlurBall, true);
    const auto ambiguous = [&](const std::vector<const Candidate*>& candidates, const Candidate* winner, bool mass) {
        if (!winner) return false;
        const double rank = mass ? winner->blur->componentMass : winner->raw.confidence;
        for (const auto* candidate : candidates) {
            const double other = mass ? candidate->blur->componentMass : candidate->raw.confidence;
            if (candidate != winner && other >= rank * config.ambiguityRatio &&
                cv::norm(candidate->centerPx - winner->centerPx) > config.agreementRadiusPx) return true;
        }
        return false;
    };
    if (ambiguous(eligibleBlurBall, blurBest, true)) { result.reason = "AMBIGUOUS_BLURBALL_CANDIDATES"; return result; }
    if (ambiguous(eligibleYolo, yoloBest, false)) { result.reason = "AMBIGUOUS_YOLO_CANDIDATES"; return result; }
    if (yoloBest) result.yoloSupportIndex = yoloBest->index;
    if (blurBest) result.blurballSupportIndex = blurBest->index;
    if (blurBest && yoloBest) {
        result.agreementDistancePx = cv::norm(blurBest->centerPx - yoloBest->centerPx);
        if (*result.agreementDistancePx > config.agreementRadiusPx) { result.reason = "DETECTOR_DISAGREEMENT"; return result; }
        result.selected = *blurBest; result.reason = "SPATIAL_AGREEMENT_BLURBALL_COORDINATE";
    } else if (blurBest) {
        if (blurBest->raw.confidence < config.blurballSoloPeak) result.reason = "BLURBALL_ONLY_INSUFFICIENT_EVIDENCE";
        else { result.selected = *blurBest; result.reason = "STRONG_BLURBALL_ONLY"; }
    } else if (yoloBest) {
        if (yoloBest->raw.confidence < config.yoloSoloScore) result.reason = "YOLO_ONLY_INSUFFICIENT_EVIDENCE";
        else { result.selected = *yoloBest; result.reason = "CONSERVATIVE_YOLO_ONLY_FALLBACK"; }
    } else result.reason = "NO_ELIGIBLE_CANDIDATE";
    return result;
}
} // namespace ball_candidates
