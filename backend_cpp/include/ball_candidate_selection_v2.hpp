#pragma once
#include "ball_candidate_selection.hpp"

// Offline experimental revision. Heatmap component mass ranks only BlurBall
// components, detector score ranks only YOLO boxes. This is a source selector,
// not a trained gate, probability calibration, temporal tracker or promotion.
namespace ball_candidates_v2 {
using ball_candidates::Candidate;
using ball_candidates::FrameIdentity;
using ball_candidates::Model;
using ball_candidates::Selection;
inline constexpr const char* selectorRevision = "blurball-primary-source-candidates-v2";
inline constexpr const char* selectionPolicy = "BLURBALL_PRIMARY_YOLO_NO_VETO";
struct Config {
    float yoloProposalScore = .1f, yoloEvidenceScore = .25f, yoloFallbackScore = .95f;
    float blurballMinimumPeak = BlurBallEngine::heatmapThreshold;
    double agreementRadiusPx = 12, ambiguityRatio = .8;
    bool allowYoloFallback = true;
};
inline void validate(const Config& config) {
    const auto score = [](float value) { return std::isfinite(value) && value > 0 && value <= 1; };
    if (!score(config.yoloProposalScore) || !score(config.yoloEvidenceScore) || !score(config.yoloFallbackScore) ||
        config.yoloProposalScore > config.yoloEvidenceScore || config.yoloEvidenceScore > config.yoloFallbackScore ||
        !score(config.blurballMinimumPeak) || config.blurballMinimumPeak < BlurBallEngine::heatmapThreshold ||
        !std::isfinite(config.agreementRadiusPx) || config.agreementRadiusPx <= 0 ||
        !std::isfinite(config.ambiguityRatio) || config.ambiguityRatio <= 0 || config.ambiguityRatio > 1)
        throw std::invalid_argument("Invalid BlurBall-primary experimental selector configuration");
}
inline Selection select(const FrameIdentity& frame, const std::string& yoloSha256, const std::string& blurballSha256,
                        const std::vector<Candidate>& yolo, const std::vector<Candidate>& blurball,
                        bool causalWindowReady, const Config& config = {}) {
    ball_candidates::validate(frame); validate(config);
    if (!ball_candidates::sha256(yoloSha256) || !ball_candidates::sha256(blurballSha256))
        throw std::invalid_argument("Invalid selection model identity");
    Selection result;
    std::vector<const Candidate*> eligibleYolo, eligibleBlurBall;
    const auto collect = [&](const std::vector<Candidate>& candidates, Model model, const std::string& modelHash,
                             std::vector<const Candidate*>& eligible, float minimum) {
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            if (!ball_candidates::sameFrame(frame, candidate.frame)) { result.reason = "FRAME_IDENTITY_MISMATCH"; return false; }
            if (candidate.model != model || candidate.modelSha256 != modelHash || candidate.index != index) {
                result.reason = "MODEL_OR_CANDIDATE_IDENTITY_MISMATCH"; return false;
            }
            if (!candidate.valid || !ball_candidates::invalidGeometry(candidate).empty()) { ++result.invalidCandidates; continue; }
            if (candidate.raw.confidence >= minimum) eligible.push_back(&candidate);
        }
        return true;
    };
    // Identity checks always precede warmup; a ready flag cannot authorize a
    // proposal from another source/frame/model. The caller supplies only the
    // newest channel of three actual consecutive frames, with no padding.
    if (!collect(yolo, Model::Yolo, yoloSha256, eligibleYolo, config.yoloEvidenceScore) ||
        !collect(blurball, Model::BlurBall, blurballSha256, eligibleBlurBall, config.blurballMinimumPeak)) return result;
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
    const auto* yoloBest = best(eligibleYolo, false);
    const auto* blurBest = best(eligibleBlurBall, true);
    // A second distant BlurBall component of comparable mass remains a reason
    // to abstain, even if YOLO agrees with one component.
    if (ambiguous(eligibleBlurBall, blurBest, true)) { result.reason = "AMBIGUOUS_BLURBALL_CANDIDATES"; return result; }
    const bool yoloAmbiguous = ambiguous(eligibleYolo, yoloBest, false);
    if (yoloBest) result.yoloSupportIndex = yoloBest->index;
    if (blurBest) result.blurballSupportIndex = blurBest->index;
    // Ambiguous YOLO has a best-score index for provenance, but cannot supply
    // one unambiguous spatial agreement measurement.
    if (blurBest && yoloBest && !yoloAmbiguous) result.agreementDistancePx = cv::norm(blurBest->centerPx - yoloBest->centerPx);
    if (blurBest) {
        result.selected = *blurBest;
        if (!yoloBest) result.reason = "BLURBALL_PRIMARY_NO_ELIGIBLE_YOLO";
        else if (yoloAmbiguous) result.reason = "BLURBALL_PRIMARY_AMBIGUOUS_YOLO";
        else if (*result.agreementDistancePx <= config.agreementRadiusPx) result.reason = "BLURBALL_PRIMARY_SPATIAL_AGREEMENT";
        else result.reason = "BLURBALL_PRIMARY_YOLO_DISAGREEMENT";
    } else if (!yoloBest) result.reason = "NO_ELIGIBLE_CANDIDATE";
    else if (!config.allowYoloFallback) result.reason = "YOLO_FALLBACK_DISABLED";
    else if (yoloAmbiguous) result.reason = "AMBIGUOUS_YOLO_CANDIDATES";
    else if (yoloBest->raw.confidence < config.yoloFallbackScore) result.reason = "YOLO_ONLY_INSUFFICIENT_EVIDENCE";
    else { result.selected = *yoloBest; result.reason = "CONSERVATIVE_YOLO_ONLY_FALLBACK"; }
    return result;
}
} // namespace ball_candidates_v2
