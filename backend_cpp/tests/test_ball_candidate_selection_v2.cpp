#include "ball_candidate_selection_v2.hpp"
#include "analysis_journal.hpp"
#include <functional>
#include <iostream>
#include <limits>

using namespace ball_candidates;
namespace primary = ball_candidates_v2;
namespace {
const std::string sourceHash(64, 'a'), imageHash(64, 'b'), yoloHash(64, 'c'), blurHash(64, 'd');
FrameIdentity frame() { return {sourceHash, imageHash, 3, .033333, {1920, 1080}}; }
Detection yoloAt(float x, float y, float score = .95f) {
    return {0, score, (x - 2) / 1920, (y - 2) / 1080, 4.f / 1920, 4.f / 1080};
}
BlurBallObservation blurAt(float x, float y, float peak = .7f, float mass = 10) {
    BlurBallObservation blur;
    blur.detection = yoloAt(x + 9, y, peak); // Distinguish box center from heatmap center.
    blur.pixelCenter = {x, y}; blur.heatmapPeak = peak; blur.componentMass = mass;
    blur.blurExtentSourcePx = 20; blur.blurAxisDeg = 0; blur.blurEndpoints = {{{x - 10, y}, {x + 10, y}}};
    return blur;
}
void require(bool value, const std::string& text) { if (!value) throw std::runtime_error(text); }
void invalidArgument(const std::function<void()>& action) {
    bool failed = false; try { action(); } catch (const std::invalid_argument&) { failed = true; }
    require(failed, "Malformed contract must throw invalid_argument");
}
Selection choose(const std::vector<Detection>& yolo, const std::vector<BlurBallObservation>& blur,
                 bool ready = true, primary::Config config = {}) {
    return primary::select(frame(), yoloHash, blurHash, normalizeYolo(frame(), yoloHash, yolo),
        normalizeBlurBall(frame(), blurHash, blur), ready, config);
}
}
int main(int argc, char** argv) try {
    if (argc > 2) throw std::invalid_argument("Usage: test_ball_candidate_selection_v2 [report.json]");
    std::vector<std::string> passed;
    const auto check = [&](const std::string& name, const std::function<void()>& test) { test(); passed.push_back(name); };
    check("v1_remains_strict_while_v2_preserves_disagreeing_blurball", [] {
        const auto yolo = normalizeYolo(frame(), yoloHash, {yoloAt(700, 200, .99f)});
        const auto blur = normalizeBlurBall(frame(), blurHash, {blurAt(200, 200, .9f)});
        const auto old = ball_candidates::select(frame(), yoloHash, blurHash, yolo, blur, true);
        const auto next = primary::select(frame(), yoloHash, blurHash, yolo, blur, true);
        require(!old.selected && old.reason == "DETECTOR_DISAGREEMENT", "v1 behavior changed");
        require(next.selected && next.selected->centerPx == cv::Point2f(200, 200) &&
            next.reason == "BLURBALL_PRIMARY_YOLO_DISAGREEMENT", "Disagreeing YOLO vetoed BlurBall");
    });
    check("eligibility_boundary_preserves_exact_heatmap_evidence_without_score_blend", [] {
        const auto raw = blurAt(336, 225, .7f, 12);
        const auto result = choose({yoloAt(333, 222, .25f)}, {raw});
        require(result.selected && result.reason == "BLURBALL_PRIMARY_SPATIAL_AGREEMENT", "Boundary candidate rejected");
        const auto& picked = *result.selected;
        require(picked.centerPx == raw.pixelCenter && picked.raw.confidence == raw.heatmapPeak &&
            picked.raw.x == raw.detection.x && picked.blur->componentMass == raw.componentMass &&
            picked.blur->blurEndpoints == raw.blurEndpoints, "Selected original evidence changed");
        require(result.yoloSupportIndex == 0 && result.blurballSupportIndex == 0 && result.agreementDistancePx.has_value(),
            "Support indices or agreement distance missing");
    });
    check("weak_absent_agreeing_disagreeing_and_ambiguous_yolo_never_veto_blurball", [] {
        for (float peak : {.7f, .8f, .95f, 1.f}) for (float score : {0.f, .1f, .24f, .25f, .5f, .94f, .95f, 1.f}) {
            for (float x : {200.f, 700.f}) {
                const auto result = choose({yoloAt(x, 200, score), yoloAt(900, 200, score)}, {blurAt(200, 200, peak)});
                require(result.selected && result.selected->model == Model::BlurBall &&
                    result.selected->centerPx == cv::Point2f(200, 200), "YOLO score/ambiguity vetoed primary (64 cases)");
            }
        }
        const auto absent = choose({}, {blurAt(200, 200)});
        require(absent.selected && absent.reason == "BLURBALL_PRIMARY_NO_ELIGIBLE_YOLO", "Absent YOLO vetoed BlurBall");
    });
    check("ambiguous_yolo_is_reported_with_primary_coordinate", [] {
        const auto result = choose({yoloAt(200, 200, 1), yoloAt(700, 200, .8f)}, {blurAt(200, 200)});
        require(result.selected && result.reason == "BLURBALL_PRIMARY_AMBIGUOUS_YOLO" && result.eligibleYolo == 2 &&
            result.yoloSupportIndex == 0 && !result.agreementDistancePx, "YOLO ambiguity provenance lost");
    });
    check("ambiguous_blurball_abstains_even_with_yolo_agreement", [] {
        const auto result = choose({yoloAt(200, 200, 1)}, {blurAt(200, 200, .99f, 10), blurAt(700, 200, .99f, 8)});
        require(!result.selected && result.reason == "AMBIGUOUS_BLURBALL_CANDIDATES" &&
            !result.yoloSupportIndex && !result.blurballSupportIndex, "YOLO resolved ambiguous BlurBall without validation");
    });
    check("blurball_ambiguity_ratio_just_below_boundary_is_allowed", [] {
        const auto result = choose({}, {blurAt(200, 200, .99f, 10), blurAt(700, 200, .99f, 7.999f)});
        require(result.selected && result.selected->index == 0, "Below ambiguity boundary rejected");
    });
    check("component_mass_not_cross_model_score_or_heatmap_peak_ranks_blurball", [] {
        const auto result = choose({yoloAt(700, 200, 1)}, {blurAt(200, 200, .99f, 5), blurAt(500, 200, .7f, 10)});
        require(result.selected && result.selected->index == 1 && result.selected->centerPx == cv::Point2f(500, 200) &&
            result.selected->raw.confidence == .7f, "Higher peak/YOLO substituted for within-model mass ranking");
    });
    check("near_components_do_not_blend_or_trigger_ambiguity", [] {
        const auto result = choose({}, {blurAt(200, 200, .9f, 10), blurAt(204, 201, .99f, 9)});
        require(result.selected && result.selected->index == 0 && result.selected->centerPx == cv::Point2f(200, 200),
            "Near components blended or rejected");
    });
    check("agreement_radius_boundary_is_inclusive", [] {
        const auto yolo = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200, .5f)});
        const auto blur = normalizeBlurBall(frame(), blurHash, {blurAt(yolo[0].centerPx.x + 12, yolo[0].centerPx.y)});
        const auto result = primary::select(frame(), yoloHash, blurHash, yolo, blur, true);
        require(result.selected && result.agreementDistancePx == 12 && result.reason == "BLURBALL_PRIMARY_SPATIAL_AGREEMENT",
            "Exact agreement boundary rejected");
    });
    check("ambiguity_radius_boundary_is_inclusive", [] {
        const auto result = choose({}, {blurAt(200, 200, .8f, 10), blurAt(212, 200, .8f, 8)});
        require(result.selected && result.selected->index == 0, "Exact spatial ambiguity boundary rejected");
    });
    check("yolo_fallback_requires_inclusive_high_score_without_blurball", [] {
        require(!choose({yoloAt(200, 200, .94f)}, {}).selected, "Weak YOLO fallback accepted");
        const auto result = choose({yoloAt(200, 200)}, {});
        require(result.selected && result.selected->model == Model::Yolo && result.reason == "CONSERVATIVE_YOLO_ONLY_FALLBACK",
            "High-score fallback threshold not inclusive");
    });
    check("ambiguous_yolo_cannot_fallback", [] {
        const auto result = choose({yoloAt(200, 200, 1), yoloAt(700, 200, .8f)}, {});
        require(!result.selected && result.reason == "AMBIGUOUS_YOLO_CANDIDATES", "Ambiguous fallback accepted");
    });
    check("explicitly_disabled_fallback_abstains_but_keeps_blurball", [] {
        primary::Config config; config.allowYoloFallback = false;
        const auto absent = choose({yoloAt(200, 200, 1)}, {}, true, config);
        require(!absent.selected && absent.reason == "YOLO_FALLBACK_DISABLED", "Disabled fallback accepted");
        require(choose({yoloAt(700, 200, 1)}, {blurAt(200, 200)}, true, config).selected.has_value(),
            "Fallback setting disabled primary BlurBall");
    });
    check("no_eligible_candidates_never_fabricate_coordinate", [] {
        for (const auto& result : {choose({}, {}), choose({yoloAt(200, 200, .24f)}, {blurAt(200, 200, .69f)})})
            require(!result.selected && result.reason == "NO_ELIGIBLE_CANDIDATE", "Ineligible outputs fabricated ball");
    });
    check("custom_thresholds_change_eligibility_without_model_score_comparison", [] {
        primary::Config config; config.blurballMinimumPeak = .8f; config.yoloFallbackScore = 1;
        const auto result = choose({yoloAt(200, 200, .95f)}, {blurAt(700, 200, .79f)}, true, config);
        require(!result.selected && result.reason == "YOLO_ONLY_INSUFFICIENT_EVIDENCE" && result.eligibleBlurBall == 0,
            "Custom raw thresholds not applied");
    });
    check("causal_warmup_never_uses_either_model_to_pad_history", [] {
        const auto result = choose({yoloAt(200, 200, 1)}, {blurAt(200, 200, 1)}, false);
        require(!result.selected && result.reason == "BLURBALL_CAUSAL_WARMUP" && result.eligibleYolo == 1 &&
            result.eligibleBlurBall == 1, "Warmup created an observed fused ball");
    });
    check("all_source_image_frame_time_size_identity_changes_fail_closed", [] {
        for (int change = 0; change < 5; ++change) for (bool ready : {false, true}) {
            auto candidates = normalizeBlurBall(frame(), blurHash, {blurAt(200, 200)});
            if (change == 0) candidates[0].frame.sourceSha256 = std::string(64, 'e');
            if (change == 1) candidates[0].frame.decodedImageSha256 = std::string(64, 'e');
            if (change == 2) candidates[0].frame.frameId++;
            if (change == 3) candidates[0].frame.timestamp += .001;
            if (change == 4) candidates[0].frame.imageSize.width++;
            const auto result = primary::select(frame(), yoloHash, blurHash, {}, candidates, ready);
            require(!result.selected && result.reason == "FRAME_IDENTITY_MISMATCH", "Cross-identity candidate accepted");
        }
    });
    check("malformed_yolo_identity_is_not_ignored_when_blurball_is_valid", [] {
        auto yolo = normalizeYolo(frame(), yoloHash, {yoloAt(700, 200, .1f)}); yolo[0].frame.frameId++;
        const auto result = primary::select(frame(), yoloHash, blurHash, yolo,
            normalizeBlurBall(frame(), blurHash, {blurAt(200, 200)}), true);
        require(!result.selected && result.reason == "FRAME_IDENTITY_MISMATCH", "Primary policy bypassed identity validation");
    });
    check("model_kind_hash_and_index_identity_fail_before_warmup", [] {
        for (int change = 0; change < 3; ++change) for (bool ready : {false, true}) {
            auto candidates = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200)});
            if (change == 0) candidates[0].model = Model::BlurBall;
            if (change == 1) candidates[0].modelSha256 = blurHash;
            if (change == 2) candidates[0].index++;
            const auto result = primary::select(frame(), yoloHash, blurHash, candidates, {}, ready);
            require(!result.selected && result.reason == "MODEL_OR_CANDIDATE_IDENTITY_MISMATCH", "Identity check bypassed");
        }
    });
    check("invalid_geometry_is_counted_and_never_selected", [] {
        auto yolo = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200)}); yolo[0].centerPx.x += 1;
        auto blur = normalizeBlurBall(frame(), blurHash, {blurAt(200, 200)}); blur[0].blur->componentMass = 0;
        const auto result = primary::select(frame(), yoloHash, blurHash, yolo, blur, true);
        require(!result.selected && result.invalidCandidates == 2 && result.reason == "NO_ELIGIBLE_CANDIDATE",
            "Tampered valid flag/geometry bypassed recheck");
    });
    check("invalid_blurball_can_fallback_only_to_valid_strong_yolo", [] {
        auto raw = blurAt(200, 200); raw.blurEndpoints[0].x = std::numeric_limits<float>::infinity();
        const auto result = choose({yoloAt(700, 200)}, {raw});
        require(result.selected && result.selected->model == Model::Yolo && result.invalidCandidates == 1 &&
            result.eligibleBlurBall == 0, "Malformed BlurBall chosen or invalid output hid valid fallback");
    });
    check("border_center_preserves_original_unclipped_box", [] {
        const auto result = choose({}, {blurAt(0, 0)});
        require(result.selected && result.selected->centerPx == cv::Point2f(0, 0) &&
            result.selected->blur->blurEndpoints[0].x < 0, "Border evidence clipped");
    });
    check("invalid_frame_and_model_contracts_throw", [] {
        auto f = frame(); f.timestamp = std::numeric_limits<double>::quiet_NaN();
        invalidArgument([&] { primary::select(f, yoloHash, blurHash, {}, {}, true); });
        invalidArgument([] { primary::select(frame(), "unknown", blurHash, {}, {}, true); });
        invalidArgument([] { primary::select(frame(), yoloHash, std::string(64, 'g'), {}, {}, true); });
    });
    check("invalid_threshold_order_nan_radius_and_ambiguity_throw", [] {
        for (int change = 0; change < 9; ++change) {
            primary::Config config;
            if (change == 0) config.yoloProposalScore = .3f;
            if (change == 1) config.yoloEvidenceScore = 1;
            if (change == 2) config.yoloFallbackScore = std::numeric_limits<float>::quiet_NaN();
            if (change == 3) config.blurballMinimumPeak = .69f;
            if (change == 4) config.blurballMinimumPeak = 1.1f;
            if (change == 5) config.agreementRadiusPx = 0;
            if (change == 6) config.agreementRadiusPx = std::numeric_limits<double>::infinity();
            if (change == 7) config.ambiguityRatio = 0;
            if (change == 8) config.ambiguityRatio = 1.01;
            invalidArgument([&] { choose({}, {}, true, config); });
        }
    });
    std::string names = "[";
    for (const auto& item : passed) { if (names.size() > 1) names += ','; names += recording::quote(item); }
    names += ']';
    const auto report = recording::json({{"passed", "true"}, {"check_count", std::to_string(passed.size())},
        {"selector_revision", recording::quote(primary::selectorRevision)}, {"selection_policy", recording::quote(primary::selectionPolicy)},
        {"checks", names}, {"gpu_inference_run", "false"}, {"accuracy_measured", "false"},
        {"thresholds_fitted", "false"}, {"promotion_approved", "false"}, {"production_backend_modified", "false"}});
    if (argc == 2) recording::writeText(std::filesystem::u8path(argv[1]), report);
    std::cout << report << '\n'; return 0;
} catch (const std::exception& error) { std::cerr << "[test_ball_candidate_selection_v2] " << error.what() << '\n'; return 1; }
