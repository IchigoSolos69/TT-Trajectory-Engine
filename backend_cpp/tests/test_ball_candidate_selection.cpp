#include "ball_candidate_selection.hpp"
#include "analysis_journal.hpp"
#include <iostream>
#include <limits>
#include <functional>

using namespace ball_candidates;
namespace {
const std::string sourceHash(64, 'a'), imageHash(64, 'b'), yoloHash(64, 'c'), blurHash(64, 'd');
FrameIdentity frame() { return {sourceHash, imageHash, 3, .033333, {1920, 1080}}; }
Detection yoloAt(float x, float y, float score = .95f) {
    return {0, score, (x - 2) / 1920, (y - 2) / 1080, 4.f / 1920, 4.f / 1080};
}
BlurBallObservation blurAt(float x, float y, float peak = .95f, float mass = 10) {
    BlurBallObservation blur;
    blur.detection = yoloAt(x + 9, y, peak); // Box midpoint deliberately differs from weighted centroid.
    blur.pixelCenter = {x, y}; blur.heatmapPeak = peak; blur.componentMass = mass;
    blur.blurExtentSourcePx = 20; blur.blurAxisDeg = 0; blur.blurEndpoints = {{{x - 10, y}, {x + 10, y}}};
    return blur;
}
void require(bool value, const std::string& text) { if (!value) throw std::runtime_error(text); }
void invalidArgument(const std::function<void()>& action) {
    bool failed = false; try { action(); } catch (const std::invalid_argument&) { failed = true; }
    require(failed, "Expected invalid_argument, no valid selection from malformed contract");
}
Selection choose(const std::vector<Detection>& yolo, const std::vector<BlurBallObservation>& blur, bool ready = true, Config config = {}) {
    return select(frame(), yoloHash, blurHash, normalizeYolo(frame(), yoloHash, yolo), normalizeBlurBall(frame(), blurHash, blur), ready, config);
}
}
int main(int argc, char** argv) try {
    if (argc > 2) throw std::invalid_argument("Usage: test_ball_candidate_selection [report.json]");
    std::vector<std::string> passed;
    const auto check = [&](const std::string& name, const std::function<void()>& test) { test(); passed.push_back(name); };
    check("source_pixel_yolo_center_and_original_box", [] {
        const auto raw = yoloAt(333, 222, .93f); const auto normalized = normalizeYolo(frame(), yoloHash, {raw});
        require(normalized.size() == 1 && normalized[0].valid, "Valid YOLO observation rejected");
        require(cv::norm(normalized[0].centerPx - cv::Point2f(333, 222)) < .0001, "YOLO top-left box not converted to source center");
        require(normalized[0].raw.x == raw.x && normalized[0].raw.w == raw.w && normalized[0].raw.confidence == raw.confidence, "Original YOLO box/score changed");
    });
    check("blurball_uses_weighted_center_not_box_midpoint", [] {
        const auto raw = blurAt(333, 222, .97f, 12); const auto normalized = normalizeBlurBall(frame(), blurHash, {raw});
        require(normalized[0].valid && normalized[0].centerPx == raw.pixelCenter, "Weighted BlurBall center lost");
        require(cv::norm(normalized[0].centerPx - cv::Point2f((raw.detection.x + raw.detection.w / 2) * 1920, (raw.detection.y + raw.detection.h / 2) * 1080)) > 8, "Test did not distinguish box midpoint");
        require(normalized[0].blur->componentMass == 12 && normalized[0].blur->blurEndpoints == raw.blurEndpoints, "Blur provenance altered");
    });
    check("agreement_selects_exact_existing_blurball_coordinate", [] {
        const auto raw = blurAt(336, 225, .8f, 10); const auto result = choose({yoloAt(333, 222, .6f)}, {raw});
        require(result.selected && result.reason == "SPATIAL_AGREEMENT_BLURBALL_COORDINATE", "Agreement did not select");
        require(result.selected->centerPx == raw.pixelCenter && result.selected->raw.confidence == .8f, "Selector blended position/score");
        require(result.yoloSupportIndex == 0 && result.blurballSupportIndex == 0 && result.agreementDistancePx.has_value(), "Agreement support missing");
    });
    check("agreement_inclusive_spatial_boundary", [] {
        auto yolo = normalizeYolo(frame(), yoloHash, {yoloAt(320, 220, .5f)});
        auto blur = normalizeBlurBall(frame(), blurHash, {blurAt(yolo[0].centerPx.x + 12, yolo[0].centerPx.y, .8f)});
        const auto result = select(frame(), yoloHash, blurHash, yolo, blur, true);
        require(result.selected && result.agreementDistancePx == 12., "Exact spatial agreement boundary incorrectly rejected");
    });
    check("outside_agreement_abstains_despite_strong_scores", [] {
        const auto result = choose({yoloAt(200, 200, .99f)}, {blurAt(220, 200, .99f)});
        require(!result.selected && result.reason == "DETECTOR_DISAGREEMENT", "Distant positions blended or stronger raw score won");
    });
    check("strong_blurball_only_is_not_forced_to_require_yolo", [] {
        const auto result = choose({}, {blurAt(200, 200)});
        require(result.selected && result.selected->model == Model::BlurBall && result.reason == "STRONG_BLURBALL_ONLY", "Strong BlurBall-only proposal rejected");
    });
    check("weak_blurball_only_abstains", [] {
        require(choose({}, {blurAt(200, 200, .94f)}).reason == "BLURBALL_ONLY_INSUFFICIENT_EVIDENCE", "Weak solo heatmap accepted");
    });
    check("conservative_yolo_only_threshold_is_inclusive", [] {
        const auto result = choose({yoloAt(200, 200)}, {});
        require(result.selected && result.selected->model == Model::Yolo && result.reason == "CONSERVATIVE_YOLO_ONLY_FALLBACK", "Conservative YOLO fallback rejected");
    });
    check("weak_yolo_only_abstains", [] {
        const auto result = choose({yoloAt(200, 200, .94f)}, {});
        require(!result.selected && result.reason == "YOLO_ONLY_INSUFFICIENT_EVIDENCE", "Weak YOLO-only proposal accepted");
    });
    check("proposal_below_agreement_is_retained_but_does_not_veto", [] {
        const auto result = choose({yoloAt(700, 200, .24f)}, {blurAt(200, 200)});
        require(result.selected && result.reason == "STRONG_BLURBALL_ONLY" && result.eligibleYolo == 0, "Weak unrelated YOLO evidence vetoed strong BlurBall");
    });
    check("equal_boundary_agreement_score_is_eligible", [] {
        const auto result = choose({yoloAt(200, 200, .25f)}, {blurAt(200, 200, .71f)});
        require(result.selected && result.eligibleYolo == 1 && result.eligibleBlurBall == 1, "Threshold boundary changed");
    });
    check("competing_blurball_mass_abstains", [] {
        const auto result = choose({yoloAt(200, 200)}, {blurAt(200, 200, .99f, 10), blurAt(500, 200, .99f, 8)});
        require(!result.selected && result.reason == "AMBIGUOUS_BLURBALL_CANDIDATES", "Competing distant mass hypotheses accepted");
    });
    check("competing_yolo_score_abstains", [] {
        const auto result = choose({yoloAt(200, 200, 1), yoloAt(500, 200, .8f)}, {blurAt(200, 200)});
        require(!result.selected && result.reason == "AMBIGUOUS_YOLO_CANDIDATES", "Competing distant YOLO hypotheses accepted");
    });
    check("near_duplicate_components_use_original_top_mass", [] {
        const auto result = choose({yoloAt(200, 200)}, {blurAt(200, 200, .8f, 9), blurAt(204, 201, .75f, 10)});
        require(result.selected && result.selected->index == 1 && result.selected->centerPx == cv::Point2f(204, 201), "Near components blended or peak substituted for mass");
    });
    check("causal_warmup_never_uses_yolo_to_pad_temporal_state", [] {
        const auto result = choose({yoloAt(200, 200, 1)}, {}, false);
        require(!result.selected && result.reason == "BLURBALL_CAUSAL_WARMUP", "Warmup created a fused observed ball");
    });
    check("empty_frame_abstains", [] {
        const auto result = choose({}, {}); require(!result.selected && result.reason == "NO_ELIGIBLE_CANDIDATE", "Empty observations fabricated a ball");
    });
    check("frame_identity_mismatch_fail_closed", [] {
        auto candidates = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200)}); candidates[0].frame.frameId++;
        require(select(frame(), yoloHash, blurHash, candidates, {}, true).reason == "FRAME_IDENTITY_MISMATCH", "Cross-frame hypothesis accepted");
    });
    check("source_image_time_and_dimension_identity_fail_closed", [] {
        const auto base = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200)});
        for (int change = 0; change < 4; ++change) {
            auto candidates = base;
            if (change == 0) candidates[0].frame.sourceSha256 = std::string(64, 'e');
            if (change == 1) candidates[0].frame.decodedImageSha256 = std::string(64, 'e');
            if (change == 2) candidates[0].frame.timestamp += .001;
            if (change == 3) candidates[0].frame.imageSize.width++;
            require(select(frame(), yoloHash, blurHash, candidates, {}, true).reason == "FRAME_IDENTITY_MISMATCH", "Mismatched source/image/time/size accepted");
        }
    });
    check("model_hash_kind_and_candidate_index_fail_closed", [] {
        const auto base = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200)});
        for (int change = 0; change < 3; ++change) {
            auto candidates = base;
            if (change == 0) candidates[0].modelSha256 = blurHash;
            if (change == 1) candidates[0].model = Model::BlurBall;
            if (change == 2) candidates[0].index++;
            require(select(frame(), yoloHash, blurHash, candidates, {}, true).reason == "MODEL_OR_CANDIDATE_IDENTITY_MISMATCH", "Mismatched model or candidate identity accepted");
        }
    });
    check("invalid_outputs_are_retained_and_never_selected", [] {
        auto nanScore = yoloAt(200, 200); nanScore.confidence = std::numeric_limits<float>::quiet_NaN();
        auto negativeBox = yoloAt(200, 200); negativeBox.w = -1;
        auto otherClass = yoloAt(200, 200); otherClass.class_id = 1;
        const auto candidates = normalizeYolo(frame(), yoloHash, {nanScore, negativeBox, otherClass, yoloAt(1920, 200)});
        require(candidates.size() == 4 && std::all_of(candidates.begin(), candidates.end(), [](const auto& c) { return !c.valid; }), "Malformed outputs not retained/rejected");
        const auto result = select(frame(), yoloHash, blurHash, candidates, {}, true);
        require(!result.selected && result.invalidCandidates == 4, "Invalid output reached selected observation");
    });
    check("finite_center_in_source_allows_box_crossing_image_border", [] {
        const auto candidates = normalizeYolo(frame(), yoloHash, {yoloAt(0, 0)});
        require(candidates[0].valid && candidates[0].raw.x < 0 && candidates[0].centerPx == cv::Point2f(0, 0), "Legitimate border center was clipped or rejected");
    });
    check("blurball_invalid_mass_peak_axis_endpoints_are_rejected", [] {
        std::vector<BlurBallObservation> raw(5, blurAt(200, 200));
        raw[0].componentMass = 0; raw[1].heatmapPeak = .8f; raw[2].blurAxisDeg = 180;
        raw[3].blurEndpoints[0].x = std::numeric_limits<float>::infinity(); raw[4].pixelCenter.y = std::numeric_limits<float>::quiet_NaN();
        const auto candidates = normalizeBlurBall(frame(), blurHash, raw);
        require(std::none_of(candidates.begin(), candidates.end(), [](const auto& c) { return c.valid; }), "Invalid BlurBall evidence accepted");
    });
    check("tampered_valid_flag_and_source_center_are_rechecked", [] {
        auto candidates = normalizeYolo(frame(), yoloHash, {yoloAt(200, 200)}); candidates[0].centerPx.x += 5;
        const auto result = select(frame(), yoloHash, blurHash, candidates, {}, true);
        require(!result.selected && result.invalidCandidates == 1, "Externally changed center accepted because valid flag was true");
    });
    check("invalid_frame_and_model_contracts_throw", [] {
        for (int change = 0; change < 6; ++change) {
            auto f = frame();
            if (change == 0) f.sourceSha256 = "unknown";
            if (change == 1) f.decodedImageSha256 = std::string(64, 'g');
            if (change == 2) f.frameId = 0;
            if (change == 3) f.timestamp = -.1;
            if (change == 4) f.timestamp = std::numeric_limits<double>::quiet_NaN();
            if (change == 5) f.imageSize.height = 0;
            invalidArgument([&] { normalizeYolo(f, yoloHash, {}); });
        }
        invalidArgument([] { normalizeBlurBall(frame(), "unknown", {}); });
    });
    check("invalid_or_inconsistent_configurations_throw", [] {
        for (int change = 0; change < 7; ++change) {
            Config config;
            if (change == 0) config.agreementRadiusPx = 0;
            if (change == 1) config.agreementRadiusPx = std::numeric_limits<double>::infinity();
            if (change == 2) config.yoloProposalScore = .3f;
            if (change == 3) config.yoloSoloScore = .1f;
            if (change == 4) config.blurballAgreementPeak = .6f;
            if (change == 5) config.ambiguityRatio = 1.1;
            if (change == 6) config.blurballSoloPeak = std::numeric_limits<float>::quiet_NaN();
            invalidArgument([&] { choose({}, {}, true, config); });
        }
    });
    std::string names = "[";
    for (const auto& item : passed) { if (names.size() > 1) names += ','; names += recording::quote(item); }
    names += ']';
    const auto report = recording::json({{"passed", "true"}, {"check_count", std::to_string(passed.size())},
        {"selector_revision", recording::quote(selectorRevision)}, {"checks", names}, {"gpu_inference_run", "false"},
        {"accuracy_measured", "false"}, {"production_backend_modified", "false"}});
    if (argc == 2) recording::writeText(std::filesystem::u8path(argv[1]), report);
    std::cout << report << '\n'; return 0;
} catch (const std::exception& error) { std::cerr << "[test_ball_candidate_selection] " << error.what() << '\n'; return 1; }
