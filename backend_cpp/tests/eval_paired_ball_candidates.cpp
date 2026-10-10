#include "ball_candidate_selection.hpp"
#include "blurball_stream.hpp"
#include "recording_controller.hpp"
#include <chrono>
#include <deque>
#include <iostream>
#include <limits>
#include <set>

// Sequential offline research capture. Both models see the same original BGR
// source frame, BlurBall uses past/current channel 2 only, and labels are never
// loaded. This executable does not alter production model selection or weights.
using namespace ball_candidates;
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out; out << std::setprecision(std::numeric_limits<double>::max_digits10) << value; return out.str();
}
std::string point(cv::Point2f value) { return "[" + number(value.x) + "," + number(value.y) + "]"; }
void traceStage(int frame, const char* stage, const char* phase, Clock::time_point evaluationStart,
                std::optional<double> stageMs = std::nullopt, const std::string& detail = "") {
    if (frame > 3) return;
    // Flush each boundary: a stalled native operation leaves its BEGIN visible.
    std::cerr << "[Paired stage trace] " << recording::json({{"frame_id", std::to_string(frame)},
        {"stage", recording::quote(stage)}, {"phase", recording::quote(phase)},
        {"evaluation_elapsed_ms", number(milliseconds(evaluationStart))},
        {"stage_wall_ms", stageMs ? number(*stageMs) : "null"},
        {"detail", detail.empty() ? "null" : recording::quote(detail)}}) << std::endl;
}
double numeric(const std::string& text) {
    size_t consumed = 0; const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) throw std::invalid_argument("Invalid finite numeric argument: " + text);
    return value;
}
struct FileReadLock {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    explicit FileReadLock(const std::filesystem::path& path) {
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot lock original source/model for stable read: " + path.u8string());
    }
    ~FileReadLock() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
    explicit FileReadLock(const std::filesystem::path&) { throw std::runtime_error("Paired evaluation currently requires Windows source locks and SHA-256"); }
#endif
    FileReadLock(const FileReadLock&) = delete; FileReadLock& operator=(const FileReadLock&) = delete;
};
class Digest {
public:
#ifdef _WIN32
    Digest() {
        if (!CryptAcquireContextW(&provider_, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) throw std::runtime_error("Cannot create diagnostic SHA-256 provider");
        if (!CryptCreateHash(provider_, CALG_SHA_256, 0, 0, &hash_)) { CryptReleaseContext(provider_, 0); provider_ = 0; throw std::runtime_error("Cannot create diagnostic SHA-256 hash"); }
    }
    ~Digest() { if (hash_) CryptDestroyHash(hash_); if (provider_) CryptReleaseContext(provider_, 0); }
    void add(const void* data, size_t length) {
        if (length > std::numeric_limits<DWORD>::max() || !CryptHashData(hash_, static_cast<const BYTE*>(data), DWORD(length), 0))
            throw std::runtime_error("Cannot hash diagnostic source pixels/configuration");
    }
    std::string finish() {
        BYTE bytes[32]; DWORD length = 32;
        if (!CryptGetHashParam(hash_, HP_HASHVAL, bytes, &length, 0) || length != 32) throw std::runtime_error("Cannot finalize diagnostic SHA-256");
        std::ostringstream out; for (BYTE byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << int(byte); return out.str();
    }
private: HCRYPTPROV provider_ = 0; HCRYPTHASH hash_ = 0;
#else
    Digest() { throw std::runtime_error("Diagnostic SHA-256 unavailable on this platform"); }
    void add(const void*, size_t) {} std::string finish() { return ""; }
#endif
};
std::string imageDigest(const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC3) throw std::runtime_error("Decoded source must be nonempty BGR8");
    Digest digest;
    const uint32_t geometry[] = {uint32_t(image.cols), uint32_t(image.rows), uint32_t(image.type())}; digest.add(geometry, sizeof(geometry));
    for (int row = 0; row < image.rows; ++row) digest.add(image.ptr(row), size_t(image.cols) * 3);
    return digest.finish();
}
std::string textDigest(const std::string& value) { Digest digest; digest.add(value.data(), value.size()); return digest.finish(); }
uint64_t sharedIndexBytes(const std::filesystem::path& path) {
#ifdef _WIN32
    // Directory attributes may cache a zero file length while a writer is open.
    // Query the live file handle, after append has flushed the index entry.
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open committed index for progress audit");
    LARGE_INTEGER length{}; const BOOL ok = GetFileSizeEx(handle, &length); CloseHandle(handle);
    if (!ok || length.QuadPart < 0) throw std::runtime_error("Cannot read live committed index length");
    return uint64_t(length.QuadPart);
#else
    std::ifstream index(path, std::ios::binary | std::ios::ate);
    const auto length = index.tellg();
    if (!index || length < 0) throw std::runtime_error("Cannot read live committed index length");
    return uint64_t(length);
#endif
}
std::string configJson(const Config& config) {
    return recording::json({{"revision", recording::quote(selectorRevision)}, {"experimental", "true"}, {"thresholds_fitted", "false"},
        {"yolo_proposal_score", number(config.yoloProposalScore)}, {"yolo_agreement_score", number(config.yoloAgreementScore)},
        {"yolo_solo_score", number(config.yoloSoloScore)}, {"blurball_agreement_peak", number(config.blurballAgreementPeak)},
        {"blurball_solo_peak", number(config.blurballSoloPeak)}, {"agreement_radius_source_px", number(config.agreementRadiusPx)},
        {"within_model_ambiguity_ratio", number(config.ambiguityRatio)}, {"coordinate_policy", recording::quote("CHOOSE_EXISTING_BLURBALL_CENTER_OR_YOLO_BOX_CENTER_NO_BLEND")},
        {"score_policy", recording::quote("RAW_MODEL_EVIDENCE_NEVER_AVERAGED_OR_PROBABILITY_CALIBRATED")}, {"warmup_policy", recording::quote("ABSTAIN_UNTIL_THREE_REAL_CONSECUTIVE_SOURCE_FRAMES")}});
}
std::string candidateJson(const Candidate& candidate) {
    recording::Object fields{{"source_sha256", recording::quote(candidate.frame.sourceSha256)}, {"decoded_image_sha256", recording::quote(candidate.frame.decodedImageSha256)},
        {"frame_id", std::to_string(candidate.frame.frameId)}, {"source_pts_s", number(candidate.frame.timestamp)},
        {"source_size", "[" + std::to_string(candidate.frame.imageSize.width) + "," + std::to_string(candidate.frame.imageSize.height) + "]"},
        {"model", recording::quote(name(candidate.model))}, {"model_sha256", recording::quote(candidate.modelSha256)},
        {"candidate_index", std::to_string(candidate.index)}, {"valid", candidate.valid ? "true" : "false"},
        {"invalid_reason", candidate.valid ? "null" : recording::quote(candidate.invalidReason)}, {"center_px", point(candidate.centerPx)},
        {"coordinate_basis", recording::quote("SOURCE_PIXELS")}, {"center_method", recording::quote(candidate.model == Model::Yolo ? "YOLO_BOX_CENTER" : "HEATMAP_WEIGHTED_CENTROID")},
        {"raw_score", number(candidate.raw.confidence)}, {"score_kind", recording::quote(candidate.model == Model::Yolo ? "DETECTOR_SCORE" : "HEATMAP_PEAK")},
        {"confidence", "null"}, {"confidence_reason", recording::quote("RAW_MODEL_SCORE_NOT_CORRECTNESS_CALIBRATED")},
        {"raw_detection", recording::json({{"class_id", std::to_string(candidate.raw.class_id)}, {"score", number(candidate.raw.confidence)},
            {"x", number(candidate.raw.x)}, {"y", number(candidate.raw.y)}, {"w", number(candidate.raw.w)}, {"h", number(candidate.raw.h)}})}};
    if (candidate.blur) {
        const auto& blur = *candidate.blur;
        fields["blur_evidence"] = recording::json({{"heatmap_peak", number(blur.heatmapPeak)}, {"component_mass", number(blur.componentMass)},
            {"extent_source_px", number(blur.blurExtentSourcePx)}, {"axis_deg", number(blur.blurAxisDeg)}, {"axis_directed", "false"},
            {"endpoints_px", "[" + point(blur.blurEndpoints[0]) + "," + point(blur.blurEndpoints[1]) + "]"}});
    } else fields["blur_evidence"] = "null";
    return recording::json(fields);
}
std::string candidatesJson(const std::vector<Candidate>& values) {
    std::string out = "["; for (const auto& value : values) { if (out.size() > 1) out += ','; out += candidateJson(value); } return out + "]";
}
std::string selectionJson(const Selection& value) {
    return recording::json({{"selected", value.selected ? candidateJson(*value.selected) : "null"}, {"reason", recording::quote(value.reason)},
        {"confidence", "null"}, {"confidence_reason", recording::quote("EXPERIMENTAL_UNCALIBRATED_SELECTOR")},
        {"yolo_support_index", value.yoloSupportIndex ? std::to_string(*value.yoloSupportIndex) : "null"},
        {"blurball_support_index", value.blurballSupportIndex ? std::to_string(*value.blurballSupportIndex) : "null"},
        {"agreement_distance_source_px", value.agreementDistancePx ? number(*value.agreementDistancePx) : "null"},
        {"invalid_candidates", std::to_string(value.invalidCandidates)}, {"eligible_yolo", std::to_string(value.eligibleYolo)},
        {"eligible_blurball", std::to_string(value.eligibleBlurBall)}});
}
std::optional<Candidate> blurBaseline(const BlurBallStream::Frame& result, const std::vector<Candidate>& candidates) {
    if (!result.selected) return std::nullopt;
    for (const auto& candidate : candidates) {
        const auto& observation = *candidate.blur;
        if (observation.pixelCenter == result.selected->pixelCenter && observation.componentMass == result.selected->componentMass &&
            observation.heatmapPeak == result.selected->heatmapPeak) {
            if (!candidate.valid) throw std::runtime_error("Standalone BlurBall selected an invalid source candidate");
            return candidate;
        }
    }
    throw std::runtime_error("Standalone BlurBall selected observation missing from causal candidate output");
}
}
int runEvaluation(int argc, char** argv) {
    std::filesystem::path directory;
    std::unique_ptr<recording::JournalWriter> journal;
    int decoded = 0, committed = 0, expected = 0, declared = 0;
    const auto evaluationStart = Clock::now();
    try {
        if (argc < 5) throw std::invalid_argument("Usage: eval_paired_ball_candidates YOLO.engine BLURBALL.engine ORIGINAL_VIDEO NEW_OUTPUT_DIRECTORY [--limit N] [--proposal-threshold S] [--nms-threshold S] [--agreement-radius-px R] [--yolo-agreement-score S] [--yolo-solo-score S] [--blurball-solo-peak S] [--ambiguity-ratio R] [--maximum-temporal-gap-s T]");
        Config config; int limit = 0; double nmsThreshold = .45, standaloneYoloThreshold = .25, maximumGap = 0;
        std::set<std::string> options;
        for (int index = 5; index < argc; ++index) {
            const std::string option = argv[index];
            if (!options.insert(option).second || index + 1 >= argc) throw std::invalid_argument("Repeated or incomplete option: " + option);
            const double value = numeric(argv[++index]);
            if (option == "--limit") { if (value < 1 || value > std::numeric_limits<int>::max() || std::floor(value) != value) throw std::invalid_argument("Limit must be a positive integer"); limit = int(value); }
            else if (option == "--proposal-threshold") config.yoloProposalScore = float(value);
            else if (option == "--nms-threshold") nmsThreshold = value;
            else if (option == "--agreement-radius-px") config.agreementRadiusPx = value;
            else if (option == "--yolo-agreement-score") config.yoloAgreementScore = float(value);
            else if (option == "--yolo-solo-score") config.yoloSoloScore = float(value);
            else if (option == "--blurball-solo-peak") config.blurballSoloPeak = float(value);
            else if (option == "--ambiguity-ratio") config.ambiguityRatio = value;
            else if (option == "--maximum-temporal-gap-s") maximumGap = value;
            else throw std::invalid_argument("Unknown option: " + option);
        }
        validate(config);
        if (nmsThreshold <= 0 || nmsThreshold > 1 || maximumGap < 0 || config.yoloProposalScore > standaloneYoloThreshold)
            throw std::invalid_argument("Invalid NMS/temporal gap or proposal threshold excludes standalone YOLO baseline");
        const auto yoloPath = std::filesystem::absolute(std::filesystem::u8path(argv[1]));
        const auto blurPath = std::filesystem::absolute(std::filesystem::u8path(argv[2]));
        const auto source = std::filesystem::absolute(std::filesystem::u8path(argv[3]));
        directory = std::filesystem::absolute(std::filesystem::u8path(argv[4]));
        for (const auto& file : {yoloPath, blurPath, source}) if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("Original local source/model file missing: " + file.u8string());
        if (std::filesystem::exists(directory)) throw std::runtime_error("Refusing to overwrite/mix an existing paired evaluation directory");
        FileReadLock sourceLock(source), yoloLock(yoloPath), blurLock(blurPath);
        cv::VideoCapture capture(source.u8string(), cv::CAP_FFMPEG);
        if (!capture.isOpened()) throw std::runtime_error("Cannot decode original local source recording");
        const double fps = capture.get(cv::CAP_PROP_FPS), declaredCount = capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (!std::isfinite(fps) || fps <= 0 || !std::isfinite(declaredCount) || declaredCount < 1 || declaredCount > std::numeric_limits<int>::max() || std::floor(declaredCount) != declaredCount)
            throw std::runtime_error("Source requires a valid nominal FPS and declared frame count for complete-coverage audit");
        declared = int(declaredCount); expected = limit ? std::min(limit, declared) : declared;
        if (maximumGap == 0) maximumGap = 2.5 / fps;
        TensorRTEngine yolo(yoloPath.u8string(), 640, 640, 1);
        if (!yolo.init()) throw std::runtime_error("YOLO initialization failed; no fallback hypotheses");
        BlurBallEngine blur(blurPath.u8string()); BlurBallStream stream(blur);
        const auto sourceHash = recording::sha256(source), yoloHash = yolo.getSha256Hash(), blurHash = blur.getSha256Hash();
        const auto configuration = configJson(config), fusionHash = textDigest(configuration);
        const auto runtimeHash = recording::sha256(std::filesystem::u8path(argv[0]));
        const auto manifest = recording::json({{"schema_version", "1"}, {"evaluation", recording::quote("PAIRED_CAUSAL_NATIVE_CANDIDATE_REPLAY")},
            {"source_path", recording::quote(source.u8string())}, {"source_sha256", recording::quote(sourceHash)}, {"source_fps", number(fps)},
            {"source_declared_frames", std::to_string(declared)}, {"requested_frames", std::to_string(expected)},
            {"source_interval", recording::quote(limit ? "CHRONOLOGICAL_PREFIX_ONLY" : "COMPLETE_RECORDING")},
            {"timestamp_basis", recording::quote("ACTUAL_OPENCV_FFMPEG_DECODER_PTS_SECONDS")}, {"invalid_pts_policy", recording::quote("FAIL_CLOSED_PARTIAL_JOURNAL")},
            {"maximum_temporal_gap_s", number(maximumGap)}, {"gap_policy", recording::quote("RESET_CAUSAL_HISTORY_AND_STANDALONE_ANCHOR")},
            {"cut_audit", recording::quote("NOT_PERFORMED_PIXEL_DUPLICATES_RECORDED_NOT_ASSUMED_TO_BE_CUTS")},
            {"yolo_model_path", recording::quote(yoloPath.u8string())}, {"yolo_model_sha256", recording::quote(yoloHash)},
            {"blurball_model_path", recording::quote(blurPath.u8string())}, {"blurball_model_sha256", recording::quote(blurHash)},
            {"runtime_sha256", recording::quote(runtimeHash)}, {"fusion_sha256", recording::quote(fusionHash)}, {"fusion", configuration},
            {"preprocessing", recording::json({{"yolo", recording::quote("opencv-bgr-rgb-letterbox-640-chw-div255-v1")}, {"blurball", recording::quote("opencv-affine-rgb-imagenet-512x288-three-frames-v1")},
                {"input_transform", recording::quote("RAW_SOURCE_NO_FILTER")}, {"decoded_image_hash", recording::quote("SHA256_UINT32_LE_WIDTH_HEIGHT_OPENCV_TYPE_THEN_PACKED_BGR_ROWS")}})},
            {"yolo_proposal_threshold", number(config.yoloProposalScore)}, {"yolo_nms_threshold", number(float(nmsThreshold))},
            {"standalone_yolo_threshold", number(standaloneYoloThreshold)}, {"standalone_blurball_selection", recording::quote("HIGHEST_COMPONENT_MASS_STRICT_300_SOURCE_PIXEL_ANCHOR_GATE")},
            {"blurball_heatmap_threshold", number(BlurBallEngine::heatmapThreshold)}, {"temporal_policy", recording::quote("CAUSAL_CHANNEL_2_T_MINUS_2_T_MINUS_1_T")},
            {"lookahead_frames", "0"}, {"warmup_frames", "2"}, {"source_images_shared", "true"}, {"opened_camera", "false"},
            {"labels_loaded", "false"}, {"confidence_calibrated", "false"}, {"promotion_approved", "false"}, {"experimental", "true"},
            {"timing_context", recording::quote("OFFLINE_SEQUENTIAL_SOURCE_REPLAY_NOT_LIVE_LATENCY_OR_MULTIPASS_BENCHMARK")},
            {"stage_trace_policy", recording::quote("FIRST_THREE_FRAMES_FLUSHED_DIAGNOSTICS_TRACE_COST_INCLUDED_IN_OUTER_WALL_TIMES")},
            {"progress_policy", recording::quote("EVERY_500_COMMITTED_FRAMES_FLUSHED_LIVE_SHARED_INDEX_HANDLE_LENGTH")},
            {"yolo_timing_entrypoint", recording::quote("CHECKED_INFERFRAME_WALL_ONLY_GPU_NOT_MEASURED")}});
        if (!std::filesystem::create_directory(directory)) throw std::runtime_error("Output directory was claimed by another evaluation");
        journal = std::make_unique<recording::JournalWriter>(directory, manifest);
        std::deque<int> windowIds; double previousPts = -1, firstPts = 0, lastPts = 0;
        std::string previousImageHash; cv::Size previousSize;
        std::vector<double> intervals; int duplicates = 0, resets = 0, warmup = 0, yoloSelected = 0, blurSelected = 0, hybridSelected = 0;
        double yoloWallMs = 0, blurWallMs = 0, blurGpuMs = 0;
        bool eof = false;
        while (!limit || decoded < expected) {
            const auto frameStart = Clock::now(); cv::Mat image;
            const int tracingFrame = decoded + 1;
            traceStage(tracingFrame, "CAPTURE_READ", "BEGIN", evaluationStart);
            const bool frameRead = capture.read(image);
            traceStage(tracingFrame, "CAPTURE_READ", "END", evaluationStart, milliseconds(frameStart),
                std::string("read=") + (frameRead ? "true" : "false") + ";empty=" + (image.empty() ? "true" : "false"));
            if (!frameRead || image.empty()) { eof = true; break; }
            ++decoded;
            const double pts = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.;
            if (!std::isfinite(pts) || pts < 0 || (previousPts >= 0 && pts <= previousPts))
                throw std::runtime_error("Nonfinite/non-increasing decoder PTS at decoded source frame " + std::to_string(decoded) + "; no nominal-timestamp substitution");
            if (decoded == 1) firstPts = pts;
            const double interval = previousPts < 0 ? 0 : pts - previousPts;
            if (previousPts >= 0) intervals.push_back(interval);
            std::string resetReason;
            if (previousPts >= 0 && interval > maximumGap) resetReason = "SOURCE_PTS_GAP";
            if (previousSize.area() && previousSize != image.size()) resetReason = "SOURCE_DIMENSION_CHANGE";
            if (!resetReason.empty()) { stream.reset(); windowIds.clear(); ++resets; }
            traceStage(decoded, "SOURCE_IMAGE_DIGEST", "BEGIN", evaluationStart);
            const auto hashStart = Clock::now();
            const std::string pixelsHash = imageDigest(image);
            traceStage(decoded, "SOURCE_IMAGE_DIGEST", "END", evaluationStart, milliseconds(hashStart));
            const bool duplicate = !previousImageHash.empty() && pixelsHash == previousImageHash;
            if (duplicate) ++duplicates;
            const FrameIdentity identity{sourceHash, pixelsHash, uint64_t(decoded), pts, image.size()};
            traceStage(decoded, "YOLO_PREPROCESS", "BEGIN", evaluationStart);
            const auto prepStart = Clock::now(); const auto input = TensorRTEngine::preprocessOpenCVFrame(image);
            const double preprocessMs = milliseconds(prepStart);
            traceStage(decoded, "YOLO_PREPROCESS", "END", evaluationStart, preprocessMs);
            traceStage(decoded, "YOLO_INFER", "BEGIN", evaluationStart);
            const auto yoloStart = Clock::now();
            const auto yoloRaw = yolo.inferFrame(input, image.cols, image.rows, config.yoloProposalScore, float(nmsThreshold));
            const double yoloMs = milliseconds(yoloStart); yoloWallMs += yoloMs;
            traceStage(decoded, "YOLO_INFER", "END", evaluationStart, yoloMs, "proposals=" + std::to_string(yoloRaw.size()));
            traceStage(decoded, "BLURBALL_INFER", "BEGIN", evaluationStart);
            const auto blurStart = Clock::now(); const auto blurResult = stream.causal(image, decoded, pts);
            const double blurMs = milliseconds(blurStart); blurWallMs += blurMs; blurGpuMs += blurResult.gpuMs;
            traceStage(decoded, "BLURBALL_INFER", "END", evaluationStart, blurMs, "components=" + std::to_string(blurResult.candidates.size()));
            traceStage(decoded, "POST_INFERENCE_IMAGE_DIGEST", "BEGIN", evaluationStart);
            const auto repeatedHashStart = Clock::now();
            if (imageDigest(image) != pixelsHash) throw std::runtime_error("A detector mutated the shared original source pixels");
            traceStage(decoded, "POST_INFERENCE_IMAGE_DIGEST", "END", evaluationStart, milliseconds(repeatedHashStart));
            windowIds.push_back(decoded); if (windowIds.size() > 3) windowIds.pop_front();
            const bool ready = windowIds.size() == 3;
            if (!ready) ++warmup;
            traceStage(decoded, "NORMALIZE_AND_SELECT", "BEGIN", evaluationStart);
            const auto selectionStart = Clock::now();
            const auto yoloCandidates = normalizeYolo(identity, yoloHash, yoloRaw);
            const auto blurCandidates = normalizeBlurBall(identity, blurHash, blurResult.candidates);
            std::optional<Candidate> yoloBaseline;
            for (const auto& candidate : yoloCandidates)
                if (candidate.valid && candidate.raw.confidence >= standaloneYoloThreshold && (!yoloBaseline || candidate.raw.confidence > yoloBaseline->raw.confidence)) yoloBaseline = candidate;
            const auto blurSelectedCandidate = blurBaseline(blurResult, blurCandidates);
            const auto selected = select(identity, yoloHash, blurHash, yoloCandidates, blurCandidates, ready, config);
            traceStage(decoded, "NORMALIZE_AND_SELECT", "END", evaluationStart, milliseconds(selectionStart), selected.reason);
            yoloSelected += yoloBaseline.has_value(); blurSelected += blurSelectedCandidate.has_value(); hybridSelected += selected.selected.has_value();
            std::string window = "["; for (int id : windowIds) { if (window.size() > 1) window += ','; window += std::to_string(id); } window += ']';
            traceStage(decoded, "SERIALIZE_ROW", "BEGIN", evaluationStart);
            const auto serializationStart = Clock::now();
            const auto telemetry = recording::json({{"frame_id", std::to_string(decoded)}, {"source_sha256", recording::quote(sourceHash)},
                {"decoded_image_sha256", recording::quote(pixelsHash)}, {"source_pts_s", number(pts)}, {"nominal_source_time_s", number((decoded - 1) / fps)},
                {"source_size", "[" + std::to_string(image.cols) + "," + std::to_string(image.rows) + "]"}, {"same_decoded_pixels_as_previous", duplicate ? "true" : "false"},
                {"temporal_window_frame_ids", window}, {"temporal_window_ready", ready ? "true" : "false"}, {"temporal_reset_reason", resetReason.empty() ? "null" : recording::quote(resetReason)},
                {"yolo_candidates", candidatesJson(yoloCandidates)}, {"blurball_candidates", candidatesJson(blurCandidates)},
                {"standalone_yolo", yoloBaseline ? candidateJson(*yoloBaseline) : "null"}, {"standalone_blurball", blurSelectedCandidate ? candidateJson(*blurSelectedCandidate) : "null"},
                {"experimental_selection", selectionJson(selected)}, {"fusion_sha256", recording::quote(fusionHash)},
                {"timings", recording::json({{"yolo_preprocess_wall_ms", number(preprocessMs)}, {"yolo_inference_wall_ms", number(yoloMs)},
                    {"yolo_gpu_ms", "null"}, {"yolo_gpu_timing_reason", recording::quote("NOT_MEASURED_CHECKED_INFERFRAME_ENTRYPOINT")},
                    {"blurball_adapter_wall_ms", number(blurMs)}, {"blurball_engine_wall_ms", number(blurResult.inferMs)},
                    {"blurball_gpu_ms", ready ? number(blurResult.gpuMs) : "null"}, {"blurball_inference_executed", ready ? "true" : "false"},
                    {"frame_before_journal_wall_ms", number(milliseconds(frameStart))}, {"live_latency_measured", "false"}})}});
            traceStage(decoded, "SERIALIZE_ROW", "END", evaluationStart, milliseconds(serializationStart), "bytes=" + std::to_string(telemetry.size()));
            traceStage(decoded, "JOURNAL_APPEND", "BEGIN", evaluationStart);
            const auto appendStart = Clock::now();
            journal->append(decoded, telemetry);
            traceStage(decoded, "JOURNAL_APPEND", "END", evaluationStart, milliseconds(appendStart));
            committed = decoded; previousPts = lastPts = pts; previousImageHash = pixelsHash; previousSize = image.size();
            if (decoded % 500 == 0) {
                const uint64_t indexBytes = sharedIndexBytes(directory / "frames.idx");
                if (indexBytes != uint64_t(committed) * sizeof(uint64_t)) throw std::runtime_error("Live index progress disagrees with committed source-frame count");
                std::cout << "[Paired progress] " << recording::json({{"source_frames_decoded", std::to_string(decoded)},
                    {"indexed_frames", std::to_string(indexBytes / sizeof(uint64_t))}, {"actual_shared_index_bytes", std::to_string(indexBytes)},
                    {"stage", recording::quote("COMMITTED")}, {"directory_attributes_used", "false"}}) << std::endl;
            }
        }
        if (decoded != expected || (!limit && !eof)) throw std::runtime_error("Sequential decode count disagrees with requested/declared complete coverage; partial journal retained");
        double medianInterval = 0;
        if (!intervals.empty()) { auto sorted = intervals; std::sort(sorted.begin(), sorted.end()); medianInterval = sorted[sorted.size() / 2]; }
        const auto report = recording::json({{"status", recording::quote("COMPLETE")}, {"source_frames_decoded", std::to_string(decoded)}, {"analyzed_frames", std::to_string(committed)},
            {"source_declared_frames", std::to_string(declared)}, {"source_count_matches_metadata", decoded == declared ? "true" : "false"},
            {"entire_recording_decoded", eof && decoded == declared ? "true" : "false"}, {"decoder_eof_reached", eof ? "true" : "false"},
            {"source_pts_strictly_increasing", "true"}, {"source_pts_first_s", number(firstPts)}, {"source_pts_last_s", number(lastPts)}, {"median_source_interval_s", intervals.empty() ? "null" : number(medianInterval)},
            {"consecutive_pixel_duplicates", std::to_string(duplicates)}, {"temporal_resets", std::to_string(resets)}, {"causal_warmup_frames", std::to_string(warmup)},
            {"standalone_yolo_selected_frames", std::to_string(yoloSelected)}, {"standalone_blurball_selected_frames", std::to_string(blurSelected)}, {"experimental_selected_frames", std::to_string(hybridSelected)},
            {"source_sha256", recording::quote(sourceHash)}, {"yolo_model_sha256", recording::quote(yoloHash)}, {"blurball_model_sha256", recording::quote(blurHash)}, {"fusion_sha256", recording::quote(fusionHash)},
            {"yolo_inference_total_wall_ms", number(yoloWallMs)}, {"yolo_gpu_total_ms", "null"}, {"blurball_adapter_total_wall_ms", number(blurWallMs)}, {"blurball_gpu_total_ms", number(blurGpuMs)},
            {"evaluation_total_wall_ms", number(milliseconds(evaluationStart))}, {"lookahead_frames", "0"}, {"opened_camera", "false"}, {"accuracy_measured", "false"},
            {"live_latency_measured", "false"}, {"promotion_approved", "false"}, {"labels_loaded", "false"}, {"original_source_mutated", "false"}, {"production_backend_modified", "false"}});
        recording::publishText(directory / "diagnostic_summary.json", report); journal->finish(expected);
        std::cout << report << '\n'; return 0;
    } catch (const std::exception& error) {
        if (journal) {
            try {
                const auto failure = recording::json({{"status", recording::quote("PARTIAL")}, {"error", recording::quote(error.what())}, {"source_frames_decoded", std::to_string(decoded)},
                    {"analyzed_frames", std::to_string(committed)}, {"total_frames", std::to_string(expected)}, {"source_declared_frames", std::to_string(declared)},
                    {"accuracy_measured", "false"}, {"promotion_approved", "false"}, {"failed_closed", "true"}});
                recording::publishText(directory / "error.json", failure); recording::publishText(directory / "completion.json", failure);
            } catch (const std::exception& writeError) { std::cerr << "[eval_paired_ball_candidates] Partial-output status write also failed: " << writeError.what() << '\n'; }
        }
        std::cerr << "[eval_paired_ball_candidates] " << error.what() << '\n'; return 1;
    }
}

// Keep original source paths in UTF-8, including space/non-ASCII Windows names.
// The CRT narrow argv otherwise uses the active ANSI code page.
#ifdef _WIN32
int wmain(int argc, wchar_t** arguments) {
    std::vector<std::string> utf8; utf8.reserve(argc);
    for (int index = 0; index < argc; ++index) {
        const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, arguments[index], -1, nullptr, 0, nullptr, nullptr);
        if (length <= 0) { std::cerr << "[eval_paired_ball_candidates] Invalid Unicode command line\n"; return 1; }
        std::string value(length, '\0');
        if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, arguments[index], -1, value.data(), length, nullptr, nullptr) != length) {
            std::cerr << "[eval_paired_ball_candidates] Unicode command conversion failed\n"; return 1;
        }
        value.pop_back(); utf8.push_back(std::move(value));
    }
    std::vector<char*> argv; for (auto& value : utf8) argv.push_back(value.data());
    return runEvaluation(argc, argv.data());
}
#else
int main(int argc, char** argv) { return runEvaluation(argc, argv); }
#endif
