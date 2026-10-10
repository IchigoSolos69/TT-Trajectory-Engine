#include "blurball_stream.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace {
void require(bool condition, const std::string& text) { if (!condition) throw std::runtime_error(text); }
void same(const std::vector<BlurBallObservation>& first, const std::vector<BlurBallObservation>& second, const std::string& context) {
    require(first.size() == second.size(), context + " component count mismatch");
    for (size_t index = 0; index < first.size(); ++index) {
        require(cv::norm(first[index].pixelCenter - second[index].pixelCenter) < .001, context + " source centroid mismatch");
        require(std::abs(first[index].componentMass - second[index].componentMass) < .001, context + " component mass mismatch");
    }
}
std::optional<BlurBallObservation> expectedSelection(const std::vector<BlurBallObservation>& candidates, const std::optional<cv::Point2f>& previous) {
    const BlurBallObservation* best = nullptr;
    for (const auto& observation : candidates) {
        if (previous && cv::norm(observation.pixelCenter - *previous) >= 300) continue;
        if (!best || observation.componentMass > best->componentMass) best = &observation;
    }
    return best ? std::optional<BlurBallObservation>(*best) : std::nullopt;
}
}
int main(int argc, char** argv) try {
    if (argc == 5 && std::string(argv[4]) == "--probe-decoder") {
        cv::VideoCapture existing(argv[2], cv::CAP_FFMPEG); cv::Mat probe;
        const double reported = existing.get(cv::CAP_PROP_FRAME_COUNT);
        const bool firstRead = existing.read(probe); const double firstPosition = existing.get(cv::CAP_PROP_POS_FRAMES);
        const bool rewindAccepted = existing.set(cv::CAP_PROP_POS_FRAMES, 0);
        const bool rewindRead = existing.read(probe); const double rewindPosition = existing.get(cv::CAP_PROP_POS_FRAMES);
        existing.release(); cv::VideoCapture fresh(argv[2], cv::CAP_FFMPEG);
        const bool freshRead = fresh.read(probe); const double freshPosition = fresh.get(cv::CAP_PROP_POS_FRAMES);
        std::cout << "{\"reported_frames\":" << reported << ",\"first_read\":" << firstRead
                  << ",\"first_position\":" << firstPosition << ",\"rewind_accepted\":" << rewindAccepted
                  << ",\"rewind_read\":" << rewindRead << ",\"rewind_position\":" << rewindPosition
                  << ",\"fresh_read\":" << freshRead << ",\"fresh_position\":" << freshPosition << "}\n";
        return 0;
    }
    if (argc != 4) throw std::runtime_error("Usage: test_blurball_stream engine original_video test_output_directory [--probe-decoder]");
    const auto output = std::filesystem::u8path(argv[3]);
    require(!std::filesystem::exists(output), "Refusing to mix stream test runs");
    std::filesystem::create_directories(output);
    BlurBallEngine engine(argv[1]);
    cv::VideoCapture oracleCapture(argv[2], cv::CAP_FFMPEG);
    require(oracleCapture.isOpened(), "Cannot decode source for stream oracle");
    const double fps = oracleCapture.get(cv::CAP_PROP_FPS);
    const int total = static_cast<int>(oracleCapture.get(cv::CAP_PROP_FRAME_COUNT));
    require(std::isfinite(fps) && fps > 0 && total >= 12, "Need valid source timing and at least 12 real frames");
    std::vector<std::vector<BlurBallObservation>> recordingCandidates(total), causalCandidates(total);
    std::deque<cv::Mat> window; std::vector<cv::Mat> firstFrames;
    int count = 0;
    cv::Mat image;
    while (oracleCapture.read(image)) {
        require(count < total, "Decoder reports more frames than source metadata");
        if (count < 12) firstFrames.push_back(image.clone());
        window.push_back(image.clone()); if (window.size() > 3) window.pop_front();
        if (window.size() == 3) {
            const auto result = engine.infer({window[0], window[1], window[2]});
            for (int channel = 0; channel < 3; ++channel) {
                auto& items = recordingCandidates[count - 2 + channel];
                items.insert(items.end(), result.observations[channel].begin(), result.observations[channel].end());
            }
            causalCandidates[count] = result.observations[2];
        }
        ++count;
    }
    require(count == total, "Oracle did not decode every source frame");
    oracleCapture.release();
    cv::VideoCapture recording(argv[2], cv::CAP_FFMPEG), sameSource(argv[2], cv::CAP_FFMPEG);
    BlurBallStream stream(engine); BlurBallStream::Frame frame; std::optional<cv::Point2f> anchor;
    int recordingFrames = 0;
    while (stream.nextRecording(recording, fps, frame)) {
        const int index = recordingFrames++;
        require(index < total && frame.id == index + 1, "Recording output skipped or duplicated a source frame");
        require(std::abs(frame.timestamp - index / fps) < 1.e-9, "Recording timestamp mapped to wrong source frame");
        require(sameSource.read(image) && cv::norm(image, frame.image, cv::NORM_INF) == 0,
                "Recording imagery does not match its declared source frame");
        same(frame.candidates, recordingCandidates[index], "Recording frame " + std::to_string(frame.id));
        const auto expected = expectedSelection(recordingCandidates[index], anchor);
        require(frame.selected.has_value() == expected.has_value(), "Recording selected presence mismatch");
        if (expected) {
            require(cv::norm(frame.selected->pixelCenter - expected->pixelCenter) < .001, "Recording selected wrong source candidate");
            anchor = expected->pixelCenter;
        } else anchor.reset();
    }
    require(recordingFrames == total, "Recording tail frames were dropped");
    require(!stream.nextRecording(recording, fps, frame), "EOF repeated an already emitted frame");
    recording.release(); sameSource.release();

    // Current/past-only live policy must exactly match channel 2, including its
    // real two-frame warmup. No camera device is opened in this test.
    stream.reset(); anchor.reset();
    for (int index = 0; index < 12; ++index) {
        frame = stream.causal(firstFrames[index], index + 1, index / fps);
        require(frame.id == index + 1 && frame.timestamp == index / fps, "Live output lost current frame identity");
        same(frame.candidates, causalCandidates[index], "Causal current-frame channel");
        const auto expected = expectedSelection(causalCandidates[index], anchor);
        require(frame.selected.has_value() == expected.has_value(), "Causal selected presence mismatch");
        if (expected) {
            require(cv::norm(frame.selected->pixelCenter - expected->pixelCenter) < .001, "Causal policy selected another output frame");
            anchor = expected->pixelCenter;
        } else anchor.reset();
        if (index < 2) require(frame.candidates.empty() && frame.inferMs == 0 && frame.gpuMs == 0, "Live warmup padded or inferred missing frames");
    }
    // A frame-ID discontinuity starts a fresh causal window.
    frame = stream.causal(firstFrames[0], 20, 20 / fps);
    require(frame.candidates.empty() && frame.inferMs == 0, "Capture gap reused old temporal frames");
    frame = stream.causal(firstFrames[1], 21, 21 / fps);
    require(frame.candidates.empty() && frame.inferMs == 0, "Capture-gap warmup was too short");
    frame = stream.causal(firstFrames[2], 22, 22 / fps);
    same(frame.candidates, causalCandidates[2], "Capture-gap reacquisition");
    // A duplicate source ID must also discard stale history.
    frame = stream.causal(firstFrames[0], 22, 22 / fps);
    require(frame.candidates.empty() && frame.inferMs == 0, "Repeated source ID reused stale temporal history");
    // Explicit reset is the reconnect/source-change boundary.
    stream.reset(); frame = stream.causal(firstFrames[0], 1, 0);
    require(frame.candidates.empty() && frame.inferMs == 0, "Reset retained an old source window");
    cv::Mat resized; cv::resize(firstFrames[0], resized, {640, 360});
    frame = stream.causal(resized, 2, 1 / fps);
    require(frame.candidates.empty() && frame.inferMs == 0, "Resolution change mixed different-size frames");

    // One/two-frame clips stay analyzable but never create artificial temporal
    // detections from repeated padding. They must still emit their full tail.
    for (int length : {1, 2}) {
        const auto file = output / ("short_" + std::to_string(length) + ".avi");
        cv::VideoWriter writer(file.u8string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, firstFrames[0].size());
        require(writer.isOpened(), "Cannot create short-recording fixture");
        for (int index = 0; index < length; ++index) writer.write(firstFrames[index]);
        writer.release(); cv::VideoCapture shortClip(file.u8string(), cv::CAP_FFMPEG);
        require(shortClip.isOpened(), "Cannot decode short-recording fixture");
        stream.reset(); int emitted = 0;
        while (stream.nextRecording(shortClip, fps, frame)) {
            ++emitted; require(frame.id == emitted, "Short recording source ID mismatch");
            require(frame.candidates.empty() && !frame.selected && frame.inferMs == 0 && frame.gpuMs == 0,
                    "Short recording fabricated a padded model observation");
        }
        require(emitted == length, "Short recording omitted its tail");
    }
    std::ofstream report(output / "stream_result.json");
    report << "{\"passed\":true,\"source_frames\":" << total << ",\"recording_frames_emitted\":" << recordingFrames
           << ",\"causal_frames_checked\":12,\"recording_all_window_components_match\":true,\"recording_image_frame_identity\":true"
           << ",\"tail_complete\":true,\"causal_channel_2_only\":true,\"real_frame_warmup\":true,\"id_gap_reset\":true"
           << ",\"duplicate_id_reset\":true,\"explicit_reset\":true,\"resolution_change_reset\":true"
           << ",\"short_recording_counts\":[1,2],\"no_temporal_padding\":true,\"opened_camera\":false}\n";
    report.close(); require(report.good(), "Cannot save stream policy report");
    std::cout << "PASS: " << total << " exact recording frames including tail; 12 causal frames, warmup/reset/gap/short-recording policies\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "[test_blurball_stream] " << error.what() << '\n'; return 1;
}
