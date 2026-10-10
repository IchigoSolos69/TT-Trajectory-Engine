#include "blurball_engine.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
std::string quoted(const std::string& value) {
    std::ostringstream output; output << '"';
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') output << '\\' << character;
        else if (character < 32) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character);
        else output << character;
    }
    output << '"'; return output.str();
}
void point(std::ostream& stream, const cv::Point2f& value) { stream << '[' << value.x << ',' << value.y << ']'; }
void observation(std::ostream& stream, const BlurBallObservation& value) {
    stream << "{\"center_px\":"; point(stream, value.pixelCenter);
    stream << ",\"component_mass\":" << value.componentMass << ",\"heatmap_peak\":" << value.heatmapPeak
           << ",\"blur_extent_source_px\":" << value.blurExtentSourcePx << ",\"blur_axis_deg\":" << value.blurAxisDeg
           << ",\"blur_endpoints_px\":[";
    point(stream, value.blurEndpoints[0]); stream << ','; point(stream, value.blurEndpoints[1]);
    stream << "],\"confidence\":null,\"confidence_reason\":\"Uncalibrated heatmap evidence\",\"detection\":{\"class_id\":0,\"score\":"
           << value.detection.confidence << ",\"x\":" << value.detection.x << ",\"y\":" << value.detection.y
           << ",\"w\":" << value.detection.w << ",\"h\":" << value.detection.h << "}}";
}
}
int main(int argc, char** argv) try {
    if (argc < 4) throw std::runtime_error("Usage: eval_blurball_native engine video output_directory [--all] [--limit N] [--logits]");
    bool all = false, logits = false; int limit = 82;
    for (int index = 4; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--all") all = true;
        else if (option == "--logits") logits = true;
        else if (option == "--limit" && index + 1 < argc) limit = std::stoi(argv[++index]);
        else throw std::runtime_error("Unknown or incomplete evaluation argument: " + option);
    }
    if (limit < 3) throw std::runtime_error("Evaluation requires at least three sequential source frames");
    BlurBallEngine engine(argv[1]);
    cv::VideoCapture video(argv[2], cv::CAP_FFMPEG);
    if (!video.isOpened()) throw std::runtime_error("Cannot decode original BlurBall evaluation video");
    const auto directory = std::filesystem::u8path(argv[3]);
    if (std::filesystem::exists(directory)) throw std::runtime_error("Evaluation directory already exists; refusing to mix results");
    std::filesystem::create_directories(directory);
    std::ofstream rows(directory / "windows.jsonl"), raw, inputs;
    rows << std::setprecision(10);
    if (logits) {
        raw.open(directory / "native_logits.bin", std::ios::binary);
        inputs.open(directory / "native_inputs.bin", std::ios::binary);
    }
    if (!rows || (logits && (!raw || !inputs))) throw std::runtime_error("Cannot open evaluation outputs");
    const std::set<int> targets{3, 4, 5, 40, 41, 42, 80, 81, 82};
    std::array<cv::Mat, 3> frames;
    int decoded = 0, windows = 0;
    const double fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("Video lacks a valid source frame rate");
    auto rejects = [&](const std::array<cv::Mat, 3>& invalid) {
        try { engine.infer(invalid); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    const cv::Mat validInput(16, 16, CV_8UC3, cv::Scalar(0));
    const bool emptyRejected = rejects({cv::Mat(), cv::Mat(), cv::Mat()});
    const bool typeRejected = rejects({validInput, cv::Mat(16, 16, CV_8UC1, cv::Scalar(0)), validInput});
    const bool dimensionsRejected = rejects({validInput, cv::Mat(16, 15, CV_8UC3, cv::Scalar(0)), validInput});
    if (!emptyRejected || !typeRejected || !dimensionsRejected)
        throw std::runtime_error("Engine accepted an invalid temporal input contract");
    while (decoded < limit) {
        cv::Mat image; if (!video.read(image)) break;
        ++decoded;
        frames[0] = std::move(frames[1]); frames[1] = std::move(frames[2]); frames[2] = std::move(image);
        if (decoded < 3) continue;
        if (decoded == 3) engine.infer(frames); // Real-frame warmup, never duplicated padding.
        if (!all && !targets.count(decoded)) continue;
        const auto result = engine.infer(frames, logits);
        if (windows == 0 && logits) {
            std::ofstream tensor(directory / "input_first.bin", std::ios::binary);
            tensor.write(reinterpret_cast<const char*>(result.inputTensor.data()), result.inputTensor.size() * sizeof(float));
            tensor.close(); if (!tensor) throw std::runtime_error("Cannot write input parity tensor");
            for (int channel = 0; channel < 3; ++channel)
                if (!cv::imwrite((directory / ("source_first_" + std::to_string(channel + 1) + ".png")).u8string(), frames[channel]))
                    throw std::runtime_error("Cannot write lossless source parity frame");
        }
        rows << "{\"source_frame_ids\":[" << decoded - 2 << ',' << decoded - 1 << ',' << decoded
             << "],\"source_pts_s\":" << video.get(cv::CAP_PROP_POS_MSEC) / 1000.
             << ",\"source_size\":[" << frames[2].cols << ',' << frames[2].rows
             << "],\"preprocess_ms\":" << result.preprocessMs << ",\"h2d_ms\":" << result.h2dMs
             << ",\"gpu_ms\":" << result.gpuMs << ",\"d2h_ms\":" << result.d2hMs
             << ",\"decode_ms\":" << result.decodeMs << ",\"wall_ms\":" << result.wallMs << ",\"channels\":[";
        for (int channel = 0; channel < 3; ++channel) {
            if (channel) rows << ',';
            rows << '[';
            for (size_t index = 0; index < result.observations[channel].size(); ++index) {
                if (index) rows << ',';
                observation(rows, result.observations[channel][index]);
            }
            rows << ']';
        }
        rows << "]}\n";
        if (logits) {
            raw.write(reinterpret_cast<const char*>(result.logits.data()), result.logits.size() * sizeof(float));
            inputs.write(reinterpret_cast<const char*>(result.inputTensor.data()), result.inputTensor.size() * sizeof(float));
        }
        if (!rows || (logits && (!raw || !inputs))) throw std::runtime_error("BlurBall evaluation disk write failed");
        if (++windows % 50 == 0) std::cerr << "Evaluated " << windows << " original source windows\n";
    }
    if (windows == 0) throw std::runtime_error("No complete source windows decoded");
    rows.close(); if (logits) {raw.close(); inputs.close();}
    if (!rows || (logits && (!raw || !inputs))) throw std::runtime_error("Cannot finalize evaluation output");
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\"engine_path\":" << quoted(engine.getEnginePath()) << ",\"engine_sha256\":" << quoted(engine.getSha256Hash())
             << ",\"source_path\":" << quoted(std::filesystem::absolute(std::filesystem::u8path(argv[2])).u8string())
             << ",\"source_fps\":" << std::setprecision(10) << fps << ",\"decoded_frames\":" << decoded
             << ",\"evaluated_windows\":" << windows << ",\"heatmap_threshold\":0.7,\"logits_saved\":" << (logits ? "true" : "false")
             << ",\"empty_inputs_rejected\":true,\"invalid_type_rejected\":true,\"mixed_dimensions_rejected\":true"
             << ",\"tensor_contract\":\"[1,9,288,512] -> [1,3,288,512] FP32\""
             << ",\"scope\":\"Sequential source-window numerical parity; offline stage timings, not live latency or accuracy\"}\n";
    manifest.close(); if (!manifest) throw std::runtime_error("Cannot finalize evaluation manifest");
    std::cerr << "Completed " << windows << " native BlurBall windows from " << decoded << " sequential source frames\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "[eval_blurball_native] " << error.what() << '\n'; return 1;
}
