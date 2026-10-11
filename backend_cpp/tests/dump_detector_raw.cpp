#include "tensorrt_engine.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cmath>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#endif

namespace fs = std::filesystem;
static std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
static std::string sha(const fs::path& path) {
#ifdef _WIN32
    HCRYPTPROV provider = 0; HCRYPTHASH hash = 0;
    if (!CryptAcquireContext(&provider, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) throw std::runtime_error("SHA256 provider failed");
    if (!CryptCreateHash(provider, CALG_SHA_256, 0, 0, &hash)) { CryptReleaseContext(provider, 0); throw std::runtime_error("SHA256 init failed"); }
    std::ifstream input(path, std::ios::binary); char block[65536]; bool valid = input.good();
    while (valid && input.read(block, sizeof(block))) valid = CryptHashData(hash, reinterpret_cast<BYTE*>(block), DWORD(input.gcount()), 0);
    if (valid && input.gcount()) valid = CryptHashData(hash, reinterpret_cast<BYTE*>(block), DWORD(input.gcount()), 0);
    valid = valid && input.eof(); BYTE digest[32]; DWORD length = sizeof(digest);
    valid = valid && CryptGetHashParam(hash, HP_HASHVAL, digest, &length, 0);
    CryptDestroyHash(hash); CryptReleaseContext(provider, 0);
    if (!valid || length != 32) throw std::runtime_error("SHA256 file read failed");
    std::ostringstream result; for (BYTE c : digest) result << std::hex << std::setw(2) << std::setfill('0') << int(c);
    return result.str();
#else
    throw std::runtime_error("SHA256 provenance currently requires Windows");
#endif
}
static void binary(const fs::path& path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    if (!file) throw std::runtime_error("Cannot write diagnostic tensor");
}
static int run(const std::vector<std::string>& arguments) try {
    const int argc = int(arguments.size());
    std::string engine, image, output; int classes = 1; bool gpu = false;
    for (int index = 1; index < argc; ++index) {
        const std::string key = arguments[index];
        if (key == "--cuda-preprocess") gpu = true;
        else if (key == "--engine" || key == "--image" || key == "--output" || key == "--classes") {
            if (++index == argc) throw std::invalid_argument("Missing option value");
            if (key == "--engine") engine = arguments[index]; else if (key == "--image") image = arguments[index];
            else if (key == "--output") output = arguments[index]; else classes = std::stoi(arguments[index]);
        } else throw std::invalid_argument("Unknown option: " + key);
    }
    if (engine.empty() || image.empty() || output.empty()) throw std::invalid_argument("Required: --engine FILE --image FILE --output NEW_DIRECTORY [--classes N] [--cuda-preprocess]");
    const fs::path imagePath = fs::u8path(image), directory = fs::u8path(output);
    if (fs::exists(directory)) throw std::invalid_argument("Refusing existing diagnostic output directory");
    const auto sourceHash = sha(imagePath);
    std::ifstream source(imagePath, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(source)), {});
    const cv::Mat frame = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (frame.empty()) throw std::runtime_error("Cannot decode source image");
    TensorRTEngine detector(engine, 640, 640, classes);
    if (!detector.init()) throw std::runtime_error("Cannot initialize detector");
    const auto result = detector.inferRawFrame(frame, gpu, true);
    if (!fs::create_directories(directory)) throw std::runtime_error("Cannot create diagnostic output directory");
    binary(directory / "input.f32", result.inputTensor); binary(directory / "output.f32", result.rawOutput);
    // Preserve exact decoded BGR bytes for an independent color/geometry check.
    std::ofstream pixels(directory / "source.bgr8", std::ios::binary);
    pixels.write(reinterpret_cast<const char*>(frame.data), frame.total() * frame.elemSize()); pixels.close();
    if (!pixels) throw std::runtime_error("Cannot save decoded source bytes");
    fs::path binaryPath;
#ifdef _WIN32
    std::vector<wchar_t> name(32768); const DWORD count = GetModuleFileNameW(nullptr, name.data(), DWORD(name.size()));
    if (!count || count >= name.size()) throw std::runtime_error("Cannot identify native executable");
    binaryPath = fs::path(std::wstring(name.data(), count));
#else
    binaryPath = fs::canonical(arguments[0]);
#endif
    const auto& shape = result.shape;
    const float scale = std::min(float(shape.inputWidth) / frame.cols, float(shape.inputHeight) / frame.rows);
    const int width = int(std::round(frame.cols * scale)), height = int(std::round(frame.rows * scale));
    const int left = (shape.inputWidth - width) / 2, top = (shape.inputHeight - height) / 2;
    std::ofstream report(directory / "contract.json"); report << std::setprecision(9)
        << "{\n\"schema_version\":1,\n\"engine_path\":" << quote(fs::absolute(fs::u8path(detector.getEnginePath())).u8string())
        << ",\n\"engine_sha256\":" << quote(detector.getSha256Hash()) << ",\n\"native_binary_path\":" << quote(binaryPath.u8string())
        << ",\n\"native_binary_sha256\":" << quote(sha(binaryPath)) << ",\n\"source_image_path\":" << quote(fs::absolute(imagePath).u8string())
        << ",\n\"source_image_sha256\":" << quote(sourceHash) << ",\n\"decoded_bgr_sha256\":" << quote(sha(directory / "source.bgr8"))
        << ",\n\"source_size\": [" << frame.cols << ',' << frame.rows << "],\n\"input_shape\": [1,3," << shape.inputHeight << ',' << shape.inputWidth
        << "],\n\"output_shape\": [1," << shape.classes + 4 << ',' << shape.anchors << "],\n\"dtype\":\"float32\",\n\"byte_order\":\"little\","
        << "\n\"preprocessing_revision\":" << quote(result.preprocessingRevision) << ",\n\"input_file\":\"input.f32\",\n\"output_file\":\"output.f32\","
        << "\n\"input_sha256\":" << quote(sha(directory / "input.f32")) << ",\n\"output_sha256\":" << quote(sha(directory / "output.f32"))
        << ",\n\"decoded_bgr_file\":\"source.bgr8\",\n\"letterbox\": {\"scale\":" << scale << ",\"unpadded_width\":" << width
        << ",\"unpadded_height\":" << height << ",\"pad_left\":" << left << ",\"pad_top\":" << top
        << ",\"pad_right\":" << shape.inputWidth - width - left << ",\"pad_bottom\":" << shape.inputHeight - height - top << ",\"pad_value\":114},"
        << "\n\"timings_ms\": {\"h2d\":" << result.timing.h2dMs << ",\"preprocess\":" << result.timing.preprocessMs
        << ",\"gpu_compute\":" << result.timing.gpuComputeMs << ",\"d2h\":" << result.timing.d2hMs << "},\n\"gpu_input_dump_in_d2h_timing\":"
        << (gpu ? "true" : "false") << ",\n\"scope\":\"Offline raw-tensor diagnostic; no accuracy or live-latency claim\"\n}\n";
    if (!report || sha(imagePath) != sourceHash) throw std::runtime_error("Report write or source immutability check failed");
    std::cout << "Saved actual pre-NMS tensor diagnostic to " << directory.u8string() << '\n'; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> arguments;
    for (int index = 0; index < argc; ++index) {
        const int length = int(wcslen(argv[index]));
        const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], length, nullptr, 0, nullptr, nullptr);
        if (length && bytes <= 0) { std::cerr << "Invalid Unicode CLI argument\n"; return 1; }
        std::string utf8(bytes, '\0');
        if (bytes && !WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], length, utf8.data(), bytes, nullptr, nullptr)) return 1;
        arguments.push_back(std::move(utf8));
    }
    return run(arguments);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
