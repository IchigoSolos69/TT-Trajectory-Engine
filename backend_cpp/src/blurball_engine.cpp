#include "blurball_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#endif
#ifdef HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
namespace {
using Clock = std::chrono::steady_clock;
float elapsed(Clock::time_point start) {
    return std::chrono::duration<float, std::milli>(Clock::now() - start).count();
}
void checked(cudaError_t code, const char* stage) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("BlurBall ") + stage + ": " + cudaGetErrorString(code));
}
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* text) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[BlurBall TensorRT] " << text << '\n';
    }
};
Logger& processLogger() {
    // TensorRT reuses the first registered logger across other runtimes. Its
    // lifetime must therefore exceed every engine, including the scene model.
    static Logger logger;
    return logger;
}
std::string fileSha(const std::filesystem::path& path) {
#ifdef _WIN32
    HCRYPTPROV provider = 0;
    HCRYPTHASH hash = 0;
    if (!CryptAcquireContext(&provider, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT))
        throw std::runtime_error("BlurBall SHA-256 provider unavailable");
    if (!CryptCreateHash(provider, CALG_SHA_256, 0, 0, &hash)) {
        CryptReleaseContext(provider, 0);
        throw std::runtime_error("BlurBall SHA-256 creation failed");
    }
    std::ifstream input(path, std::ios::binary);
    bool valid = input.good();
    char block[65536];
    while (valid && input.read(block, sizeof(block)))
        valid = CryptHashData(hash, reinterpret_cast<const BYTE*>(block), static_cast<DWORD>(input.gcount()), 0) != 0;
    if (valid && input.gcount() > 0)
        valid = CryptHashData(hash, reinterpret_cast<const BYTE*>(block), static_cast<DWORD>(input.gcount()), 0) != 0;
    BYTE digest[32]; DWORD size = sizeof(digest);
    valid = valid && input.eof() && CryptGetHashParam(hash, HP_HASHVAL, digest, &size, 0) && size == 32;
    CryptDestroyHash(hash); CryptReleaseContext(provider, 0);
    if (!valid) throw std::runtime_error("BlurBall model SHA-256 failed");
    std::ostringstream out;
    for (BYTE value : digest) out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
    return out.str();
#else
    return "SHA256_UNAVAILABLE_ON_THIS_PLATFORM";
#endif
}
// Matches upstream get_affine_transform(center, max(width,height), 0,[512,288]).
// The maximum source dimension sets the destination width; portrait/non-16:9
// footage therefore follows the model's trained affine crop, not YOLO letterbox.
cv::Mat affine(cv::Size size, bool inverse) {
    const cv::Point2f center(size.width * .5f, size.height * .5f);
    const float scale = static_cast<float>(std::max(size.width, size.height));
    const std::array<cv::Point2f, 3> source = {center,
        center + cv::Point2f(0, -.5f * scale), center + cv::Point2f(-.5f * scale, -.5f * scale)};
    const cv::Point2f target(BlurBallEngine::inputWidth * .5f, BlurBallEngine::inputHeight * .5f);
    const std::array<cv::Point2f, 3> destination = {target,
        target + cv::Point2f(0, -.5f * BlurBallEngine::inputWidth),
        target + cv::Point2f(-.5f * BlurBallEngine::inputWidth, -.5f * BlurBallEngine::inputWidth)};
    return inverse ? cv::getAffineTransform(destination.data(), source.data())
                   : cv::getAffineTransform(source.data(), destination.data());
}
cv::Point2d transform(const cv::Mat& matrix, cv::Point2d point) {
    return {matrix.at<double>(0, 0) * point.x + matrix.at<double>(0, 1) * point.y + matrix.at<double>(0, 2),
            matrix.at<double>(1, 0) * point.x + matrix.at<double>(1, 1) * point.y + matrix.at<double>(1, 2)};
}
std::vector<BlurBallObservation> decode(const float* logits, const cv::Mat& inverse, cv::Size sourceSize) {
    cv::Mat heatmap(BlurBallEngine::inputHeight, BlurBallEngine::inputWidth, CV_32F);
    cv::Mat mask(heatmap.size(), CV_8U);
    for (int y = 0; y < heatmap.rows; ++y) {
        float* probabilities = heatmap.ptr<float>(y); auto* binary = mask.ptr<unsigned char>(y);
        for (int x = 0; x < heatmap.cols; ++x) {
            const float logit = logits[y * heatmap.cols + x];
            if (!std::isfinite(logit)) throw std::runtime_error("Nonfinite BlurBall output logit");
            // Stable sigmoid preserves very negative logits without exp overflow.
            const float exponential = std::exp(-std::abs(logit));
            probabilities[x] = logit >= 0 ? 1.f / (1.f + exponential) : exponential / (1.f + exponential);
            binary[x] = probabilities[x] > BlurBallEngine::heatmapThreshold ? 1 : 0;
        }
    }
    cv::Mat labels;
    const int count = cv::connectedComponents(mask, labels, 8, CV_32S);
    struct Component {
        double mass = 0, weightedX = 0, weightedY = 0;
        float peak = 0;
        std::vector<cv::Point2f> points;
    };
    std::vector<Component> components(count);
    for (int y = 0; y < heatmap.rows; ++y) {
        const int* row = labels.ptr<int>(y); const float* values = heatmap.ptr<float>(y);
        for (int x = 0; x < heatmap.cols; ++x) if (row[x]) {
            auto& component = components[row[x]];
            component.mass += values[x]; component.weightedX += x * static_cast<double>(values[x]);
            component.weightedY += y * static_cast<double>(values[x]);
            component.peak = std::max(component.peak, values[x]);
            component.points.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    std::vector<BlurBallObservation> observations;
    for (int label = 1; label < count; ++label) {
        const auto& component = components[label];
        if (!(component.mass > 0) || component.points.empty()) continue;
        const cv::Point2d center(component.weightedX / component.mass, component.weightedY / component.mass);
        const cv::Point2d sourceCenter = transform(inverse, center);
        if (sourceCenter.x < 0 || sourceCenter.y < 0 || sourceCenter.x >= sourceSize.width || sourceCenter.y >= sourceSize.height)
            continue; // Padding/cropped-source hypotheses never become observations.
        cv::Mat coordinates(static_cast<int>(component.points.size()), 2, CV_32F);
        float minX = component.points.front().x, maxX = minX, minY = component.points.front().y, maxY = minY;
        for (int index = 0; index < coordinates.rows; ++index) {
            const auto point = component.points[index];
            coordinates.at<float>(index, 0) = point.x; coordinates.at<float>(index, 1) = point.y;
            minX = std::min(minX, point.x); maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y); maxY = std::max(maxY, point.y);
        }
        cv::Point2d axis(1, 0), mean(center);
        double halfLength = 0;
        if (coordinates.rows > 1) {
            cv::PCA pca(coordinates, cv::Mat(), cv::PCA::DATA_AS_ROW);
            mean = {pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1)};
            if (!pca.eigenvectors.empty()) axis = {pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1)};
            double low = 0, high = 0;
            for (int index = 0; index < coordinates.rows; ++index) {
                const auto point = component.points[index];
                const double projection = (point.x - mean.x) * axis.x + (point.y - mean.y) * axis.y;
                if (index == 0) low = high = projection;
                else { low = std::min(low, projection); high = std::max(high, projection); }
            }
            halfLength = (high - low) * .5;
        }
        const cv::Point2d first = transform(inverse, center - halfLength * axis);
        const cv::Point2d second = transform(inverse, center + halfLength * axis);
        const cv::Point2d boxLow = transform(inverse, {minX - .5, minY - .5});
        const cv::Point2d boxHigh = transform(inverse, {maxX + .5, maxY + .5});
        const double x0 = std::clamp(boxLow.x, 0., static_cast<double>(sourceSize.width));
        const double y0 = std::clamp(boxLow.y, 0., static_cast<double>(sourceSize.height));
        const double x1 = std::clamp(boxHigh.x, 0., static_cast<double>(sourceSize.width));
        const double y1 = std::clamp(boxHigh.y, 0., static_cast<double>(sourceSize.height));
        BlurBallObservation observation;
        observation.pixelCenter = cv::Point2f(sourceCenter);
        observation.componentMass = static_cast<float>(component.mass); observation.heatmapPeak = component.peak;
        observation.blurEndpoints = {cv::Point2f(first), cv::Point2f(second)};
        observation.blurExtentSourcePx = static_cast<float>(cv::norm(second - first));
        double angle = std::atan2(second.y - first.y, second.x - first.x) * 180. / CV_PI;
        angle = std::fmod(angle + 180., 180.);
        observation.blurAxisDeg = static_cast<float>(angle);
        observation.detection = {0, component.peak, static_cast<float>(x0 / sourceSize.width),
            static_cast<float>(y0 / sourceSize.height), static_cast<float>((x1 - x0) / sourceSize.width),
            static_cast<float>((y1 - y0) / sourceSize.height)};
        observations.push_back(observation);
    }
    return observations;
}
}
struct BlurBallEngine::Impl {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string path, sha;
    std::mutex mutex;
    cudaStream_t stream = nullptr;
    std::array<cudaEvent_t, 4> events{};
    float* hostInput = nullptr; float* hostOutput = nullptr;
    void* deviceInput = nullptr; void* deviceOutput = nullptr;
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        context.reset();
        if (deviceInput) cudaFree(deviceInput);
        if (deviceOutput) cudaFree(deviceOutput);
        if (hostInput) cudaFreeHost(hostInput);
        if (hostOutput) cudaFreeHost(hostOutput);
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
    }
};
BlurBallEngine::BlurBallEngine(const std::string& enginePath) : impl_(std::make_unique<Impl>()) {
    if (enginePath.empty()) throw std::runtime_error("An explicit BlurBall engine path is required");
    const auto canonical = std::filesystem::canonical(std::filesystem::u8path(enginePath));
    auto& i = *impl_; i.path = canonical.u8string(); i.sha = fileSha(canonical);
    std::ifstream file(canonical, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open BlurBall engine: " + i.path);
    std::vector<char> serialized(static_cast<size_t>(file.tellg())); file.seekg(0);
    if (!file.read(serialized.data(), static_cast<std::streamsize>(serialized.size())))
        throw std::runtime_error("Cannot read complete BlurBall engine: " + i.path);
    i.runtime.reset(nvinfer1::createInferRuntime(processLogger()));
    if (!i.runtime) throw std::runtime_error("BlurBall TensorRT runtime creation failed");
    i.engine.reset(i.runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!i.engine || i.engine->getNbIOTensors() != 2)
        throw std::runtime_error("BlurBall engine must expose exactly two tensors");
    const char* inputName = nullptr; const char* outputName = nullptr;
    for (int index = 0; index < 2; ++index) {
        const char* name = i.engine->getIOTensorName(index);
        if (!name || i.engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
            i.engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
            throw std::runtime_error("BlurBall requires named FP32 device I/O");
        const auto shape = i.engine->getTensorShape(name);
        const bool input = i.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        if (shape.nbDims != 4 || shape.d[0] != 1 || shape.d[1] != (input ? 9 : 3) ||
            shape.d[2] != inputHeight || shape.d[3] != inputWidth)
            throw std::runtime_error("BlurBall contract is [1,9,288,512] -> [1,3,288,512]");
        if (i.engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
            throw std::runtime_error("BlurBall requires linear NCHW tensor memory");
        if (input) { if (inputName) throw std::runtime_error("Duplicate BlurBall input"); inputName = name; }
        else { if (outputName) throw std::runtime_error("Duplicate BlurBall output"); outputName = name; }
    }
    if (!inputName || !outputName) throw std::runtime_error("Missing BlurBall input/output");
    i.context.reset(i.engine->createExecutionContext());
    if (!i.context) throw std::runtime_error("BlurBall execution context creation failed");
    checked(cudaStreamCreateWithFlags(&i.stream, cudaStreamNonBlocking), "stream creation");
    for (auto& event : i.events) checked(cudaEventCreate(&event), "timing event creation");
    checked(cudaMallocHost(reinterpret_cast<void**>(&i.hostInput), inputElements * sizeof(float)), "pinned input allocation");
    checked(cudaMallocHost(reinterpret_cast<void**>(&i.hostOutput), outputElements * sizeof(float)), "pinned output allocation");
    checked(cudaMalloc(&i.deviceInput, inputElements * sizeof(float)), "device input allocation");
    checked(cudaMalloc(&i.deviceOutput, outputElements * sizeof(float)), "device output allocation");
    if (!i.context->setTensorAddress(inputName, i.deviceInput) || !i.context->setTensorAddress(outputName, i.deviceOutput))
        throw std::runtime_error("BlurBall tensor binding failed");
    std::cerr << "[BlurBall] " << i.path << " SHA-256 " << i.sha << " | FP32 [1,9,288,512] -> [1,3,288,512]\n";
}
BlurBallEngine::~BlurBallEngine() = default;
const std::string& BlurBallEngine::getEnginePath() const { return impl_->path; }
const std::string& BlurBallEngine::getSha256Hash() const { return impl_->sha; }
BlurBallEngine::Result BlurBallEngine::infer(const std::array<cv::Mat, 3>& frames, bool includeLogits) {
    auto& i = *impl_; std::lock_guard<std::mutex> lock(i.mutex);
    const auto started = Clock::now(); Result result;
    const auto size = frames[0].size();
    for (const auto& frame : frames)
        if (frame.empty() || frame.type() != CV_8UC3 || frame.size() != size)
            throw std::runtime_error("BlurBall needs three same-size nonempty CV_8UC3 BGR frames");
    const auto forward = affine(size, false), inverse = affine(size, true);
    constexpr float means[3] = {.485f, .456f, .406f}, stds[3] = {.229f, .224f, .225f};
    constexpr int plane = inputWidth * inputHeight;
    for (int frameIndex = 0; frameIndex < 3; ++frameIndex) {
        cv::Mat rgb, warped;
        cv::cvtColor(frames[frameIndex], rgb, cv::COLOR_BGR2RGB);
        cv::warpAffine(rgb, warped, forward, {inputWidth, inputHeight}, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        for (int y = 0; y < inputHeight; ++y) {
            const auto* row = warped.ptr<cv::Vec3b>(y);
            for (int x = 0; x < inputWidth; ++x) for (int channel = 0; channel < 3; ++channel)
                i.hostInput[(frameIndex * 3 + channel) * plane + y * inputWidth + x] =
                    (static_cast<float>(row[x][channel]) / 255.f - means[channel]) / stds[channel];
        }
    }
    result.preprocessMs = elapsed(started);
    checked(cudaEventRecord(i.events[0], i.stream), "upload timing start");
    checked(cudaMemcpyAsync(i.deviceInput, i.hostInput, inputElements * sizeof(float), cudaMemcpyHostToDevice, i.stream), "input upload");
    checked(cudaEventRecord(i.events[1], i.stream), "upload timing stop");
    if (!i.context->enqueueV3(i.stream)) throw std::runtime_error("BlurBall enqueueV3 failed");
    checked(cudaEventRecord(i.events[2], i.stream), "inference timing stop");
    checked(cudaMemcpyAsync(i.hostOutput, i.deviceOutput, outputElements * sizeof(float), cudaMemcpyDeviceToHost, i.stream), "output download");
    checked(cudaEventRecord(i.events[3], i.stream), "download timing stop");
    checked(cudaStreamSynchronize(i.stream), "inference synchronization");
    checked(cudaEventElapsedTime(&result.h2dMs, i.events[0], i.events[1]), "upload elapsed time");
    checked(cudaEventElapsedTime(&result.gpuMs, i.events[1], i.events[2]), "inference elapsed time");
    checked(cudaEventElapsedTime(&result.d2hMs, i.events[2], i.events[3]), "download elapsed time");
    const auto decoding = Clock::now();
    for (int channel = 0; channel < 3; ++channel)
        result.observations[channel] = decode(i.hostOutput + channel * plane, inverse, size);
    if (includeLogits) {
        result.logits.assign(i.hostOutput, i.hostOutput + outputElements);
        result.inputTensor.assign(i.hostInput, i.hostInput + inputElements);
    }
    result.decodeMs = elapsed(decoding); result.wallMs = elapsed(started);
    return result;
}
#else
struct BlurBallEngine::Impl { std::string path, sha; };
BlurBallEngine::BlurBallEngine(const std::string&) { throw std::runtime_error("BlurBall requires TensorRT"); }
BlurBallEngine::~BlurBallEngine() = default;
const std::string& BlurBallEngine::getEnginePath() const { return impl_->path; }
const std::string& BlurBallEngine::getSha256Hash() const { return impl_->sha; }
BlurBallEngine::Result BlurBallEngine::infer(const std::array<cv::Mat, 3>&, bool) { throw std::runtime_error("BlurBall requires TensorRT"); }
#endif
