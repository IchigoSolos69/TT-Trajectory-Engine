#include "tensorrt_engine.hpp"
#include "cuda_preprocess.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#pragma comment(lib, "advapi32.lib")
#endif

#ifdef HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT C++ Engine] " << msg << std::endl;
        }
    }
} gLogger;

static void checkCuda(cudaError_t status, const char* stage) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + stage + ": " + cudaGetErrorString(status));
}
#endif

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdlib>

static std::string computeSha256(const std::string& filePath) {
#ifdef _WIN32
    HCRYPTPROV provider = 0;
    HCRYPTHASH hash = 0;
    if (!CryptAcquireContext(&provider, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT))
        throw std::runtime_error("Cannot initialize Windows SHA-256 provider.");
    if (!CryptCreateHash(provider, CALG_SHA_256, 0, 0, &hash)) {
        CryptReleaseContext(provider, 0);
        throw std::runtime_error("Cannot initialize SHA-256 hash.");
    }
    std::ifstream file(std::filesystem::u8path(filePath), std::ios::binary);
    bool valid = file.good();
    char buffer[65536];
    while (valid && file.read(buffer, sizeof(buffer))) {
        valid = CryptHashData(hash, reinterpret_cast<const BYTE*>(buffer), static_cast<DWORD>(file.gcount()), 0);
    }
    if (valid && file.gcount() > 0)
        valid = CryptHashData(hash, reinterpret_cast<const BYTE*>(buffer), static_cast<DWORD>(file.gcount()), 0);
    valid = valid && file.eof();
    BYTE digest[32];
    DWORD length = sizeof(digest);
    valid = valid && CryptGetHashParam(hash, HP_HASHVAL, digest, &length, 0);
    CryptDestroyHash(hash);
    CryptReleaseContext(provider, 0);
    if (!valid) throw std::runtime_error("Cannot hash model file: " + filePath);
    std::ostringstream result;
    for (BYTE byte : digest) result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    return result.str();
#else
    return "SHA256_UNAVAILABLE_ON_THIS_PLATFORM";
#endif
}
static std::string getFileLastModifiedIso(const std::string& filePath) {
    try {
        namespace fs = std::filesystem;
        if (!fs::exists(fs::u8path(filePath))) return "N/A";
        auto ftime = fs::last_write_time(fs::u8path(filePath));
        auto s_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
        );
        std::time_t tt = std::chrono::system_clock::to_time_t(s_time);
        std::tm* gmt = std::gmtime(&tt);
        char buf[64];
        if (gmt && std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmt)) {
            return std::string(buf);
        }
    } catch (...) {}
    return "N/A";
}

std::string TensorRTEngine::resolveEnginePath(const std::string& userPath) {
    if (!userPath.empty() && std::ifstream(std::filesystem::u8path(userPath)).good()) {
        return userPath;
    }
    if (!userPath.empty()) {
        for (const auto& prefix : {"../", "../../", "../../../"}) {
            const auto path = std::string(prefix) + userPath;
            if (std::ifstream(std::filesystem::u8path(path)).good()) return path;
        }
        throw std::runtime_error("Requested engine was not found: " + userPath);
    }
    const char* envPath = std::getenv("TT_ENGINE_PATH");
    if (envPath && std::ifstream(std::filesystem::u8path(envPath)).good()) {
        return std::string(envPath);
    }

    std::vector<std::string> candidateSearchList = {
        "models/yolov11_ttgames.engine",
        "../models/yolov11_ttgames.engine",
        "../../models/yolov11_ttgames.engine",
        "../../../models/yolov11_ttgames.engine"
    };

    if (!userPath.empty()) {
        candidateSearchList.insert(candidateSearchList.begin(), userPath);
        candidateSearchList.insert(candidateSearchList.begin() + 1, "../" + userPath);
        candidateSearchList.insert(candidateSearchList.begin() + 2, "../../" + userPath);
    }

    std::vector<std::string> existingPaths;
    std::vector<std::string> canonicalPaths;
    namespace fs = std::filesystem;

    for (const auto& path : candidateSearchList) {
        if (std::ifstream(fs::u8path(path)).good()) {
            std::string cPath = path;
            try {
                cPath = fs::canonical(fs::u8path(path)).u8string();
            } catch (...) {}

            // Check if canonical path is unique
            if (std::find(canonicalPaths.begin(), canonicalPaths.end(), cPath) == canonicalPaths.end()) {
                existingPaths.push_back(path);
                canonicalPaths.push_back(cPath);
            }
        }
    }

    if (canonicalPaths.size() > 1) {
        std::cerr << "========================================================================\n";
        std::cerr << "[TensorRTEngine] CRITICAL FATAL ERROR: Multiple distinct engine files detected on search paths!\n";
        std::cerr << "[TensorRTEngine] Ambiguity hazard! The engine resolver found " << canonicalPaths.size() << " conflicting engine files:\n";
        for (size_t i = 0; i < existingPaths.size(); ++i) {
            uint64_t fSize = 0;
            try { fSize = fs::file_size(fs::u8path(existingPaths[i])); } catch(...) {}
            std::string sha = computeSha256(existingPaths[i]);
            std::cerr << "  [" << (i + 1) << "] Relative: " << existingPaths[i]
                      << " | Canonical: " << canonicalPaths[i]
                      << " | Size: " << fSize << " bytes"
                      << " | SHA-256: " << sha << "\n";
        }
        std::cerr << "[TensorRTEngine] Please remove duplicate engine files to guarantee determinism.\n";
        std::cerr << "========================================================================\n";
        std::exit(1);
    }

    if (!existingPaths.empty()) {
        return existingPaths[0];
    }

    return "models/yolov11_ttgames.engine";
}

TensorRTEngine::TensorRTEngine(const std::string& enginePath, int inputWidth, int inputHeight, int classes)
    : enginePath_(resolveEnginePath(enginePath)),
      classes_(classes),
      inputW_(inputWidth),
      inputH_(inputHeight),
      isInitialized_(false),
      context_(nullptr),
      engine_(nullptr),
      stream_(nullptr),
      hostInputBuffer_(nullptr),
      hostOutputBuffer_(nullptr),
      deviceInputBuffer_(nullptr),
      deviceOutputBuffer_(nullptr) {
    if (inputWidth <= 0 || inputHeight <= 0 || classes < 1 || classes > 80)
        throw std::invalid_argument("Positive input hints and 1..80 configured YOLO classes required.");
}

void DetectorTensorShape::validate() const {
    if (inputWidth <= 0 || inputHeight <= 0 || inputWidth % 32 || inputHeight % 32 ||
        classes < 1 || classes > 80 || anchors <= 0 ||
        static_cast<size_t>(inputWidth) * inputHeight > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        static_cast<size_t>(anchors) > std::numeric_limits<size_t>::max() / (4 + classes) / sizeof(float))
        throw std::invalid_argument("Malformed static FP32 YOLO tensor contract.");
}
size_t DetectorTensorShape::inputElements() const { validate(); return static_cast<size_t>(3) * inputWidth * inputHeight; }
size_t DetectorTensorShape::outputElements() const { validate(); return static_cast<size_t>(4 + classes) * anchors; }
DetectorTensorShape TensorRTEngine::getTensorShape() const {
    if (!isInitialized_) throw std::runtime_error("Detector tensor contract requires initialized engine.");
    return {inputW_, inputH_, classes_, anchorCount_};
}

TensorRTEngine::~TensorRTEngine() {
#ifdef HAS_TENSORRT
    if (evStartH2D_) cudaEventDestroy(static_cast<cudaEvent_t>(evStartH2D_));
    if (evStopH2D_) cudaEventDestroy(static_cast<cudaEvent_t>(evStopH2D_));
    if (evStartPrep_) cudaEventDestroy(static_cast<cudaEvent_t>(evStartPrep_));
    if (evStopPrep_) cudaEventDestroy(static_cast<cudaEvent_t>(evStopPrep_));
    if (evStartExec_) cudaEventDestroy(static_cast<cudaEvent_t>(evStartExec_));
    if (evStopExec_) cudaEventDestroy(static_cast<cudaEvent_t>(evStopExec_));
    if (evStartD2H_) cudaEventDestroy(static_cast<cudaEvent_t>(evStartD2H_));
    if (evStopD2H_) cudaEventDestroy(static_cast<cudaEvent_t>(evStopD2H_));
    if (context_) delete static_cast<nvinfer1::IExecutionContext*>(context_);
    if (engine_) delete static_cast<nvinfer1::ICudaEngine*>(engine_);
    if (deviceInputBuffer_) cudaFree(deviceInputBuffer_);
    if (deviceOutputBuffer_) cudaFree(deviceOutputBuffer_);
    if (deviceBGRBuffer_) cudaFree(deviceBGRBuffer_);
    if (hostInputBuffer_) cudaFreeHost(hostInputBuffer_);
    if (hostOutputBuffer_) cudaFreeHost(hostOutputBuffer_);
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
#else
    if (hostInputBuffer_) free(hostInputBuffer_);
    if (hostOutputBuffer_) free(hostOutputBuffer_);
#endif
}

static std::string fetchGitCommitSha() {
#ifdef _WIN32
    FILE* pipe = _popen("git rev-parse HEAD", "r");
#else
    FILE* pipe = popen("git rev-parse HEAD", "r");
#endif
    if (!pipe) return "UNKNOWN_COMMIT";
    char buffer[128];
    std::string result = "";
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result.empty() ? "UNKNOWN_COMMIT" : result;
}

#ifndef COMPILE_GIT_COMMIT_SHA
#define COMPILE_GIT_COMMIT_SHA "NOT_EMBEDDED_AT_BUILD"
#endif

bool TensorRTEngine::init() {
    if (isInitialized_) return true;
    std::cout << "[TensorRTEngine] Loading engine file: " << enginePath_ << std::endl;

    std::ifstream file(std::filesystem::u8path(enginePath_), std::ios::binary);
    if (!file.good()) {
        std::cerr << "[TensorRTEngine] CRITICAL FATAL ERROR: Cannot open engine file at '"
                  << enginePath_ << "'. Exiting program with code 1." << std::endl;
        std::exit(1);
    }

    file.seekg(0, file.end);
    fileSizeBytes_ = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fileSizeBytes_);
    file.read(engineData.data(), fileSizeBytes_);
    file.close();

    lastModifiedIso_ = getFileLastModifiedIso(enginePath_);
    sha256Hash_ = computeSha256(enginePath_);
    gitCommitSha_ = fetchGitCommitSha();

    std::cout << "[TensorRTEngine] Metadata | Size: " << fileSizeBytes_ << " bytes | Last Modified: " << lastModifiedIso_ << std::endl;
    std::cout << "[TensorRTEngine] SHA-256 Checksum  : " << sha256Hash_ << std::endl;
    std::cout << "[TensorRTEngine] Runtime Git SHA   : " << gitCommitSha_ << std::endl;
    std::cout << "[TensorRTEngine] Compile-Time SHA  : " << COMPILE_GIT_COMMIT_SHA << std::endl;

#ifdef HAS_TENSORRT
    int trtVersion = getInferLibVersion();
    std::cout << "[TensorRTEngine] TensorRT Library Version : " << (trtVersion / 10000) << "." << ((trtVersion / 100) % 100) << "." << (trtVersion % 100) << " (raw: " << trtVersion << ")" << std::endl;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime.");
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fileSizeBytes_);
    if (!engine) {
        std::cerr << "[TensorRTEngine] CRITICAL FATAL ERROR: Failed to deserialize CUDA engine. Exiting code 1." << std::endl;
        std::exit(1);
    }

    engine_ = engine;
    if (engine->getNbIOTensors() != 2) throw std::runtime_error("Expected exactly one model input and output.");
    bool foundInput = false, foundOutput = false;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        bool input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        bool shapeOk = input ? (!foundInput && dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3 && dims.d[2] > 0 && dims.d[3] > 0 &&
                                dims.d[2] <= std::numeric_limits<int>::max() && dims.d[3] <= std::numeric_limits<int>::max())
                             : (!foundOutput && dims.nbDims == 3 && dims.d[0] == 1 && dims.d[1] == 4 + classes_ && dims.d[2] > 0 &&
                                dims.d[2] <= std::numeric_limits<int>::max());
        if (!shapeOk || engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
            engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
            throw std::runtime_error("Unexpected detector I/O shape or precision for configured class count.");
        if (input) { inputH_ = dims.d[2]; inputW_ = dims.d[3]; foundInput = true; }
        else { anchorCount_ = dims.d[2]; foundOutput = true; }
    }
    if (!foundInput || !foundOutput) throw std::runtime_error("Expected one detector input and one output.");
    const DetectorTensorShape contract{inputW_, inputH_, classes_, anchorCount_};
    inputSize_ = contract.inputElements() * sizeof(float);
    outputSize_ = contract.outputElements() * sizeof(float);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    context_ = context;
    if (!context) throw std::runtime_error("Cannot create TensorRT execution context.");

    // Allocate Pinned Host Memory for zero-copy DMA execution
    checkCuda(cudaMallocHost(&hostInputBuffer_, inputSize_), "allocate pinned input");
    checkCuda(cudaMallocHost(&hostOutputBuffer_, outputSize_), "allocate pinned output");

    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "create stream");
    stream_ = stream;

    // Log data type for all I/O tensors via TensorRT 10 API
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tName = engine->getIOTensorName(i);
        nvinfer1::DataType dt = engine->getTensorDataType(tName);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(tName);
        std::string dtStr = (dt == nvinfer1::DataType::kFLOAT) ? "Float32 (kFLOAT)" :
                            (dt == nvinfer1::DataType::kHALF) ? "Float16 (kHALF)" : "Other";
        std::string ioStr = (ioMode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        std::cout << "[TensorRTEngine] getTensorDataType | Tensor: '" << tName << "' [" << ioStr << "] -> " << dtStr << std::endl;
    }

    // Allocate GPU memory
    checkCuda(cudaMalloc(&deviceInputBuffer_, inputSize_), "allocate input");
    checkCuda(cudaMalloc(&deviceOutputBuffer_, outputSize_), "allocate output");

    // Set static tensor addresses once during init
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tName = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(tName) == nvinfer1::TensorIOMode::kINPUT) {
            if (!context->setTensorAddress(tName, deviceInputBuffer_)) throw std::runtime_error("Cannot bind detector input.");
        } else {
            if (!context->setTensorAddress(tName, deviceOutputBuffer_)) throw std::runtime_error("Cannot bind detector output.");
        }
    }

    for (void** event : {&evStartH2D_, &evStopH2D_, &evStartPrep_, &evStopPrep_,
                         &evStartExec_, &evStopExec_, &evStartD2H_, &evStopD2H_})
        checkCuda(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(event)), "create timing event");

    std::cout << "[TensorRTEngine] CUDA buffers allocated successfully on Device GPU (Pinned Host Memory Enabled)." << std::endl;
    std::cout << "[TensorRTEngine] Input Tensor Shape  : [1, 3, " << inputH_ << ", " << inputW_ << "] (Float32 CHW)" << std::endl;
    std::cout << "[TensorRTEngine] Output Tensor Shape : [1, " << 4 + classes_ << ", " << anchorCount_ << "] (Float32)" << std::endl;
    std::cout << "[TensorRTEngine] Validated static FP32 I/O; internal precision belongs to engine export metadata." << std::endl;
#else
    std::cerr << "[TensorRTEngine] TensorRT support is required for analysis." << std::endl;
    return false;
#endif

    isInitialized_ = true;
    return true;
}

std::vector<float> TensorRTEngine::preprocessBGRtoCHW(const unsigned char* bgrData, int width, int height) {
    std::vector<float> chwBuffer(3 * width * height);
    int channelSize = width * height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcIdx = (y * width + x) * 3;
            unsigned char b = bgrData[srcIdx + 0];
            unsigned char g = bgrData[srcIdx + 1];
            unsigned char r = bgrData[srcIdx + 2];

            int dstPixelIdx = y * width + x;
            // Normalize 0..255 to 0.0..1.0 and convert BGR -> RGB planar
            chwBuffer[0 * channelSize + dstPixelIdx] = r / 255.0f;
            chwBuffer[1 * channelSize + dstPixelIdx] = g / 255.0f;
            chwBuffer[2 * channelSize + dstPixelIdx] = b / 255.0f;
        }
    }

    return chwBuffer;
}

std::vector<float> TensorRTEngine::preprocessOpenCVFrame(const cv::Mat& inputFrame, int inputWidth, int inputHeight) {
    if (inputFrame.empty() || inputFrame.type() != CV_8UC3 || inputWidth <= 0 || inputHeight <= 0)
        throw std::invalid_argument("Preprocessing requires a nonempty BGR8 frame and positive input dimensions.");
    // 1. Calculate letterbox scale ratio and padding for aspect ratio preservation (e.g. 1920x1080 -> 640x640)
    float scaleR = std::min(static_cast<float>(inputWidth) / inputFrame.cols, static_cast<float>(inputHeight) / inputFrame.rows);
    int unpadW = static_cast<int>(std::round(inputFrame.cols * scaleR)); // 640
    int unpadH = static_cast<int>(std::round(inputFrame.rows * scaleR)); // 360
    int padW = (inputWidth - unpadW) / 2;  // 0
    int padH = (inputHeight - unpadH) / 2; // 140

    cv::Mat resizedImage, paddedImage;
    cv::resize(inputFrame, resizedImage, cv::Size(unpadW, unpadH), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resizedImage, paddedImage, padH, inputHeight - unpadH - padH,
                      padW, inputWidth - unpadW - padW, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 2. Convert BGR (OpenCV default) to RGB (YOLO default)
    cv::Mat rgbImage, floatImage;
    cv::cvtColor(paddedImage, rgbImage, cv::COLOR_BGR2RGB);
    rgbImage.convertTo(floatImage, CV_32FC3, 1.0f / 255.0f);

    // 3. Flat CHW data vector for GPU
    std::vector<float> inputData(3 * inputHeight * inputWidth);
    std::vector<cv::Mat> chw(3);
    chw[0] = cv::Mat(inputHeight, inputWidth, CV_32FC1, inputData.data());
    chw[1] = cv::Mat(inputHeight, inputWidth, CV_32FC1, inputData.data() + inputHeight * inputWidth);
    chw[2] = cv::Mat(inputHeight, inputWidth, CV_32FC1, inputData.data() + 2 * inputHeight * inputWidth);
    cv::split(floatImage, chw);

    return inputData;
}

std::vector<Detection> TensorRTEngine::inferFrame(const std::vector<float>& rgbNormalizedBuffer, int origImgWidth, int origImgHeight, float confThreshold, float nmsThreshold) {
    std::vector<Detection> detections;

    if (!isInitialized_) {
        return detections;
    }

    std::vector<float> rawOutput(outputSize_ / sizeof(float), 0.0f);
    executeInference(rgbNormalizedBuffer, rawOutput);
    detections = postProcess(rawOutput, getTensorShape(), origImgWidth, origImgHeight, confThreshold, nmsThreshold);

    return detections;
}

std::vector<Detection> TensorRTEngine::inferFrameDetailed(const std::vector<float>& rgbNormalizedBuffer, InferenceTimingDetailed& timing, int origImgWidth, int origImgHeight, float confThreshold, float nmsThreshold) {
    std::vector<Detection> detections;

    if (!isInitialized_) {
        return detections;
    }

    std::vector<float> rawOutput(outputSize_ / sizeof(float), 0.0f);
    executeInferenceDetailed(rgbNormalizedBuffer, rawOutput, timing.h2dMs, timing.gpuComputeMs, timing.d2hMs);

    auto tDecodeStart = std::chrono::high_resolution_clock::now();
    detections = postProcess(rawOutput, getTensorShape(), origImgWidth, origImgHeight, confThreshold, nmsThreshold);
    auto tDecodeEnd = std::chrono::high_resolution_clock::now();
    timing.decodeMs = std::chrono::duration<float, std::milli>(tDecodeEnd - tDecodeStart).count();

    return detections;
}

std::vector<Detection> TensorRTEngine::inferFrameGPUDetailed(const cv::Mat& frame, InferenceTimingDetailed& timing, float confThreshold, float nmsThreshold) {
    if (!isInitialized_ || frame.empty()) return {};
    const auto raw = inferRawFrame(frame, true, false);
    timing = raw.timing;
    const auto start = std::chrono::steady_clock::now();
    auto detections = postProcess(raw.rawOutput, raw.shape, frame.cols, frame.rows, confThreshold, nmsThreshold);
    timing.decodeMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    return detections;
}

RawDetectorInference TensorRTEngine::inferRawFrame(const cv::Mat& frame, bool cudaPreprocessing, bool includeInputTensor) {
    if (!isInitialized_) throw std::runtime_error("Raw inference requires initialized detector.");
    if (frame.empty() || frame.type() != CV_8UC3) throw std::invalid_argument("Raw inference requires nonempty BGR8 frame.");
    RawDetectorInference result;
    result.shape = getTensorShape(); result.sourceWidth = frame.cols; result.sourceHeight = frame.rows;
    result.rawOutput.resize(result.shape.outputElements());
    if (!cudaPreprocessing) {
        result.preprocessingRevision = "opencv-linear-letterbox-round-pad114-bgr8-rgb-chw-v1";
        const auto start = std::chrono::steady_clock::now();
        auto input = preprocessOpenCVFrame(frame, inputW_, inputH_);
        result.timing.preprocessMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        executeInferenceDetailed(input, result.rawOutput, result.timing.h2dMs, result.timing.gpuComputeMs, result.timing.d2hMs);
        if (includeInputTensor) result.inputTensor = std::move(input);
        return result;
    }
    result.preprocessingRevision = "cuda-nearest-floor-letterbox-round-pad114-bgr8-rgb-chw-v1";

#ifdef HAS_TENSORRT
    // The kernel requires contiguous rows. A ROI can carry its parent's stride;
    // uploading total()*elemSize() while preserving that stride is unsafe.
    const cv::Mat contiguous = frame.isContinuous() ? frame : frame.clone();
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    size_t reqBGRSize = contiguous.total() * contiguous.elemSize();

    if (deviceBGRCapacity_ < reqBGRSize) {
        if (deviceBGRBuffer_) { checkCuda(cudaFree(deviceBGRBuffer_), "release BGR buffer"); deviceBGRBuffer_ = nullptr; }
        checkCuda(cudaMalloc(&deviceBGRBuffer_, reqBGRSize), "allocate BGR buffer");
        deviceBGRCapacity_ = reqBGRSize;
    }

    cudaEvent_t startH2D = static_cast<cudaEvent_t>(evStartH2D_);
    cudaEvent_t stopH2D = static_cast<cudaEvent_t>(evStopH2D_);
    cudaEvent_t startPrep = static_cast<cudaEvent_t>(evStartPrep_);
    cudaEvent_t stopPrep = static_cast<cudaEvent_t>(evStopPrep_);
    cudaEvent_t startExec = static_cast<cudaEvent_t>(evStartExec_);
    cudaEvent_t stopExec = static_cast<cudaEvent_t>(evStopExec_);
    cudaEvent_t startD2H = static_cast<cudaEvent_t>(evStartD2H_);
    cudaEvent_t stopD2H = static_cast<cudaEvent_t>(evStopD2H_);

    // 1. Copy raw frame BGR bytes directly to GPU memory (H2D)
    checkCuda(cudaEventRecord(startH2D, stream), "record H2D begin");
    checkCuda(cudaMemcpyAsync(deviceBGRBuffer_, contiguous.data, reqBGRSize, cudaMemcpyHostToDevice, stream), "upload BGR");
    checkCuda(cudaEventRecord(stopH2D, stream), "record H2D end");

    // 2. Launch CUDA Letterbox Preprocessing Kernel (NV12/BGR -> RGB planar CHW float)
    checkCuda(cudaEventRecord(startPrep, stream), "record preprocess begin");
    launchLetterboxPreprocessKernel(
        static_cast<const unsigned char*>(deviceBGRBuffer_), contiguous.cols, contiguous.rows, static_cast<int>(contiguous.step),
        static_cast<float*>(deviceInputBuffer_), inputW_, inputH_,
        stream
    );
    checkCuda(cudaGetLastError(), "launch preprocessing");
    checkCuda(cudaEventRecord(stopPrep, stream), "record preprocess end");

    // 3. TensorRT GPU Compute
    checkCuda(cudaEventRecord(startExec, stream), "record inference begin");
    auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);
    if (!ctx->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueueV3 failed.");
    checkCuda(cudaEventRecord(stopExec, stream), "record inference end");

    // 4. Device to Host Output Copy (into Pinned Host Memory)
    checkCuda(cudaEventRecord(startD2H, stream), "record D2H begin");
    checkCuda(cudaMemcpyAsync(hostOutputBuffer_, deviceOutputBuffer_, outputSize_, cudaMemcpyDeviceToHost, stream), "download raw output");
    if (includeInputTensor) {
        result.inputTensor.resize(result.shape.inputElements());
        checkCuda(cudaMemcpyAsync(hostInputBuffer_, deviceInputBuffer_, inputSize_, cudaMemcpyDeviceToHost, stream), "download actual preprocessed input");
    }
    checkCuda(cudaEventRecord(stopD2H, stream), "record D2H end");
    checkCuda(cudaStreamSynchronize(stream), "synchronize inference");

    if (hostOutputBuffer_) {
        std::memcpy(result.rawOutput.data(), hostOutputBuffer_, outputSize_);
    }
    if (includeInputTensor) std::memcpy(result.inputTensor.data(), hostInputBuffer_, inputSize_);

    checkCuda(cudaEventElapsedTime(&result.timing.h2dMs, startH2D, stopH2D), "time H2D");
    checkCuda(cudaEventElapsedTime(&result.timing.preprocessMs, startPrep, stopPrep), "time preprocessing");
    checkCuda(cudaEventElapsedTime(&result.timing.gpuComputeMs, startExec, stopExec), "time inference");
    checkCuda(cudaEventElapsedTime(&result.timing.d2hMs, startD2H, stopD2H), "time D2H");
#else
    throw std::runtime_error("CUDA preprocessing requires TensorRT build.");
#endif
    return result;
}

std::vector<Detection> TensorRTEngine::inferFrameGPU(const cv::Mat& frame, float confThreshold, float nmsThreshold) {
    InferenceTimingDetailed dummyTiming;
    return inferFrameGPUDetailed(frame, dummyTiming, confThreshold, nmsThreshold);
}

void TensorRTEngine::executeInference(const std::vector<float>& inputCPU, std::vector<float>& outputCPU) {
    float h2d = 0, compute = 0, d2h = 0;
    executeInferenceDetailed(inputCPU, outputCPU, h2d, compute, d2h);
}

void TensorRTEngine::executeInferenceDetailed(const std::vector<float>& inputCPU, std::vector<float>& outputCPU, float& h2dMs, float& gpuComputeMs, float& d2hMs) {
    if (!isInitialized_) throw std::runtime_error("Inference requires initialized detector.");
    if (inputCPU.size() != inputSize_ / sizeof(float) || outputCPU.size() != outputSize_ / sizeof(float))
        throw std::invalid_argument("Unexpected inference input/output element count.");
#ifdef HAS_TENSORRT
    const cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    const auto beginH2D = static_cast<cudaEvent_t>(evStartH2D_), endH2D = static_cast<cudaEvent_t>(evStopH2D_);
    const auto beginExec = static_cast<cudaEvent_t>(evStartExec_), endExec = static_cast<cudaEvent_t>(evStopExec_);
    const auto beginD2H = static_cast<cudaEvent_t>(evStartD2H_), endD2H = static_cast<cudaEvent_t>(evStopD2H_);
    std::memcpy(hostInputBuffer_, inputCPU.data(), inputSize_);
    checkCuda(cudaEventRecord(beginH2D, stream), "record H2D begin");
    checkCuda(cudaMemcpyAsync(deviceInputBuffer_, hostInputBuffer_, inputSize_, cudaMemcpyHostToDevice, stream), "upload input");
    checkCuda(cudaEventRecord(endH2D, stream), "record H2D end");
    checkCuda(cudaEventRecord(beginExec, stream), "record inference begin");
    if (!static_cast<nvinfer1::IExecutionContext*>(context_)->enqueueV3(stream))
        throw std::runtime_error("TensorRT enqueueV3 failed.");
    checkCuda(cudaEventRecord(endExec, stream), "record inference end");
    checkCuda(cudaEventRecord(beginD2H, stream), "record D2H begin");
    checkCuda(cudaMemcpyAsync(hostOutputBuffer_, deviceOutputBuffer_, outputSize_, cudaMemcpyDeviceToHost, stream), "download output");
    checkCuda(cudaEventRecord(endD2H, stream), "record D2H end");
    checkCuda(cudaStreamSynchronize(stream), "synchronize inference");
    std::memcpy(outputCPU.data(), hostOutputBuffer_, outputSize_);
    checkCuda(cudaEventElapsedTime(&h2dMs, beginH2D, endH2D), "time H2D");
    checkCuda(cudaEventElapsedTime(&gpuComputeMs, beginExec, endExec), "time inference");
    checkCuda(cudaEventElapsedTime(&d2hMs, beginD2H, endD2H), "time D2H");
#else
    throw std::runtime_error("Inference requires TensorRT build.");
#endif
}

std::vector<Detection> TensorRTEngine::postProcess(const std::vector<float>& rawOutput, int imgWidth, int imgHeight, float confThreshold, float nmsThreshold) {
    // Compatibility adapter for the original standard P3 640-square API only.
    // New engine paths always supply their actual contract via the overload.
    const int anchors = (640 / 8) * (640 / 8) + (640 / 16) * (640 / 16) + (640 / 32) * (640 / 32);
    if (rawOutput.size() % anchors) throw std::runtime_error("Malformed legacy YOLO output tensor");
    const int classes = static_cast<int>(rawOutput.size() / anchors) - 4;
    return postProcess(rawOutput, DetectorTensorShape{640, 640, classes, anchors}, imgWidth, imgHeight, confThreshold, nmsThreshold);
}

std::vector<Detection> TensorRTEngine::postProcess(const std::vector<float>& rawOutput, const DetectorTensorShape& shape, int imgWidth, int imgHeight, float confThreshold, float nmsThreshold) {
    shape.validate();
    if (rawOutput.size() != shape.outputElements() || imgWidth <= 0 || imgHeight <= 0 ||
        !std::isfinite(confThreshold) || confThreshold < 0 || confThreshold > 1 ||
        !std::isfinite(nmsThreshold) || nmsThreshold <= 0 || nmsThreshold > 1)
        throw std::invalid_argument("Malformed YOLO tensor, source geometry or decoding thresholds.");
    std::vector<Detection> finalDetections;
    std::vector<cv::Rect2d> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    const int numAnchors = shape.anchors;
    const int numClasses = shape.classes;

    // Un-letterboxing parameters based on target input shape (640x640)
    float scaleR = std::min(static_cast<float>(shape.inputWidth) / imgWidth, static_cast<float>(shape.inputHeight) / imgHeight);
    float padW = static_cast<float>((shape.inputWidth - static_cast<int>(std::round(imgWidth * scaleR))) / 2);
    float padH = static_cast<float>((shape.inputHeight - static_cast<int>(std::round(imgHeight * scaleR))) / 2);

    // Loop through all 8400 predictions
    for (int i = 0; i < numAnchors; ++i) {
        float maxProb = 0.0f;
        int maxClassId = -1;

        for (int c = 0; c < numClasses; ++c) {
            float prob = rawOutput[(4 + c) * numAnchors + i];
            if (!std::isfinite(prob) || prob < 0 || prob > 1) throw std::runtime_error("Nonfinite or out-of-range YOLO class score.");
            if (prob > maxProb) {
                maxProb = prob;
                maxClassId = c;
            }
        }

        if (maxProb > confThreshold) {
            // Extract bbox coordinates (YOLO outputs center x/y, width, height in 640x640 space)
            float cx = rawOutput[0 * numAnchors + i];
            float cy = rawOutput[1 * numAnchors + i];
            float w  = rawOutput[2 * numAnchors + i];
            float h  = rawOutput[3 * numAnchors + i];
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h))
                throw std::runtime_error("Nonfinite YOLO box geometry.");
            if (w <= 0 || h <= 0) continue;

            // Un-letterbox: subtract padding offset and divide by scale ratio
            float cxOrig = (cx - padW) / scaleR;
            float cyOrig = (cy - padH) / scaleR;
            float wOrig  = w / scaleR;
            float hOrig  = h / scaleR;

            double left   = cxOrig - wOrig / 2.0;
            double top    = cyOrig - hOrig / 2.0;
            double width  = wOrig;
            double height = hOrig;

            boxes.push_back(cv::Rect2d(left, top, width, height));
            confidences.push_back(maxProb);
            classIds.push_back(maxClassId);
        }
    }

    // Perform Non-Maximum Suppression (NMS) to remove overlapping duplicate boxes
    std::vector<int> nmsIndices;
    // Class-aware NMS: a person overlapping the table must survive.
    for (int cls = 0; cls < numClasses; ++cls) {
        std::vector<cv::Rect2d> cb; std::vector<float> cs; std::vector<int> ids, kept;
        for (size_t j = 0; j < boxes.size(); ++j) if (classIds[j] == cls) {
            cb.push_back(boxes[j]); cs.push_back(confidences[j]); ids.push_back(static_cast<int>(j));
        }
        cv::dnn::NMSBoxes(cb, cs, confThreshold, nmsThreshold, kept);
        for (int j : kept) nmsIndices.push_back(ids[j]);
    }
    std::sort(nmsIndices.begin(), nmsIndices.end(), [&](int a, int b){return confidences[a] > confidences[b];});

    // Package the final filtered detections
    for (int idx : nmsIndices) {
        Detection det;
        det.class_id = classIds[idx];
        det.confidence = confidences[idx];
        det.x = static_cast<float>(boxes[idx].x / imgWidth);
        det.y = static_cast<float>(boxes[idx].y / imgHeight);
        det.w = static_cast<float>(boxes[idx].width / imgWidth);
        det.h = static_cast<float>(boxes[idx].height / imgHeight);
        finalDetections.push_back(det);
    }

    return finalDetections;
}
