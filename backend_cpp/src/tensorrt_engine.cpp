#include "tensorrt_engine.hpp"
#include "cuda_preprocess.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

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
#endif

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdlib>

static std::string computeSha256(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) return "HASH_ERROR_READ";

    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef4a394, 0x1e63d01e
    };

    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    auto rightRotate = [](uint32_t val, uint32_t bits) {
        return (val >> bits) | (val << (32 - bits));
    };

    uint64_t totalBits = 0;
    uint8_t buffer[64];
    size_t bytesRead = 0;
    bool padded = false;

    while (!padded) {
        file.read(reinterpret_cast<char*>(buffer), 64);
        bytesRead = file.gcount();
        totalBits += bytesRead * 8;

        if (bytesRead < 64) {
            buffer[bytesRead] = 0x80;
            bytesRead++;
            if (bytesRead <= 56) {
                std::memset(buffer + bytesRead, 0, 56 - bytesRead);
            } else {
                std::memset(buffer + bytesRead, 0, 64 - bytesRead);
                uint32_t W[64];
                for (int t = 0; t < 16; t++) {
                    W[t] = (buffer[t*4] << 24) | (buffer[t*4+1] << 16) | (buffer[t*4+2] << 8) | (buffer[t*4+3]);
                }
                for (int t = 16; t < 64; t++) {
                    uint32_t s0 = rightRotate(W[t-15], 7) ^ rightRotate(W[t-15], 18) ^ (W[t-15] >> 3);
                    uint32_t s1 = rightRotate(W[t-2], 17) ^ rightRotate(W[t-2], 19) ^ (W[t-2] >> 10);
                    W[t] = W[t-16] + s0 + W[t-7] + s1;
                }
                uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
                for (int t = 0; t < 64; t++) {
                    uint32_t S1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
                    uint32_t ch = (e & f) ^ ((~e) & g);
                    uint32_t temp1 = h + S1 + ch + K[t] + W[t];
                    uint32_t S0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
                    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                    uint32_t temp2 = S0 + maj;
                    h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
                }
                H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
                std::memset(buffer, 0, 56);
            }
            for (int i = 0; i < 8; i++) {
                buffer[56 + i] = (totalBits >> (56 - i * 8)) & 0xFF;
            }
            padded = true;
        }

        uint32_t W[64];
        for (int t = 0; t < 16; t++) {
            W[t] = (buffer[t*4] << 24) | (buffer[t*4+1] << 16) | (buffer[t*4+2] << 8) | (buffer[t*4+3]);
        }
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = rightRotate(W[t-15], 7) ^ rightRotate(W[t-15], 18) ^ (W[t-15] >> 3);
            uint32_t s1 = rightRotate(W[t-2], 17) ^ rightRotate(W[t-2], 19) ^ (W[t-2] >> 10);
            W[t] = W[t-16] + s0 + W[t-7] + s1;
        }
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
        for (int t = 0; t < 64; t++) {
            uint32_t S1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + K[t] + W[t];
            uint32_t S0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    std::ostringstream ss;
    for (int i = 0; i < 8; i++) {
        ss << std::hex << std::setw(8) << std::setfill('0') << H[i];
    }
    return ss.str();
}

static std::string getFileLastModifiedIso(const std::string& filePath) {
    try {
        namespace fs = std::filesystem;
        if (!fs::exists(filePath)) return "N/A";
        auto ftime = fs::last_write_time(filePath);
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
    if (!userPath.empty() && std::ifstream(userPath).good()) {
        return userPath;
    }
    const char* envPath = std::getenv("TT_ENGINE_PATH");
    if (envPath && std::ifstream(envPath).good()) {
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
        if (std::ifstream(path).good()) {
            std::string cPath = path;
            try {
                cPath = fs::canonical(path).string();
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
            try { fSize = fs::file_size(existingPaths[i]); } catch(...) {}
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

TensorRTEngine::TensorRTEngine(const std::string& enginePath, int inputWidth, int inputHeight)
    : enginePath_(resolveEnginePath(enginePath)),
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
    inputSize_ = 1 * 3 * inputW_ * inputH_ * sizeof(float);
    // Allocate space for up to 84 channels x 8400 anchors (supports both 5-channel ball and 84-channel COCO engines)
    outputSize_ = 1 * 84 * 8400 * sizeof(float); 
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
    std::cout << "[TensorRTEngine] Loading engine file: " << enginePath_ << std::endl;

    std::ifstream file(enginePath_, std::ios::binary);
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
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fileSizeBytes_);
    if (!engine) {
        std::cerr << "[TensorRTEngine] CRITICAL FATAL ERROR: Failed to deserialize CUDA engine. Exiting code 1." << std::endl;
        delete runtime;
        std::exit(1);
    }

    engine_ = engine;
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    context_ = context;

    // Allocate Pinned Host Memory for zero-copy DMA execution
    cudaMallocHost(&hostInputBuffer_, inputSize_);
    cudaMallocHost(&hostOutputBuffer_, outputSize_);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
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
    cudaMalloc(&deviceInputBuffer_, inputSize_);
    cudaMalloc(&deviceOutputBuffer_, outputSize_);

    // Set static tensor addresses once during init
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tName = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(tName) == nvinfer1::TensorIOMode::kINPUT) {
            context->setTensorAddress(tName, deviceInputBuffer_);
        } else {
            context->setTensorAddress(tName, deviceOutputBuffer_);
        }
    }

    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStartH2D_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStopH2D_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStartPrep_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStopPrep_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStartExec_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStopExec_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStartD2H_));
    cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&evStopD2H_));

    std::cout << "[TensorRTEngine] CUDA buffers allocated successfully on Device GPU (Pinned Host Memory Enabled)." << std::endl;
    std::cout << "[TensorRTEngine] Input Tensor Shape  : [1, 3, " << inputW_ << ", " << inputH_ << "] (Float32 CHW)" << std::endl;
    std::cout << "[TensorRTEngine] Output Tensor Shape : [1, 5, 8400] (1-class ball detector FP16 engine)" << std::endl;
    delete runtime;
#else
    hostInputBuffer_ = malloc(inputSize_);
    hostOutputBuffer_ = malloc(outputSize_);
    std::cout << "[TensorRTEngine] Input Tensor Shape  : [1, 3, " << inputW_ << ", " << inputH_ << "] (Float32 CHW)" << std::endl;
    std::cout << "[TensorRTEngine] Output Tensor Shape : [1, 5, 8400] (Host CPU buffer allocated)" << std::endl;
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
    // 1. Calculate letterbox scale ratio and padding for aspect ratio preservation (e.g. 1920x1080 -> 640x640)
    float scaleR = std::min(static_cast<float>(inputWidth) / inputFrame.cols, static_cast<float>(inputHeight) / inputFrame.rows);
    int unpadW = static_cast<int>(std::round(inputFrame.cols * scaleR)); // 640
    int unpadH = static_cast<int>(std::round(inputFrame.rows * scaleR)); // 360
    int padW = (inputWidth - unpadW) / 2;  // 0
    int padH = (inputHeight - unpadH) / 2; // 140

    cv::Mat resizedImage, paddedImage;
    cv::resize(inputFrame, resizedImage, cv::Size(unpadW, unpadH), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resizedImage, paddedImage, padH, padH, padW, padW, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

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

    std::vector<float> rawOutput(84 * 8400, 0.0f);
    executeInference(rgbNormalizedBuffer, rawOutput);
    detections = postProcess(rawOutput, origImgWidth, origImgHeight, confThreshold, nmsThreshold);

    return detections;
}

std::vector<Detection> TensorRTEngine::inferFrameDetailed(const std::vector<float>& rgbNormalizedBuffer, InferenceTimingDetailed& timing, int origImgWidth, int origImgHeight, float confThreshold, float nmsThreshold) {
    std::vector<Detection> detections;

    if (!isInitialized_) {
        return detections;
    }

    std::vector<float> rawOutput(84 * 8400, 0.0f);
    executeInferenceDetailed(rgbNormalizedBuffer, rawOutput, timing.h2dMs, timing.gpuComputeMs, timing.d2hMs);

    auto tDecodeStart = std::chrono::high_resolution_clock::now();
    detections = postProcess(rawOutput, origImgWidth, origImgHeight, confThreshold, nmsThreshold);
    auto tDecodeEnd = std::chrono::high_resolution_clock::now();
    timing.decodeMs = std::chrono::duration<float, std::milli>(tDecodeEnd - tDecodeStart).count();

    return detections;
}

std::vector<Detection> TensorRTEngine::inferFrameGPUDetailed(const cv::Mat& frame, InferenceTimingDetailed& timing, float confThreshold, float nmsThreshold) {
    std::vector<Detection> detections;
    if (!isInitialized_ || frame.empty()) return detections;

#ifdef HAS_TENSORRT
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    size_t reqBGRSize = frame.total() * frame.elemSize();

    if (deviceBGRCapacity_ < reqBGRSize) {
        if (deviceBGRBuffer_) cudaFree(deviceBGRBuffer_);
        cudaMalloc(&deviceBGRBuffer_, reqBGRSize);
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
    cudaEventRecord(startH2D, stream);
    cudaMemcpyAsync(deviceBGRBuffer_, frame.data, reqBGRSize, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(stopH2D, stream);

    // 2. Launch CUDA Letterbox Preprocessing Kernel (NV12/BGR -> RGB planar CHW float)
    cudaEventRecord(startPrep, stream);
    launchLetterboxPreprocessKernel(
        static_cast<const unsigned char*>(deviceBGRBuffer_), frame.cols, frame.rows, static_cast<int>(frame.step),
        static_cast<float*>(deviceInputBuffer_), inputW_, inputH_,
        stream
    );
    cudaEventRecord(stopPrep, stream);

    // 3. TensorRT GPU Compute
    cudaEventRecord(startExec, stream);
    auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);
    ctx->enqueueV3(stream);
    cudaEventRecord(stopExec, stream);

    // 4. Device to Host Output Copy (into Pinned Host Memory)
    cudaEventRecord(startD2H, stream);
    std::vector<float> rawOutput(84 * 8400, 0.0f);
    size_t copyBytes = std::min(outputSize_, rawOutput.size() * sizeof(float));
    cudaMemcpyAsync(hostOutputBuffer_ ? hostOutputBuffer_ : rawOutput.data(), deviceOutputBuffer_, copyBytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stopD2H, stream);

    cudaStreamSynchronize(stream);

    if (hostOutputBuffer_) {
        std::memcpy(rawOutput.data(), hostOutputBuffer_, copyBytes);
    }

    cudaEventElapsedTime(&timing.h2dMs, startH2D, stopH2D);
    cudaEventElapsedTime(&timing.preprocessMs, startPrep, stopPrep);
    cudaEventElapsedTime(&timing.gpuComputeMs, startExec, stopExec);
    cudaEventElapsedTime(&timing.d2hMs, startD2H, stopD2H);

    // 5. Unletterbox Postprocess
    auto tDecodeStart = std::chrono::high_resolution_clock::now();
    detections = postProcess(rawOutput, frame.cols, frame.rows, confThreshold, nmsThreshold);
    auto tDecodeEnd = std::chrono::high_resolution_clock::now();
    timing.decodeMs = std::chrono::duration<float, std::milli>(tDecodeEnd - tDecodeStart).count();
#endif

    return detections;
}

std::vector<Detection> TensorRTEngine::inferFrameGPU(const cv::Mat& frame, float confThreshold, float nmsThreshold) {
    InferenceTimingDetailed dummyTiming;
    return inferFrameGPUDetailed(frame, dummyTiming, confThreshold, nmsThreshold);
}

void TensorRTEngine::executeInference(const std::vector<float>& inputCPU, std::vector<float>& outputCPU) {
#ifdef HAS_TENSORRT
    if (context_ && stream_) {
        cudaStream_t stream = static_cast<cudaStream_t>(stream_);

        if (hostInputBuffer_) {
            std::memcpy(hostInputBuffer_, inputCPU.data(), inputSize_);
        }

        // 1. Host to Device (CPU -> GPU)
        cudaMemcpyAsync(deviceInputBuffer_, hostInputBuffer_ ? hostInputBuffer_ : inputCPU.data(), inputSize_, cudaMemcpyHostToDevice, stream);

        // 2. Execute TRT inference
        auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);
        ctx->enqueueV3(stream);

        // 3. Device to Host (GPU -> CPU)
        size_t copyBytes = std::min(outputSize_, outputCPU.size() * sizeof(float));
        cudaMemcpyAsync(hostOutputBuffer_ ? hostOutputBuffer_ : outputCPU.data(), deviceOutputBuffer_, copyBytes, cudaMemcpyDeviceToHost, stream);

        // 4. Block CPU until stream finishes
        cudaStreamSynchronize(stream);

        if (hostOutputBuffer_) {
            std::memcpy(outputCPU.data(), hostOutputBuffer_, copyBytes);
        }
    }
#endif
}

void TensorRTEngine::executeInferenceDetailed(const std::vector<float>& inputCPU, std::vector<float>& outputCPU, float& h2dMs, float& gpuComputeMs, float& d2hMs) {
#ifdef HAS_TENSORRT
    if (context_ && stream_) {
        cudaStream_t stream = static_cast<cudaStream_t>(stream_);
        cudaEvent_t startH2D, stopH2D, startExec, stopExec, startD2H, stopD2H;
        cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
        cudaEventCreate(&startExec); cudaEventCreate(&stopExec);
        cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

        if (hostInputBuffer_) {
            std::memcpy(hostInputBuffer_, inputCPU.data(), inputSize_);
        }

        // 1. H2D (Host to Device)
        cudaEventRecord(startH2D, stream);
        cudaMemcpyAsync(deviceInputBuffer_, hostInputBuffer_ ? hostInputBuffer_ : inputCPU.data(), inputSize_, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(stopH2D, stream);

        // 2. GPU Execution
        cudaEventRecord(startExec, stream);
        auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);
        ctx->enqueueV3(stream);
        cudaEventRecord(stopExec, stream);

        // 3. D2H (Device to Host)
        cudaEventRecord(startD2H, stream);
        size_t copyBytes = std::min(outputSize_, outputCPU.size() * sizeof(float));
        cudaMemcpyAsync(hostOutputBuffer_ ? hostOutputBuffer_ : outputCPU.data(), deviceOutputBuffer_, copyBytes, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(stopD2H, stream);

        cudaStreamSynchronize(stream);

        if (hostOutputBuffer_) {
            std::memcpy(outputCPU.data(), hostOutputBuffer_, copyBytes);
        }

        cudaEventElapsedTime(&h2dMs, startH2D, stopH2D);
        cudaEventElapsedTime(&gpuComputeMs, startExec, stopExec);
        cudaEventElapsedTime(&d2hMs, startD2H, stopD2H);

        cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
        cudaEventDestroy(startExec); cudaEventDestroy(stopExec);
        cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);
    }
#endif
}

std::vector<Detection> TensorRTEngine::postProcess(const std::vector<float>& rawOutput, int imgWidth, int imgHeight, float confThreshold, float nmsThreshold) {
    std::vector<Detection> finalDetections;
    std::vector<cv::Rect2d> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    const int numAnchors = 8400;
    const int totalOutputElements = static_cast<int>(rawOutput.size());
    const int numChannels = totalOutputElements / numAnchors; // 5 for 1-class ball model, 84 for COCO model

    if (numChannels != 5 && numChannels != 84) {
        std::cerr << "[TensorRTEngine] Warning: Unexpected output channels: " << numChannels << std::endl;
    }

    const int numClasses = numChannels - 4; // 1 class (ball) for 5-channel layout

    // Un-letterboxing parameters based on target input shape (640x640)
    float scaleR = std::min(640.0f / imgWidth, 640.0f / imgHeight);
    float padW = (640.0f - imgWidth * scaleR) / 2.0f;
    float padH = (640.0f - imgHeight * scaleR) / 2.0f;

    // Loop through all 8400 predictions
    for (int i = 0; i < numAnchors; ++i) {
        float maxProb = 0.0f;
        int maxClassId = -1;
        
        for (int c = 0; c < numClasses; ++c) {
            float prob = rawOutput[(4 + c) * numAnchors + i]; 
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
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

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
