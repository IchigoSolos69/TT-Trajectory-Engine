#include "tensorrt_engine.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>

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
    return "models/yolov11_ttgames.engine";
}

TensorRTEngine::TensorRTEngine(const std::string& enginePath, int inputWidth, int inputHeight)
    : enginePath_(resolveEnginePath(enginePath)),
      inputW_(inputWidth),
      inputH_(inputHeight),
      isInitialized_(false),
      context_(nullptr),
      engine_(nullptr),
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
    if (context_) delete static_cast<nvinfer1::IExecutionContext*>(context_);
    if (engine_) delete static_cast<nvinfer1::ICudaEngine*>(engine_);
    if (deviceInputBuffer_) cudaFree(deviceInputBuffer_);
    if (deviceOutputBuffer_) cudaFree(deviceOutputBuffer_);
#endif
    if (hostInputBuffer_) free(hostInputBuffer_);
    if (hostOutputBuffer_) free(hostOutputBuffer_);
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

    hostInputBuffer_ = malloc(inputSize_);
    hostOutputBuffer_ = malloc(outputSize_);

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

    std::cout << "[TensorRTEngine] CUDA buffers allocated successfully on Device GPU." << std::endl;
    std::cout << "[TensorRTEngine] Input Tensor Shape  : [1, 3, " << inputW_ << ", " << inputH_ << "] (Float32 CHW)" << std::endl;
    std::cout << "[TensorRTEngine] Output Tensor Shape : [1, 5, 8400] (1-class ball detector FP16 engine)" << std::endl;
    delete runtime;
#else
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

std::vector<Detection> TensorRTEngine::inferFrame(const std::vector<float>& rgbNormalizedBuffer) {
    std::vector<Detection> detections;

    if (!isInitialized_) {
        return detections;
    }

    std::vector<float> rawOutput(84 * 8400, 0.0f);
    executeInference(rgbNormalizedBuffer, rawOutput);
    detections = postProcess(rawOutput, inputW_, inputH_, 0.15f, 0.45f);

    return detections;
}

void TensorRTEngine::executeInference(const std::vector<float>& inputCPU, std::vector<float>& outputCPU) {
#ifdef HAS_TENSORRT
    if (context_) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // 1. Host to Device (CPU -> GPU)
        cudaMemcpyAsync(deviceInputBuffer_, inputCPU.data(), inputSize_, cudaMemcpyHostToDevice, stream);

        // 2. Set tensor addresses and execute TRT 10 inference
        auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);
        auto* eng = static_cast<nvinfer1::ICudaEngine*>(engine_);
        
        for (int i = 0; i < eng->getNbIOTensors(); ++i) {
            const char* tName = eng->getIOTensorName(i);
            if (eng->getTensorIOMode(tName) == nvinfer1::TensorIOMode::kINPUT) {
                ctx->setTensorAddress(tName, deviceInputBuffer_);
            } else {
                ctx->setTensorAddress(tName, deviceOutputBuffer_);
            }
        }
        
        ctx->enqueueV3(stream);

        // 3. Device to Host (GPU -> CPU)
        cudaMemcpyAsync(outputCPU.data(), deviceOutputBuffer_, outputSize_, cudaMemcpyDeviceToHost, stream);

        // 4. Block CPU until stream finishes
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
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

