#include "temporal_recovery.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cmath>
#ifdef HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
namespace {
class RecoveryLogger:public nvinfer1::ILogger {
    void log(Severity severity,const char* text)noexcept override{
        if(severity<=Severity::kWARNING)std::cerr<<"[TemporalRecovery] "<<text<<'\n';
    }
};
void checked(cudaError_t result,const char* stage){if(result!=cudaSuccess)throw std::runtime_error(std::string(stage)+": "+cudaGetErrorString(result));}
}
struct TemporalRecoveryEngine::Impl {
    RecoveryLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,stop=nullptr;
    void* input=nullptr;void* output=nullptr;
    ~Impl(){
        if(stream)cudaStreamSynchronize(stream);
        if(input)cudaFree(input);if(output)cudaFree(output);
        if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);
        if(stream)cudaStreamDestroy(stream);
    }
};
TemporalRecoveryEngine::TemporalRecoveryEngine(const std::string& path):impl_(std::make_unique<Impl>()){
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file||file.tellg()<=0)throw std::runtime_error("Cannot open temporal engine: "+path);
    const auto bytes=static_cast<size_t>(file.tellg());std::vector<char> data(bytes);file.seekg(0);
    if(!file.read(data.data(),bytes))throw std::runtime_error("Cannot read temporal engine");
    auto& i=*impl_;i.runtime.reset(nvinfer1::createInferRuntime(i.logger));
    if(!i.runtime)throw std::runtime_error("Temporal TensorRT runtime creation failed");
    i.engine.reset(i.runtime->deserializeCudaEngine(data.data(),data.size()));
    if(!i.engine||i.engine->getNbIOTensors()!=2)throw std::runtime_error("Temporal engine requires exactly two tensors");
    const char* inputName=nullptr;const char* outputName=nullptr;
    for(int n=0;n<2;++n){const auto name=i.engine->getIOTensorName(n);const auto shape=i.engine->getTensorShape(name);
        if(i.engine->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT)throw std::runtime_error("Temporal engine requires FP32 I/O");
        if(i.engine->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT){
            if(shape.nbDims!=4||shape.d[0]!=1||shape.d[1]!=15||shape.d[2]!=128||shape.d[3]!=128)throw std::runtime_error("Temporal input shape must be [1,15,128,128]");inputName=name;
        }else{
            if(shape.nbDims!=2||shape.d[0]!=1||shape.d[1]!=1025)throw std::runtime_error("Temporal output shape must be [1,1025]");outputName=name;
        }
    }
    if(!inputName||!outputName)throw std::runtime_error("Missing temporal input/output");
    i.context.reset(i.engine->createExecutionContext());if(!i.context)throw std::runtime_error("Temporal context creation failed");
    checked(cudaStreamCreate(&i.stream),"Temporal stream");checked(cudaEventCreate(&i.start),"Temporal start event");checked(cudaEventCreate(&i.stop),"Temporal stop event");
    checked(cudaMalloc(&i.input,inputElements*sizeof(float)),"Temporal input allocation");checked(cudaMalloc(&i.output,outputElements*sizeof(float)),"Temporal output allocation");
    if(!i.context->setTensorAddress(inputName,i.input)||!i.context->setTensorAddress(outputName,i.output))throw std::runtime_error("Temporal tensor binding failed");
}
TemporalRecoveryEngine::~TemporalRecoveryEngine()=default;
TemporalRecoveryEngine::Result TemporalRecoveryEngine::infer(const std::vector<float>& input){
    if(input.size()!=inputElements)throw std::runtime_error("Incorrect temporal input length");
    for(float v:input)if(!std::isfinite(v)||v<0||v>1)throw std::runtime_error("Temporal input must be finite RGB in [0,1]");
    auto& i=*impl_;Result result;result.logits.resize(outputElements);
    const auto begin=std::chrono::steady_clock::now();
    checked(cudaMemcpyAsync(i.input,input.data(),inputElements*sizeof(float),cudaMemcpyHostToDevice,i.stream),"Temporal upload");
    checked(cudaEventRecord(i.start,i.stream),"Temporal timing start");
    if(!i.context->enqueueV3(i.stream))throw std::runtime_error("Temporal enqueueV3 failed");
    checked(cudaEventRecord(i.stop,i.stream),"Temporal timing stop");
    checked(cudaMemcpyAsync(result.logits.data(),i.output,outputElements*sizeof(float),cudaMemcpyDeviceToHost,i.stream),"Temporal download");
    checked(cudaStreamSynchronize(i.stream),"Temporal synchronize");
    checked(cudaEventElapsedTime(&result.gpuMs,i.start,i.stop),"Temporal elapsed time");
    result.wallMs=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-begin).count();
    for(float v:result.logits)if(!std::isfinite(v))throw std::runtime_error("Nonfinite temporal output");
    return result;
}
#else
struct TemporalRecoveryEngine::Impl {};
TemporalRecoveryEngine::TemporalRecoveryEngine(const std::string&){throw std::runtime_error("Temporal recovery requires TensorRT");}
TemporalRecoveryEngine::~TemporalRecoveryEngine()=default;
TemporalRecoveryEngine::Result TemporalRecoveryEngine::infer(const std::vector<float>&){throw std::runtime_error("Temporal recovery requires TensorRT");}
#endif
