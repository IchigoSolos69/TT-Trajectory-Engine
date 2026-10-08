#include "temporal_recovery.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
int main(int argc,char** argv)try{
    if(argc!=5)throw std::runtime_error("Usage: temporal.engine input.bin output.bin timing.json");
    TemporalRecoveryEngine model(argv[1]);
    std::ifstream input(argv[2],std::ios::binary);std::ofstream output(argv[3],std::ios::binary),timing(argv[4]);
    if(!input||!output||!timing)throw std::runtime_error("Cannot open parity files");
    std::vector<float> sample(TemporalRecoveryEngine::inputElements);int count=0;
    timing<<"{\"scope\":\"One pass over fixed ROI tensors; excludes capture, ROI extraction and dashboard. Not a live-stream FPS benchmark.\",\"samples\":[";
    while(input.read(reinterpret_cast<char*>(sample.data()),sample.size()*sizeof(float))){
        auto result=model.infer(sample);
        output.write(reinterpret_cast<char*>(result.logits.data()),result.logits.size()*sizeof(float));
        if(count++)timing<<',';timing<<"{\"gpu_ms\":"<<result.gpuMs<<",\"wall_ms\":"<<result.wallMs<<"}";
    }
    if(!input.eof()||input.gcount()!=0||count==0)throw std::runtime_error("Incomplete/empty temporal input tensor");
    bool rejects=false;try{model.infer({0.f});}catch(const std::runtime_error&){rejects=true;}
    if(!rejects)throw std::runtime_error("Temporal model accepted an invalid input shape");
    timing<<"],\"invalid_shape_rejected\":true}";
    if(!output.good()||!timing.good())throw std::runtime_error("Temporal output write failed");
    std::cout<<"Completed "<<count<<" native temporal inferences and input-shape rejection test\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
