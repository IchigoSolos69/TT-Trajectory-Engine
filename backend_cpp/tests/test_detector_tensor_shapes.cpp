#include "tensorrt_engine.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <functional>

static int checks = 0;
static void require(bool valid, const std::string& name) { if (!valid) throw std::runtime_error(name); ++checks; std::cout << "PASS " << name << '\n'; }
static void rejects(const std::function<void()>& call, const std::string& name) { bool failed = false; try { call(); } catch (const std::exception&) { failed = true; } require(failed, name); }
static int anchors(int width, int height, bool p2) { int count = 0; for (int stride : {4,8,16,32}) if (p2 || stride != 4) count += width / stride * (height / stride); return count; }
int main(int argc, char** argv) try {
    for (const auto& size : {cv::Size(640,640), cv::Size(960,960), cv::Size(960,544)}) for (bool p2 : {false,true}) {
        const DetectorTensorShape shape{size.width,size.height,1,anchors(size.width,size.height,p2)};
        for (const auto& source : {cv::Size(1920,1080), cv::Size(1280,721), cv::Size(721,1280), cv::Size(641,479)}) {
            cv::Mat frame(source,CV_8UC3,cv::Scalar(0,0,255));
            const auto input = TensorRTEngine::preprocessOpenCVFrame(frame,shape.inputWidth,shape.inputHeight);
            const float scale = std::min(float(size.width)/source.width,float(size.height)/source.height);
            const int left = (size.width-int(std::round(source.width*scale)))/2, top = (size.height-int(std::round(source.height*scale)))/2;
            std::vector<float> raw(shape.outputElements(),0);
            const size_t idx = shape.anchors - 1; // Proves decoder reaches all anchors, including P2 tail.
            raw[idx] = 123*scale+left; raw[shape.anchors+idx] = 234*scale+top;
            raw[2*shape.anchors+idx] = raw[3*shape.anchors+idx] = 10;
            raw[4*shape.anchors+idx] = .9f;
            const auto detections = TensorRTEngine::postProcess(raw,shape,source.width,source.height);
            require(input.size()==shape.inputElements() && detections.size()==1 &&
                std::abs((detections[0].x+detections[0].w/2)*source.width-123)<.002f &&
                std::abs((detections[0].y+detections[0].h/2)*source.height-234)<.002f,
                "shape/source coordinate round trip "+std::to_string(size.width)+"x"+std::to_string(size.height)+" "+(p2?"P2":"P3")+" source "+std::to_string(source.width)+"x"+std::to_string(source.height));
        }
    }
    const DetectorTensorShape scene{960,960,2,18900};
    std::vector<float> sceneRaw(scene.outputElements());
    for (int i=0;i<2;++i) { sceneRaw[i]=480;sceneRaw[scene.anchors+i]=480;sceneRaw[2*scene.anchors+i]=100;sceneRaw[3*scene.anchors+i]=100;sceneRaw[(4+i)*scene.anchors+i]=.9f; }
    require(TensorRTEngine::postProcess(sceneRaw,scene,960,960).size()==2,"960 class-aware scene NMS");
    rejects([&]{TensorRTEngine::postProcess(std::vector<float>(sceneRaw.size()-1),scene,960,960);},"reject wrong raw element count");
    rejects([&]{TensorRTEngine::postProcess(sceneRaw,scene,0,960);},"reject zero source width");
    rejects([&]{TensorRTEngine::postProcess(sceneRaw,scene,960,960,std::numeric_limits<float>::quiet_NaN());},"reject nonfinite confidence threshold");
    auto malformed=sceneRaw;malformed[4*scene.anchors]=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{TensorRTEngine::postProcess(malformed,scene,960,960);},"reject nonfinite class score");
    malformed=sceneRaw;malformed[0]=std::numeric_limits<float>::infinity();
    rejects([&]{TensorRTEngine::postProcess(malformed,scene,960,960);},"reject nonfinite selected geometry");
    rejects([&]{DetectorTensorShape{640,640,1,0}.validate();},"reject dynamic/nonpositive anchors");
    rejects([&]{DetectorTensorShape{641,640,1,8400}.validate();},"reject unsupported non-stride-aligned input");
    rejects([&]{DetectorTensorShape{640,640,0,8400}.validate();},"reject missing classes");
    // Optional real engine verification; no model is created or overwritten.
    if (argc>1) {
        if (argc != 5) throw std::invalid_argument("Optional arguments: ENGINE EXPECTED_SIZE EXPECTED_ANCHORS CLASSES");
        const int expectedSize=std::stoi(argv[2]), expectedAnchors=std::stoi(argv[3]), classes=std::stoi(argv[4]);
        TensorRTEngine detector(argv[1],640,640,classes); require(detector.init(),"real engine initialization");
        const auto shape=detector.getTensorShape();
        require(shape.inputWidth==expectedSize && shape.inputHeight==expectedSize && shape.anchors==expectedAnchors && shape.classes==classes,"actual engine contract overrides640hints");
        cv::Mat parent(583,1033,CV_8UC3); for (int y=0;y<parent.rows;++y) for (int x=0;x<parent.cols;++x) parent.at<cv::Vec3b>(y,x)={static_cast<uchar>(x%256),static_cast<uchar>(y%256),static_cast<uchar>((x+y)%256)};
        const cv::Mat source=parent(cv::Rect(3,2,1027,579));
        const auto cpu=detector.inferRawFrame(source,false,true), gpu=detector.inferRawFrame(source,true,true);
        require(cpu.inputTensor==TensorRTEngine::preprocessOpenCVFrame(source,shape.inputWidth,shape.inputHeight),"raw CPU dump is actual OpenCV input");
        require(cpu.rawOutput.size()==shape.outputElements() && gpu.rawOutput.size()==shape.outputElements() && gpu.inputTensor.size()==shape.inputElements(),"actual raw tensors match engine allocation");
        const float scale=std::min(float(shape.inputWidth)/source.cols,float(shape.inputHeight)/source.rows);
        const int width=int(std::round(source.cols*scale)),height=int(std::round(source.rows*scale)),left=(shape.inputWidth-width)/2,top=(shape.inputHeight-height)/2;
        float maxDelta=0;const size_t channel=size_t(shape.inputWidth)*shape.inputHeight;
        for (int y=0;y<shape.inputHeight;++y) for (int x=0;x<shape.inputWidth;++x) for (int c=0;c<3;++c) {
            float expected=114.f/255;
            if(x>=left && x<left+width && y>=top && y<top+height) {
                const int sx=std::min(source.cols-1,std::max(0,int(std::floor((x-left)/scale)))),sy=std::min(source.rows-1,std::max(0,int(std::floor((y-top)/scale))));
                expected=source.at<cv::Vec3b>(sy,sx)[2-c]/255.f;
            }
            maxDelta=std::max(maxDelta,std::abs(expected-gpu.inputTensor[c*channel+y*shape.inputWidth+x]));
        }
        require(maxDelta<.000001f,"CUDA actual dimensions/color/padding/noncontiguousROI independent nearest reference");
        std::vector<float> badInput(shape.inputElements()-1), output(shape.outputElements());
        rejects([&]{detector.executeInference(badInput,output);},"real engine rejects truncated input");
        std::vector<float> correctInput(shape.inputElements()),badOutput(shape.outputElements()-1);
        rejects([&]{detector.executeInference(correctInput,badOutput);},"real engine rejects truncated output");
        rejects([&]{detector.inferRawFrame(cv::Mat(10,10,CV_8UC1));},"real engine rejects unsupported image type");
        require(cpu.preprocessingRevision!=gpu.preprocessingRevision,"different interpolation provenance preserved");
    }
    std::cout<<"TOTAL "<<checks<<" checks passed\n";return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
