#include "tensorrt_engine.hpp"
#include "scene_geometry.hpp"
#include "scene_postprocess.hpp"
#include <fstream>
#include <set>
#include <iomanip>
int main(int argc,char**argv) try {
    if(argc!=5) throw std::runtime_error("Usage: dump_scene_predictions ENGINE VIDEO FRAME_IDS OUTPUT.csv");
    TensorRTEngine model(argv[1],640,640,2); if(!model.init()) return 1;
    std::ifstream ids(argv[3]); if(!ids) throw std::runtime_error("Cannot read frame IDs");
    std::set<int> wanted; int id; while(ids>>id)wanted.insert(id);
    if(wanted.empty())throw std::runtime_error("Empty evaluation set");
    cv::VideoCapture cap(argv[2],cv::CAP_FFMPEG); if(!cap.isOpened())throw std::runtime_error("Cannot decode video");
    std::ofstream out(argv[4]);if(!out)throw std::runtime_error("Cannot write predictions");
    out<<"frame,class,confidence,x,y,w,h,quad_x0,quad_y0,quad_x1,quad_y1,quad_x2,quad_y2,quad_x3,quad_y3\n"<<std::fixed<<std::setprecision(6);
    int count=0;
    for(int frameId=1;frameId<=*wanted.rbegin();++frameId){
        if(!cap.grab())throw std::runtime_error("Video ended before final requested frame");
        if(!wanted.count(frameId))continue;
        cv::Mat frame;if(!cap.retrieve(frame))throw std::runtime_error("Cannot retrieve frame");
        const auto raw=model.inferFrame(TensorRTEngine::preprocessOpenCVFrame(frame),frame.cols,frame.rows,.05f);
        const auto detections=postprocessScene(raw).accepted;
        const auto quad=findTableSurface(frame,detections);
        for(const auto& d:detections){
            out<<frameId<<","<<d.class_id<<","<<d.confidence<<","<<d.x<<","<<d.y<<","<<d.w<<","<<d.h;
            for(int i=0;i<4;++i)out<<","<<(quad.size()==4?quad[i].x/frame.cols:-1)<<","<<(quad.size()==4?quad[i].y/frame.rows:-1);
            out<<"\n";
        }
        // Explicit sentinel records successful inference even with no detections.
        out<<frameId<<",-1,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1\n";
        ++count;
    }
    std::cout<<"COMPLETE: "<<count<<" labeled frames. Engine SHA256="<<model.getSha256Hash()<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
