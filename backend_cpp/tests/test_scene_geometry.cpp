#include "scene_geometry.hpp"
#include "scene_postprocess.hpp"
#include <stdexcept>
#include <iostream>
static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main() {
    cv::Mat frame(540,960,CV_8UC3,cv::Scalar(15,15,15));
    const std::vector<cv::Point> quad{{260,220},{700,220},{810,420},{150,420}};
    cv::fillConvexPoly(frame,quad,cv::Scalar(180,80,20));
    cv::rectangle(frame,{0,0,960,100},cv::Scalar(200,50,10),cv::FILLED);
    const Detection table{1,.95f,145.f/960,215.f/540,670.f/960,215.f/540};
    const auto found=findTableSurface(frame,{table});
    require(found.size()==4,"Expected refined quadrilateral inside learned ROI");
    for(size_t i=0;i<4;++i) require(cv::norm(found[i]-cv::Point2f(quad[i]))<8,"Wrong surface corner");
    require(findTableSurface(frame,{}).empty(),"No table prior must mean no calibration");
    auto low=table;low.confidence=.2f;
    require(findTableSurface(frame,{low}).empty(),"Uncertain table must not calibrate");
    cv::Mat blank(540,960,CV_8UC3,cv::Scalar(0,0,0));
    require(findTableSurface(blank,{table}).empty(),"No invented default corners");
    std::vector<float> output(6*8400,0);
    // Two identical boxes of different classes must both survive NMS.
    for(int i=0;i<2;++i){output[i]=320;output[8400+i]=320;output[16800+i]=100;output[25200+i]=100;output[(4+i)*8400+i]=.9f;}
    require(TensorRTEngine::postProcess(output,640,640,.4f).size()==2,"Class-aware NMS lost a class");
    const Detection person{0,.8f,.20f,.40f,.10f,.48f};
    const Detection leg{0,.51f,.26f,.75f,.04f,.11f};
    const auto clean=postprocessScene({person,leg,table});
    require(clean.accepted.size()==2&&clean.suppressedPersonParts.size()==1,"Contained lower-leg duplicate survived");
    auto other=person;other.x+=.04f;other.confidence=.6f;
    require(postprocessScene({person,other}).accepted.size()==2,"Overlapping full people must survive");
    auto upper=leg;upper.y=.45f;
    require(postprocessScene({person,upper}).accepted.size()==2,"Contained upper person is ambiguous and must survive");
    auto stronger=leg;stronger.confidence=.9f;
    require(postprocessScene({person,stronger}).accepted.size()==2,"Stronger small observation must survive");
    auto outside=leg;outside.x=.29f;
    require(postprocessScene({person,outside}).accepted.size()==2,"Non-contained person must survive");
    std::cout<<"PASS: scene geometry, unavailable calibration, class-aware NMS"<<std::endl;
}
