#include "temporal_recovery_window.hpp"
#include <iostream>
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main()try{
    TemporalRecoveryWindow window;
    cv::Mat frame(256,320,CV_8UC3,cv::Scalar(10,20,30));
    for(int i=1;i<=5;++i)require(!window.update(i,i/120.,frame,cv::Point2f(100+i,100)),"Detector bypass failed");
    for(int i=6;i<=9;++i){
        auto r=window.update(i,i/120.,frame,std::nullopt);
        require(r.has_value()&&r->missedFrames==i-5,"Recovery gap budget failed");
        require(cv::norm(r->predicted-cv::Point2f(float(100+i),100))<.001,"Causal extrapolation failed");
        require(r->input.size()==15*128*128,"Tensor size failed");
        require(std::abs(r->input[0]-30.f/255)<1e-6&&std::abs(r->input[16384]-20.f/255)<1e-6&&std::abs(r->input[32768]-10.f/255)<1e-6,"RGB layout failed");
    }
    require(!window.update(10,10/120.,frame,std::nullopt),"Recovery exceeded four source frames");
    require(!window.update(11,11/120.,frame,std::nullopt),"Unbounded inferred feedback");
    window.update(12,12/120.,frame,cv::Point2f(112,100));
    require(!window.update(14,14/120.,frame,std::nullopt),"Skipped source frame retained temporal input");
    window.reset();
    for(int i=1;i<=5;++i)window.update(i,i/120.,frame,cv::Point2f(2,2));
    auto edge=window.update(6,6/120.,frame,std::nullopt);
    require(edge&&std::abs(edge->input[0]-114.f/255)<1e-6,"ROI border mismatch");
    require(!window.update(7,1.,frame,std::nullopt),"Timestamp discontinuity retained anchors");
    window.reset();
    for(int i=1;i<=4;++i){
        cv::Mat changing(256,320,CV_8UC3,cv::Scalar(i,2*i,3*i));
        window.update(i,i/120.,changing,cv::Point2f(110,110));
    }
    cv::Mat current(256,320,CV_8UC3,cv::Scalar(5,10,15));
    auto order=window.update(5,5/120.,current,std::nullopt);
    require(order&&std::abs(order->input[0]-3.f/255)<1e-6&&
        std::abs(order->input[12*128*128]-15.f/255)<1e-6,"Temporal tensor is not oldest to current");
    bool invalid=false;try{window.update(8,2.,cv::Mat(),std::nullopt);}catch(const std::runtime_error&){invalid=true;}
    require(invalid,"Invalid source was silently swallowed");
    std::cout<<"PASS: causal ROI, bounded recovery, source gaps, RGB layout, border padding, invalid input\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
