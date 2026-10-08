#include "observed_spin.hpp"
#include <iostream>
#include <stdexcept>

void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main() try {
    ObservedSpinEstimator estimator;
    cv::Mat texture(128,128,CV_8UC3,cv::Scalar(30,30,30));
    cv::circle(texture,{64,64},45,cv::Scalar(240,240,240),-1);
    for(auto p:std::vector<cv::Point>{{48,49},{69,44},{81,60},{69,76},{46,73},{57,61}})
        cv::rectangle(texture,cv::Rect(p.x,p.y,5,7),cv::Scalar(40,40,40),-1);
    ObservedSpin result;
    for(int i=0;i<9;++i){
        cv::Mat frame;
        cv::warpAffine(texture,frame,cv::getRotationMatrix2D({64,64},i*5,1),texture.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(30,30,30));
        result=estimator.update(frame,{17,17,94,94},.05f);
    }
    check(result.available,"Marked rotating ball must yield an estimate");
    check(std::abs(result.rpm-16.6667f)<3,"Known 5 degree / 50 ms rotation must match within 3 RPM");
    std::cout<<"Measured synthetic rotation: "<<result.rpm<<" RPM; expected 16.667 RPM\n";
    check(!estimator.update(texture,{50,50,12,12},.05f).available,"Tiny ball must remain unresolved");
    check(!estimator.update(texture,{17,17,94,94},.2f).available,"Large time gap must reset estimation");
    estimator.reset();
    cv::Mat blank(128,128,CV_8UC3,cv::Scalar(30,30,30));cv::circle(blank,{64,64},45,cv::Scalar(240,240,240),-1);
    for(int i=0;i<8;++i)check(!estimator.update(blank,{17,17,94,94},.05f).available,"Unmarked sphere must not invent spin");
    std::cout<<"PASS: observed rotation, unresolved texture, tiny ball and temporal discontinuity\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
