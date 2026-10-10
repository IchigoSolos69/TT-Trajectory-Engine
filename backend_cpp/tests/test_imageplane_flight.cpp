#include "imageplane_flight.hpp"
#include <iostream>
#include <stdexcept>
static void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main()try{
    std::deque<ImagePlaneFlight::Point> truth;
    const auto trajectory=[](double t){return cv::Point2f(float(300+600*t+40*t*t),float(200-70*t+250*t*t));};
    for(int i=0;i<9;++i){double t=i/60.;truth.push_back({i+1,t,trajectory(t)});}
    const auto fit=ImagePlaneFlight::fit(truth);require(bool(fit),"Quadratic fit unavailable");
    for(double t:{.15,.175,.183333333})require(cv::norm(ImagePlaneFlight::position(*fit,t)-cv::Point2d(trajectory(t)))<.001,"Known quadratic forecast error");
    ImagePlaneFlight flight;std::string packet;
    for(const auto& p:truth)packet=flight.update(p.frame,p.time,{1280,720},p.pixel);
    require(recording::object(packet).at("status")==recording::quote("EXPERIMENTAL_FORECAST"),"Supported flight unavailable");
    packet=flight.update(10,.15,{1280,720},{});require(recording::object(packet).at("status")==recording::quote("EXPERIMENTAL_FORECAST"),"One missing observation prematurely lost");
    packet=flight.update(11,.2,{1280,720},{});require(recording::object(packet).at("status")==recording::quote("UNAVAILABLE"),"Missing observation predicted past horizon");
    flight.reset();for(const auto& p:truth)flight.update(p.frame,p.time,{1280,720},p.pixel);
    packet=flight.update(10,.15,{1280,720},trajectory(.15),true);require(recording::object(packet).at("status")==recording::quote("UNAVAILABLE"),"Contact bridged flights");
    flight.reset();for(const auto& p:truth)flight.update(p.frame,p.time,{1280,720},p.pixel);
    packet=flight.update(10,.15,{1280,720},cv::Point2f(1000,100));require(recording::object(packet).at("status")==recording::quote("UNAVAILABLE"),"Motion discontinuity bridged");
    packet=flight.update(10,.15,{1280,720},trajectory(.15));require(recording::object(packet).at("status")==recording::quote("UNAVAILABLE"),"Duplicate frame not reset");
    require(!ImagePlaneFlight::fit({{1,1,{1,1}},{2,1,{2,2}},{3,1,{3,3}},{4,1,{4,4}},{5,1,{5,5}}}),"Singular time window accepted");
    std::cout<<"PASS: quadratic forecast, bounded missing data, contact and discontinuity resets, source time validation\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
