#include "rally_intelligence.hpp"
#include "hard_sample_selector.hpp"
#include <iostream>
void require(bool b,const char* reason){if(!b)throw std::runtime_error(reason);}
int main()try{
    using namespace intelligence;
    std::vector<TrajectorySample> points;
    for(int i=0;i<5;++i){TrajectorySample s;s.timestamp=i*.01;s.table={.4f+i*.02f,.3f+i*.05f};s.covariance=cv::Matx22f::eye()*.001f;points.push_back(s);}
    auto fit=RallyIntelligence::fit(points);require(fit.has_value(),"motion fit missing");
    require(cv::norm(fit->first-cv::Point2f(2,5))<.001&&fit->second<.001,"weighted velocity fit differs from analytical motion");
    RallyIntelligence events;
    for(int i=0;i<9;++i){auto s=points[0];s.frameId=i+1;s.timestamp=i*.01;s.positionValid=s.accepted=true;s.state=ObservationState::Measured;s.segmentId=1;s.pixel={60,70};s.table={.5f,.05f+std::abs(i-4)*.03f};s.trackingConfidence=s.calibrationConfidence=.9f;events.update(s,{{0,.95f,0,0,.3f,1}}, {640,480});}
    require(events.events.size()==2&&events.events.back().type=="STRIKE","multi-cue synthetic reversal must create strike");
    require(events.events.back().status=="RULE_SUPPORTED","event must distinguish rule support from verified ground truth");
    require(events.events.back().playerSide==0,"near-end association incorrect");
    require(events.strikeCount==1&&std::abs(events.peakStrikeKmh-10.8f)<.01,"summary metrics must agree with the authoritative event graph");
    const auto count=events.events.size();events.seek();
    auto duplicate=points[0];duplicate.frameId=5;events.update(duplicate,{}, {640,480});require(events.events.size()==count,"seek must not duplicate graph");
    HardSampleSelector selector;cv::Mat image(480,640,CV_8UC3,cv::Scalar(30,30,30));TrajectorySample s;s.timestamp=0;s.frameId=1;
    auto first=selector.select(image,s,{}, {},false,"ballHash","sceneHash");require(!first.empty(),"uncalibrated frame should be selected");
    s.timestamp=.3;require(selector.select(image,s,{}, {},false,"ballHash","sceneHash").empty(),"duplicate hard scene must be suppressed");
    image.setTo(cv::Scalar(220,220,220));s.timestamp=.6;require(!selector.select(image,s,{}, {},false,"ballHash","sceneHash").empty(),"visually distinct hard example must survive");
    require(first.find("\"reviewed\":false")!=std::string::npos&&first.find("ballHash")!=std::string::npos,"review provenance must be retained");
    std::cout<<"PASS: weighted motion fits, multi-cue strike candidate, side association, seek deduplication, hard-example diversity and unreviewed provenance\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
