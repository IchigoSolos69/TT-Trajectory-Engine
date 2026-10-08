#include "trajectory_intelligence.hpp"
#include <iostream>
#include <stdexcept>
void require(bool b,const char* reason){if(!b)throw std::runtime_error(reason);}
Detection ball(float x,float y,float confidence=.9f){return {0,confidence,(x-3)/640,(y-3)/480,6.f/640,6.f/480};}
int main()try{
    using namespace intelligence;
    cv::Mat frame(480,640,CV_8UC3,cv::Scalar(20,20,20));
    std::vector<cv::Point2f> corners{{100,100},{400,100},{400,400},{100,400}};
    cv::rectangle(frame,{100,100,300,300},cv::Scalar(200,200,200),2);
    TableCalibration c;
    for(int i=0;i<10;++i)c.update(frame,corners,i*.02);
    require(c.locked&&c.valid(.2),"stable geometry must lock");
    const auto center=c.project({250,250});require(cv::norm(center-cv::Point2f(.7625,1.37))<.001,"table world dimensions must be conserved");
    auto rotated=corners;std::rotate(rotated.begin(),rotated.begin()+1,rotated.end());c.update(frame,rotated,.22);
    require(cv::norm(c.project({250,250})-center)<.001,"cyclic corner order must not flip table axes");
    c.update(frame,{},.24);require(c.valid(.24)&&c.state=="DEGRADED","brief dropout preserves geometry");
    c.update(frame,{},.9);require(!c.valid(.9),"stale geometry expires");
    c.reset();for(int i=0;i<10;++i)c.update(frame,corners,i*.01);
    auto invalid=corners;invalid[0].x=NAN;c.update(frame,invalid,.11);
    require(!c.locked&&c.state=="DEGRADED","invalid corners must degrade rather than retain LOCKED");
    c.update(frame,invalid,.7);require(!c.valid(.7),"invalid corners must not retain stale calibration");
    c.reset();for(int i=0;i<10;++i)c.update(frame,corners,i*.01);
    int epoch=c.epoch;auto moved=corners;for(auto& p:moved)p.x+=30;
    c.update(frame,moved,.1);require(!c.valid(.1)&&c.state=="REACQUIRING","camera movement invalidates geometry immediately");
    c.update(frame,moved,.11);c.update(frame,moved,.12);require(c.valid(.12)&&c.epoch>epoch,"consistent new pose creates epoch");
    c.reset();for(int i=0;i<10;++i)c.update(frame,corners,i*.01);
    TrajectoryTracker tracker;TrajectorySample s;
    for(int i=0;i<12;++i)s=tracker.update(i,i*.01,frame,{ball(200+i*2,200)},{},c);
    require(s.accepted&&s.positionValid,"measured ball must project");
    auto initialVariance=s.pixelCovariance(0,0);
    s=tracker.update(12,.12,frame,{}, {},c);require(s.state==ObservationState::Predicted,"missing ball becomes predicted");
    require(s.pixelCovariance(0,0)>initialVariance,"uncertainty must grow during prediction");
    auto missedVariance=s.pixelCovariance(0,0);
    s=tracker.update(13,.13,frame,{ball(226,200)}, {},c);require(s.state==ObservationState::Reacquired,"real observation after gap becomes reacquired");
    require(s.pixelCovariance(0,0)<missedVariance,"measurement should reduce uncertainty");
    s=tracker.update(14,.14,frame,{ball(590,40)}, {},c);require(s.state==ObservationState::Rejected&&!s.accepted,"teleport must be rejected");
    require(s.pixel.x<300,"rejected candidate must not move trajectory to outlier");
    for(int i=15;i<25;++i)s=tracker.update(i,i*.01,frame,{}, {},c);
    require(s.state==ObservationState::Lost&&!s.positionValid,"prediction must expire");
    tracker.reset();tracker.update(1,0,frame,{ball(200,200)}, {},c);
    s=tracker.update(11,.08,frame,{}, {},c);
    require(s.state==ObservationState::Lost,"source-frame skips count toward the prediction limit");
    tracker.reset();for(int i=0;i<10;++i)tracker.update(i,i*.01,frame,{ball(200+i*2,200)}, {},c);
    s=tracker.update(10,.1,frame,{ball(400,200)}, {},c);const int oldSegment=s.segmentId;
    require(s.state==ObservationState::Rejected,"first large residual cannot reset the track");
    s=tracker.update(11,.11,frame,{ball(402,200)}, {},c);
    require(s.state==ObservationState::Reacquired&&s.segmentId!=oldSegment,"coherent observations recover into a new segment without bridging a jump");
    tracker.reset();s=tracker.update(1,0,frame,{ball(200,200,.2)}, {},c);require(!s.accepted,"weak detection cannot initialize a ghost track");
    auto json=serialize(c,tracker,.3,"ball-hash","scene-hash");require(json.find("MODEL_BASED_NOT_COVERAGE_CALIBRATED")!=std::string::npos,"uncertainty provenance required");
    std::cout<<"PASS: calibration lock/dropout/movement/order, known dimensions, measured/predicted/reacquired/rejected/lost, covariance growth/contraction, bounded prediction, provenance\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
