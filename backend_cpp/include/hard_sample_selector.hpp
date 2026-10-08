#pragma once
#include "trajectory_intelligence.hpp"
#include <set>

class HardSampleSelector {
public:
    void reset(){representatives_.clear();lastTime_=-1;}
    std::string select(const cv::Mat& frame,const intelligence::TrajectorySample& s,const std::vector<Detection>& balls,const std::vector<Detection>& scene,bool calibrated,const std::string& ballHash,const std::string& sceneHash){
        if(lastTime_>=0&&s.timestamp-lastTime_<.2)return "";
        std::vector<std::string> reasons;
        if(s.state==intelligence::ObservationState::Rejected)reasons.push_back("PHYSICS_DISAGREEMENT");
        if(s.state==intelligence::ObservationState::Predicted)reasons.push_back("MISSING_EXPECTED_BALL");
        if(!balls.empty()&&balls.front().confidence<.4f)reasons.push_back("LOW_DETECTOR_SCORE");
        if(balls.size()>1&&balls[1].confidence>.25f)reasons.push_back("COMPETING_BALL_CANDIDATES");
        if(s.blurSeverity>.7f&&s.localContrast>0)reasons.push_back("LOW_ROI_SHARPNESS");
        if(!calibrated)reasons.push_back("CALIBRATION_UNAVAILABLE");
        for(const auto& d:scene)if(d.class_id==0&&d.confidence<.65f){reasons.push_back("PERSON_CANDIDATE_REVIEW");break;}
        if(reasons.empty())return "";
        cv::Mat gray,descriptor;cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);cv::resize(gray,descriptor,{16,9},0,0,cv::INTER_AREA);
        const int bx=balls.empty()?-1:int((balls.front().x+balls.front().w/2)*12),by=balls.empty()?-1:int((balls.front().y+balls.front().h/2)*7);
        std::string key=std::to_string(bx)+":"+std::to_string(by);for(const auto& reason:reasons)key+=":"+reason;
        for(const auto& representative:representatives_)if(representative.key==key&&cv::norm(descriptor,representative.image,cv::NORM_L1)/descriptor.total()<10)return "";
        representatives_.push_back({key,descriptor});if(representatives_.size()>128)representatives_.pop_front();lastTime_=s.timestamp;
        std::ostringstream out;out<<std::fixed<<std::setprecision(6)<<"{\"reviewed\":false,\"source_frame\":"<<s.frameId<<",\"source_time\":"<<s.timestamp<<",\"ball_model_sha256\":"<<std::quoted(ballHash)<<",\"scene_model_sha256\":"<<std::quoted(sceneHash)<<",\"reasons\":[";
        for(size_t i=0;i<reasons.size();++i){if(i)out<<',';out<<std::quoted(reasons[i]);}
        out<<"],\"selection\":\"PERCEPTUAL_DIVERSITY_AND_REASONS\",\"suggestions\":{";
        for(int cls=0;cls<3;++cls){if(cls)out<<',';out<<std::quoted(cls==0?"ball":cls==1?"person":"table")<<":[";
            bool first=true;const auto& boxes=cls==0?balls:scene;
            for(const auto& d:boxes){if(cls!=0&&d.class_id!=cls-1)continue;if(!first)out<<',';first=false;out<<"["<<d.x<<","<<d.y<<","<<d.w<<","<<d.h<<"]";}out<<"]";}
        out<<"}}";return out.str();
    }
private:
    struct Representative{std::string key;cv::Mat image;};std::deque<Representative> representatives_;double lastTime_=-1;
};
