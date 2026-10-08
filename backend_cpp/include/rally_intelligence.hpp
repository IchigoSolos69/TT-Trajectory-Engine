#pragma once
#include "trajectory_intelligence.hpp"

namespace intelligence {
struct RallyEvent {
    int id=0,frame=0,rally=0,previousId=0,playerSide=-1;
    double time=0,flightDuration=0;cv::Point2f position{},incoming{},outgoing{};
    float quality=0,headingChange=0,acceleration=0,fitResidual=0;
    bool positionValid=false;
    std::string type,status="CANDIDATE",reason;
};
class RallyIntelligence {
public:
    std::deque<RallyEvent> events;
    int strikeCount=0;float peakStrikeKmh=0;double observedDuration=0;
    void reset(){*this=RallyIntelligence{};}
    void seek(){window_.clear();active_=false;previousState_=ObservationState::Lost;}
    static std::optional<std::pair<cv::Point2f,float>> fit(const std::vector<TrajectorySample>& points){
        if(points.size()<3)return {};
        double total=0,meanT=0;cv::Point2d mean;
        const double origin=points.front().timestamp;
        for(const auto& p:points){double w=1/std::max(.00001f,p.covariance(0,0)+p.covariance(1,1));total+=w;meanT+=w*(p.timestamp-origin);mean+=cv::Point2d(p.table)*w;}
        meanT/=total;mean*=1/total;double denominator=0;cv::Point2d numerator;
        for(const auto& p:points){double w=1/std::max(.00001f,p.covariance(0,0)+p.covariance(1,1));double t=p.timestamp-origin-meanT;denominator+=w*t*t;numerator+=(cv::Point2d(p.table)-mean)*(w*t);}
        if(denominator<1e-9)return {};cv::Point2d v=numerator/denominator;double residual=0;
        for(const auto& p:points){auto e=cv::Point2d(p.table)-mean-v*(p.timestamp-origin-meanT);residual+=e.dot(e);}
        return std::make_pair(cv::Point2f(v),float(std::sqrt(residual/points.size())));
    }
    void update(const TrajectorySample& sample,const std::vector<Detection>& scene,cv::Size imageSize){
        if(sample.frameId<=furthestFrame_)return;
        furthestFrame_=sample.frameId;
        if(sample.state==ObservationState::Lost){
            if(previousState_!=ObservationState::Lost)add(sample,"BALL_LOST","OBSERVED_STATE","Bounded prediction expired");
            if(active_&&sample.timestamp-lastAccepted_>.5){add(sample,"RALLY_END","CANDIDATE","Ball unobserved for 0.5 seconds; point outcome unknown");active_=false;window_.clear();}
        }
        if(sample.state==ObservationState::Reacquired)add(sample,"BALL_REACQUIRED","OBSERVED_STATE",sample.reason);
        previousState_=sample.state;
        if(!sample.accepted||!sample.positionValid){window_.clear();return;}
        lastAccepted_=sample.timestamp;
        if(lastObserved_>=0&&sample.timestamp-lastObserved_<=.06)observedDuration+=sample.timestamp-lastObserved_;
        lastObserved_=sample.timestamp;
        if(!window_.empty()&&(window_.back().segmentId!=sample.segmentId||sample.timestamp-window_.back().timestamp>.06))window_.clear();
        window_.push_back(sample);while(window_.size()>9)window_.pop_front();
        if(window_.size()<9)return;
        if(!active_){active_=true;++rally_;add(window_.front(),"RALLY_START","CANDIDATE","Nine continuous accepted observations; serve unclassified");}
        const auto& pivot=window_[4];
        if(pivot.timestamp-lastEventTime_<.12)return;
        std::vector<TrajectorySample> incoming(window_.begin(),window_.begin()+4),outgoing(window_.begin()+5,window_.end());
        auto a=fit(incoming),b=fit(outgoing);if(!a||!b||a->second>.08||b->second>.08)return;
        const float inSpeed=float(cv::norm(a->first)),outSpeed=float(cv::norm(b->first));
        if(inSpeed<1||outSpeed<1||inSpeed>42||outSpeed>42)return;
        const float cosine=a->first.dot(b->first)/(inSpeed*outSpeed);
        bool personNear=false;
        for(const auto& d:scene)if(d.class_id==0&&d.confidence>=.6f){cv::Rect2f r(d.x*imageSize.width,d.y*imageSize.height,d.w*imageSize.width,d.h*imageSize.height);float margin=imageSize.width*.025f;r.x-=margin;r.y-=margin;r.width+=2*margin;r.height+=2*margin;if(r.contains(pivot.pixel))personNear=true;}
        const bool inTable=pivot.table.x>=0&&pivot.table.x<=tableWidth&&pivot.table.y>=0&&pivot.table.y<=tableLength;
        const float beforeVy=(window_[3].pixel.y-window_[0].pixel.y)/float(window_[3].timestamp-window_[0].timestamp);
        const float afterVy=(window_[8].pixel.y-window_[5].pixel.y)/float(window_[8].timestamp-window_[5].timestamp);
        std::string type,reason,status="CANDIDATE";int side=-1;
        if(cosine<-.2f&&personNear&&(pivot.table.y<.45f||pivot.table.y>tableLength-.45f)){
            type="STRIKE";reason="Separate velocity fits, heading reversal and person proximity";status="RULE_SUPPORTED";side=pivot.table.y<tableLength/2?0:1;
        }else if(inTable&&beforeVy>40&&afterVy<-20&&!personNear){
            type="TABLE_BOUNCE";reason="Image vertical reversal inside table; height/contact unobserved";
        }else if(std::abs(pivot.table.y-tableLength/2)<.12f&&outSpeed<inSpeed*.55f){type="NET_CONTACT";reason="Speed reduction near projected net; contact unobserved";}
        if(type.empty())return;
        auto& event=add(pivot,type,status,reason);event.incoming=a->first;event.outgoing=b->first;event.playerSide=side;
        event.headingChange=float(std::atan2(a->first.x*b->first.y-a->first.y*b->first.x,a->first.dot(b->first))*180/CV_PI);
        event.acceleration=float(cv::norm(b->first-a->first)/float(outgoing.front().timestamp-incoming.back().timestamp));
        event.fitResidual=std::max(a->second,b->second);lastEventTime_=pivot.timestamp;
        if(type=="STRIKE"){++strikeCount;peakStrikeKmh=std::max(peakStrikeKmh,outSpeed*3.6f);}
    }
    std::string json()const{
        std::ostringstream out;out<<std::fixed<<std::setprecision(6)<<"{\"status\":\"EXPERIMENTAL_RULES_NOT_EVENT_ACCURACY_VALIDATED\",\"events\":[";
        bool first=true;for(const auto& e:events){if(!first)out<<',';first=false;
            out<<"{\"id\":"<<e.id<<",\"previous_event_id\":"<<e.previousId<<",\"rally_id\":"<<e.rally<<",\"frame_id\":"<<e.frame<<",\"timestamp\":"<<e.time<<",\"type\":"<<std::quoted(e.type)<<",\"status\":"<<std::quoted(e.status)<<",\"reason\":"<<std::quoted(e.reason)<<",\"position\":";
            if(e.positionValid)out<<"["<<e.position.x<<","<<e.position.y<<"]";else out<<"null";
            out<<",\"height\":null,\"incoming_velocity\":["<<e.incoming.x<<","<<e.incoming.y<<"],\"outgoing_velocity\":["<<e.outgoing.x<<","<<e.outgoing.y<<"],\"incoming_speed_kmh\":"<<cv::norm(e.incoming)*3.6<<",\"outgoing_speed_kmh\":"<<cv::norm(e.outgoing)*3.6
               <<",\"heading_change_deg\":"<<e.headingChange<<",\"acceleration_estimate_m_s2\":"<<e.acceleration<<",\"fit_residual_m\":"<<e.fitResidual<<",\"quality\":"<<e.quality<<",\"player_side\":";
            if(e.playerSide<0)out<<"null";else out<<e.playerSide;
            out<<",\"spin_rpm\":null,\"reaction_time_ms\":null,\"flight_duration_s\":"<<e.flightDuration<<"}";
        }
        // Ambiguous contact candidates do not enter tactical aggregates.
        out<<"],\"tactics\":{\"eligible_bounces\":0,\"status\":\"AWAITING_VALIDATED_CONTACT_EVENTS\"},\"highlights\":[";
        first=true;for(const auto& e:events)if(e.type=="STRIKE"&&e.status=="RULE_SUPPORTED"&&cv::norm(e.outgoing)*3.6>=70){
            if(!first)out<<',';first=false;out<<"{\"event_id\":"<<e.id<<",\"start_s\":"<<std::max(0.,e.time-1.5)<<",\"end_s\":"<<e.time+1.5<<",\"reason\":\"HIGH_ESTIMATED_PLANAR_SPEED\",\"provenance\":\"RULE_BASED_CANDIDATE\"}";
        }out<<"]}";return out.str();
    }
private:
    RallyEvent& add(const TrajectorySample& s,const std::string& type,const std::string& status,const std::string& reason){
        RallyEvent e;e.id=++nextId_;e.rally=rally_;e.frame=s.frameId;e.time=s.timestamp;e.position=s.table;e.positionValid=s.positionValid;e.type=type;e.status=status;e.reason=reason;e.quality=s.trackingConfidence*s.calibrationConfidence;
        if(!events.empty()){e.previousId=events.back().id;e.flightDuration=std::max(0.,e.time-events.back().time);}
        events.push_back(e);if(events.size()>200)events.pop_front();return events.back();
    }
    std::deque<TrajectorySample> window_;ObservationState previousState_=ObservationState::Lost;
    bool active_=false;int rally_=0,nextId_=0,furthestFrame_=0;double lastAccepted_=0,lastEventTime_=-1,lastObserved_=-1;
};
} // namespace intelligence
