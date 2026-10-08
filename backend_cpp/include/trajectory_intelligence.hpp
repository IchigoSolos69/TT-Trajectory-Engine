#pragma once
#include "tensorrt_engine.hpp"
#include <deque>
#include <optional>
#include <array>
#include <iomanip>
#include <sstream>

namespace intelligence {
constexpr float tableWidth=1.525f,tableLength=2.74f;
enum class ObservationState { Measured, Predicted, Reacquired, Rejected, Lost };
inline const char* name(ObservationState s){switch(s){case ObservationState::Measured:return "MEASURED";case ObservationState::Predicted:return "PREDICTED";case ObservationState::Reacquired:return "REACQUIRED";case ObservationState::Rejected:return "REJECTED";default:return "LOST";}}
struct TrajectorySample {
    int frameId=0,segmentId=0,missedFrames=0;double timestamp=0;
    cv::Point2f pixel{},table{},velocity{};
    cv::Matx22f pixelCovariance=cv::Matx22f::eye(),covariance=cv::Matx22f::eye();
    std::optional<cv::Point2f> observation;
    float detectorConfidence=0,trackingConfidence=0,calibrationConfidence=0;
    float blurSeverity=0,localContrast=0,elongation=1,blurAngle=0,occlusionEvidence=0,physicsResidual=0;
    float ellipseMajor=0,ellipseMinor=0,ellipseAngle=0;
    bool positionValid=false,accepted=false;ObservationState state=ObservationState::Lost;
    std::string reason="NO_OBSERVATION";
};

// A tracked planar calibration. Corner ordering is aligned to the previous
// geometry before filtering; matrices themselves are never averaged.
class TableCalibration {
public:
    cv::Mat H;std::vector<cv::Point2f> observed,filtered;
    std::string state="SEARCHING";float quality=0,cornerResidualPx=0,cornerSigmaPx=3,borderSupport=0;
    double lastMeasured=-1;int stable=0,epoch=0;bool locked=false;
    void reset(){*this=TableCalibration{};}
    bool valid(double time)const{return !H.empty() && lastMeasured>=0 && time-lastMeasured<=.5;}
    static std::vector<cv::Point2f> align(const std::vector<cv::Point2f>& points,const std::vector<cv::Point2f>& reference){
        if(reference.size()!=4)return points;
        std::vector<cv::Point2f> best;double score=1e30;
        for(int flip:{1,-1})for(int offset=0;offset<4;++offset){std::vector<cv::Point2f> trial;double cost=0;
            for(int i=0;i<4;++i){auto p=points[(offset+flip*i+8)%4];trial.push_back(p);cost+=cv::norm(p-reference[i]);}
            if(cost<score){score=cost;best=trial;}}
        return best;
    }
    void update(const cv::Mat& frame,std::vector<cv::Point2f> corners,double time){
        const bool bounded=corners.size()==4 && std::all_of(corners.begin(),corners.end(),[&](const cv::Point2f& p){return std::isfinite(p.x)&&std::isfinite(p.y)&&p.x>=0&&p.y>=0&&p.x<frame.cols&&p.y<frame.rows;});
        if(!bounded || !cv::isContourConvex(corners) || std::abs(cv::contourArea(corners))<frame.total()*.015){
            if(!H.empty()){state="DEGRADED";quality*=.85f;locked=false;if(time-lastMeasured>.5){H.release();filtered.clear();stable=0;state="SEARCHING";}}
            return;
        }
        corners=align(corners,filtered);observed=corners;
        const float threshold=std::max(5.f,frame.cols*.006f);
        float displacement=0;if(filtered.size()==4)for(int i=0;i<4;++i)displacement+=float(cv::norm(corners[i]-filtered[i]))/4;
        if(filtered.size()==4 && displacement>threshold){
            auto aligned=align(corners,pending_);float shift=0;if(pending_.size()==4)for(int i=0;i<4;++i)shift+=float(cv::norm(aligned[i]-pending_[i]))/4;
            pending_=aligned;changes_=shift<threshold?changes_+1:1;state="REACQUIRING";quality*=.5f;locked=false;
            // Camera movement invalidates world coordinates immediately. Three
            // consistent candidates establish a new calibration epoch.
            H.release();if(changes_<3)return;
            filtered=corners;stable=0;++epoch;changes_=0;
        }else changes_=0;
        if(filtered.empty()){filtered=corners;stable=0;++epoch;state="CANDIDATE";
            horizontal_=cv::norm(corners[1]-corners[0])+cv::norm(corners[2]-corners[3])>cv::norm(corners[3]-corners[0])+cv::norm(corners[2]-corners[1]);}
        cornerResidualPx=displacement;
        cornerSigmaPx=std::max(1.5f,.9f*cornerSigmaPx+.1f*displacement);
        for(int i=0;i<4;++i)filtered[i]=filtered[i]*.85f+corners[i]*.15f;
        ++stable;lastMeasured=time;
        const std::vector<cv::Point2f> world=horizontal_?std::vector<cv::Point2f>{{0,0},{0,tableLength},{tableWidth,tableLength},{tableWidth,0}}:
            std::vector<cv::Point2f>{{0,0},{tableWidth,0},{tableWidth,tableLength},{0,tableLength}};
        H=cv::getPerspectiveTransform(filtered,world);
        // Independent image-edge support, not the misleading zero residual
        // obtained by reprojecting the four corners used to fit a homography.
        cv::Mat gray,edges,dist;cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);cv::Canny(gray,edges,70,160);cv::distanceTransform(255-edges,dist,cv::DIST_L2,3);
        int supported=0,total=0;
        for(int edge=0;edge<4;++edge)for(int k=1;k<20;++k){auto p=filtered[edge]*(1-k/20.f)+filtered[(edge+1)%4]*(k/20.f);
            int x=cvRound(p.x),y=cvRound(p.y);if(x>=0&&y>=0&&x<dist.cols&&y<dist.rows){++total;if(dist.at<float>(y,x)<4)++supported;}}
        borderSupport=total?float(supported)/total:0;
        quality=std::clamp(std::min(1.f,stable/8.f)*std::exp(-cornerResidualPx/threshold)*(.5f+.5f*borderSupport),0.f,1.f);
        locked=stable>=8&&quality>=.65f;state=locked?"LOCKED":"STABILIZING";
    }
    cv::Point2f project(cv::Point2f pixel)const{
        const double w=H.at<double>(2,0)*pixel.x+H.at<double>(2,1)*pixel.y+H.at<double>(2,2);
        if(std::abs(w)<1e-8)return {NAN,NAN};
        return {float((H.at<double>(0,0)*pixel.x+H.at<double>(0,1)*pixel.y+H.at<double>(0,2))/w),float((H.at<double>(1,0)*pixel.x+H.at<double>(1,1)*pixel.y+H.at<double>(1,2))/w)};
    }
    cv::Matx22f jacobian(cv::Point2f p)const{auto x=project(p+cv::Point2f(.5f,0))-project(p-cv::Point2f(.5f,0));auto y=project(p+cv::Point2f(0,.5f))-project(p-cv::Point2f(0,.5f));return {x.x,y.x,x.y,y.y};}
private:
    bool horizontal_=false;int changes_=0;std::vector<cv::Point2f> pending_;
};

struct RoiQuality {float blur=0,contrast=0,elongation=1,angle=0;};
inline RoiQuality qualityOf(const cv::Mat& frame,const Detection& d){
    RoiQuality q;cv::Rect r(int(d.x*frame.cols)-3,int(d.y*frame.rows)-3,int(d.w*frame.cols)+6,int(d.h*frame.rows)+6);r&=cv::Rect(0,0,frame.cols,frame.rows);
    if(r.width<3||r.height<3)return q;
    cv::Mat gray,lap,mask;cv::cvtColor(frame(r),gray,cv::COLOR_BGR2GRAY);cv::Laplacian(gray,lap,CV_32F);cv::Scalar mean,sd;cv::meanStdDev(lap,mean,sd);
    q.blur=1.f/(1.f+float(sd[0]*sd[0])/100.f);cv::meanStdDev(gray,mean,sd);q.contrast=float(sd[0]);
    cv::threshold(gray,mask,std::min(240.,mean[0]+sd[0]),255,cv::THRESH_BINARY);auto m=cv::moments(mask,true);
    if(m.m00>=5){float a=float(m.mu20/m.m00),b=float(m.mu11/m.m00),c=float(m.mu02/m.m00);float delta=std::sqrt((a-c)*(a-c)+4*b*b);
        q.elongation=std::sqrt(std::max(.1f,(a+c+delta)/2)/std::max(.1f,(a+c-delta)/2));q.angle=.5f*std::atan2(2*b,a-c);}
    return q;
}

class TrajectoryTracker {
public:
    explicit TrajectoryTracker(float accelerationSigma=4000.f,float measurementSigmaScale=1.f):accelerationSigma_(accelerationSigma),measurementSigmaScale_(measurementSigmaScale){
        if(!std::isfinite(accelerationSigma)||accelerationSigma<=0||!std::isfinite(measurementSigmaScale)||measurementSigmaScale<=0)throw std::invalid_argument("Invalid trajectory noise parameters");
    }
    float accelerationSigma()const{return accelerationSigma_;}
    float measurementSigmaScale()const{return measurementSigmaScale_;}
    std::deque<TrajectorySample> history;
    void reset(){initialized_=false;missed_=0;lastTime_=-1;lastFrame_=-1;history.clear();outsideCount_=0;++segment_;}
    TrajectorySample update(int frameId,double time,const cv::Mat& frame,const std::vector<Detection>& detections,const std::vector<Detection>& people,const TableCalibration& table){
        TrajectorySample s;s.frameId=frameId;s.timestamp=time;s.segmentId=segment_;s.calibrationConfidence=table.quality;
        const double dt=lastTime_<0?0:time-lastTime_;lastTime_=time;
        const int frameDelta=lastFrame_<0?1:std::max(1,frameId-lastFrame_);lastFrame_=frameId;
        if(dt<=0 || dt>.25){initialized_=false;missed_=0;++segment_;s.segmentId=segment_;}
        if(initialized_){
            const float t=float(dt),t2=t*t,t3=t2*t,t4=t2*t2;
            filter_.transitionMatrix=(cv::Mat_<float>(4,4)<<1,0,t,0,0,1,0,t,0,0,1,0,0,0,0,1);
            // White acceleration process model in image pixels. These priors
            // require held-out calibration before claiming probability coverage.
            const float q=accelerationSigma_*accelerationSigma_;
            filter_.processNoiseCov=(cv::Mat_<float>(4,4)<<t4/4*q,0,t3/2*q,0,0,t4/4*q,0,t3/2*q,t3/2*q,0,t2*q,0,0,t3/2*q,0,t2*q);
            filter_.predict();
        }
        const Detection* selected=nullptr;RoiQuality selectedQuality;float best=1e30,bestResidual=0;cv::Mat bestR;
        bool rejected=false,clusterReacquired=false;std::optional<cv::Point2f> rejectedPixel;
        for(const auto& d:detections){
            if(d.class_id!=0 || d.confidence<.1f)continue;
            cv::Point2f p((d.x+d.w*.5f)*frame.cols,(d.y+d.h*.5f)*frame.rows);
            if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<0||p.y<0||p.x>=frame.cols||p.y>=frame.rows)continue;
            auto q=qualityOf(frame,d);const float sigma=(1.5f+(1-d.confidence)*7+q.blur*4)*measurementSigmaScale_;
            cv::Mat R=cv::Mat::eye(2,2,CV_32F)*(sigma*sigma);float residual=0;
            if(initialized_){
                cv::Mat z=(cv::Mat_<float>(2,1)<<p.x,p.y);cv::Mat innovation=z-filter_.measurementMatrix*filter_.statePre;
                cv::Mat S=filter_.measurementMatrix*filter_.errorCovPre*filter_.measurementMatrix.t()+R;
                cv::Mat distance=innovation.t()*S.inv(cv::DECOMP_SVD)*innovation;residual=distance.at<float>(0);
                const auto last=cv::Point2f(filter_.statePost.at<float>(0),filter_.statePost.at<float>(1));
                const float pixelsPerSec=float(cv::norm(p-last)/std::max(dt,.001));
                if(residual>13.82f || pixelsPerSec>frame.cols*6.f){
                    rejected=true;rejectedPixel=p;bestResidual=std::max(bestResidual,residual);
                    // A constant-velocity gate must not lock out a real strike
                    // forever. Two strong, mutually coherent observations form
                    // a new segment; never bridge an unsupported velocity jump.
                    if(d.confidence>=.6f && !selected){
                        const bool coherent=outsideCount_>0 && time-outsideTime_>0 && time-outsideTime_<=.04 && cv::norm(p-outsidePixel_)<frame.cols*6*(time-outsideTime_);
                        if(coherent){selected=&d;selectedQuality=q;bestR=R;clusterReacquired=true;best=0;bestResidual=residual;}
                        else{outsidePixel_=p;outsideTime_=time;outsideCount_=1;}
                    }
                    continue;
                }
            }else if(d.confidence<.5f)continue;
            const float cost=initialized_?residual+2*(1-d.confidence):1-d.confidence;
            if(cost<best){selected=&d;selectedQuality=q;best=cost;bestR=R;bestResidual=residual;}
        }
        if(selected){
            cv::Point2f p((selected->x+selected->w*.5f)*frame.cols,(selected->y+selected->h*.5f)*frame.rows);
            const bool reacquired=clusterReacquired || (initialized_ && missed_>0);
            if(clusterReacquired)initialized_=false;
            if(!initialized_){
                filter_.init(4,2);filter_.measurementMatrix=(cv::Mat_<float>(2,4)<<1,0,0,0,0,1,0,0);
                filter_.statePost=(cv::Mat_<float>(4,1)<<p.x,p.y,0,0);filter_.errorCovPost=cv::Mat::eye(4,4,CV_32F);
                filter_.errorCovPost.at<float>(0,0)=filter_.errorCovPost.at<float>(1,1)=bestR.at<float>(0,0);
                filter_.errorCovPost.at<float>(2,2)=filter_.errorCovPost.at<float>(3,3)=1e6f;initialized_=true;++segment_;
            }else{filter_.measurementNoiseCov=bestR;filter_.correct((cv::Mat_<float>(2,1)<<p.x,p.y));}
            s.state=reacquired?ObservationState::Reacquired:ObservationState::Measured;s.accepted=true;s.observation=p;
            s.reason=selected->confidence<.5f?"ACCEPT_LOW_CONFIDENCE":selectedQuality.blur>.7f?"DOWNWEIGHT_BLUR":"ACCEPT_MEASURED";
            if(clusterReacquired)s.reason="REACQUIRE_COHERENT_OBSERVATIONS";
            s.detectorConfidence=selected->confidence;s.blurSeverity=selectedQuality.blur;s.localContrast=selectedQuality.contrast;s.elongation=selectedQuality.elongation;s.blurAngle=selectedQuality.angle;
            missed_=0;lastMeasured_=time;outsideCount_=0;
        }else if(initialized_){
            missed_+=frameDelta;s.state=rejected?ObservationState::Rejected:ObservationState::Predicted;
            s.reason=rejected?"REJECT_OUTSIDE_GATE":"PREDICT_MISSING_OBSERVATION";s.observation=rejectedPixel;
            if(missed_>8 || time-lastMeasured_>.15){initialized_=false;s.state=ObservationState::Lost;s.reason="PREDICTION_EXPIRED";++segment_;}
        }
        s.segmentId=segment_;s.missedFrames=missed_;s.physicsResidual=bestResidual;
        if(initialized_){
            s.pixel={filter_.statePost.at<float>(0),filter_.statePost.at<float>(1)};
            s.pixelCovariance={filter_.errorCovPost.at<float>(0,0),filter_.errorCovPost.at<float>(0,1),filter_.errorCovPost.at<float>(1,0),filter_.errorCovPost.at<float>(1,1)};
            s.trackingConfidence=std::exp(-std::sqrt(std::max(s.pixelCovariance(0,0),s.pixelCovariance(1,1)))/30.f);
            for(const auto& person:people)if(person.class_id==0 && cv::Rect2f(person.x*frame.cols,person.y*frame.rows,person.w*frame.cols,person.h*frame.rows).contains(s.pixel))s.occlusionEvidence=std::max(s.occlusionEvidence,person.confidence);
            if(!s.accepted && s.occlusionEvidence>.5f)s.reason=rejected?s.reason:"PREDICT_POSSIBLE_OCCLUSION";
            if(table.valid(time)){
                s.table=table.project(s.pixel);auto J=table.jacobian(s.pixel);
                const auto C=s.pixelCovariance+cv::Matx22f::eye()*(table.cornerSigmaPx*table.cornerSigmaPx);
                s.covariance=J*C*J.t();auto v=J*cv::Vec2f(filter_.statePost.at<float>(2),filter_.statePost.at<float>(3));s.velocity={v[0],v[1]};
                s.positionValid=std::isfinite(s.table.x)&&std::isfinite(s.table.y)&&std::abs(s.table.x)<10&&std::abs(s.table.y)<10&&std::isfinite(s.covariance(0,0))&&std::isfinite(s.covariance(1,1));
                const float a=s.covariance(0,0),b=s.covariance(0,1),c=s.covariance(1,1),delta=std::sqrt((a-c)*(a-c)+4*b*b);
                // Chi-square(2) 95% contour, explicitly model-based/unvalidated.
                s.ellipseMajor=std::sqrt(5.991f*std::max(0.f,(a+c+delta)/2));s.ellipseMinor=std::sqrt(5.991f*std::max(0.f,(a+c-delta)/2));s.ellipseAngle=.5f*std::atan2(2*b,a-c);
            }
        }
        history.push_back(s);while(history.size()>180)history.pop_front();return s;
    }
private:
    float accelerationSigma_,measurementSigmaScale_;
    cv::KalmanFilter filter_;bool initialized_=false;int missed_=0,segment_=0,outsideCount_=0,lastFrame_=-1;double lastTime_=-1,lastMeasured_=-1,outsideTime_=0;cv::Point2f outsidePixel_{};
};
inline void sampleJson(std::ostream& out,const TrajectorySample& s){
    out<<"{\"frame_id\":"<<s.frameId<<",\"timestamp\":"<<s.timestamp<<",\"segment_id\":"<<s.segmentId<<",\"state\":"<<std::quoted(name(s.state))<<",\"reason\":"<<std::quoted(s.reason)
       <<",\"accepted\":"<<(s.accepted?"true":"false")<<",\"pixel\":["<<s.pixel.x<<","<<s.pixel.y<<"],\"raw_observation\":";
    if(s.observation)out<<"["<<s.observation->x<<","<<s.observation->y<<"]";else out<<"null";
    out<<",\"table_position\":";if(s.positionValid)out<<"["<<s.table.x<<","<<s.table.y<<"]";else out<<"null";
    out<<",\"pixel_covariance\":["<<s.pixelCovariance(0,0)<<","<<s.pixelCovariance(0,1)<<","<<s.pixelCovariance(1,0)<<","<<s.pixelCovariance(1,1)<<"]";
    out<<",\"z\":null,\"velocity\":";if(s.positionValid)out<<"["<<s.velocity.x<<","<<s.velocity.y<<"]";else out<<"null";
    out<<",\"covariance\":";if(s.positionValid)out<<"["<<s.covariance(0,0)<<","<<s.covariance(0,1)<<","<<s.covariance(1,0)<<","<<s.covariance(1,1)<<"]";else out<<"null";
    out<<",\"ellipse\":";if(s.positionValid)out<<"["<<s.ellipseMajor<<","<<s.ellipseMinor<<","<<s.ellipseAngle<<"]";else out<<"null";
    out<<",\"detector_score\":"<<s.detectorConfidence<<",\"tracking_quality\":"<<s.trackingConfidence<<",\"calibration_quality\":"<<s.calibrationConfidence
       <<",\"blur_score\":"<<s.blurSeverity<<",\"contrast\":"<<s.localContrast<<",\"elongation\":"<<s.elongation<<",\"blur_angle_rad\":"<<s.blurAngle
       <<",\"occlusion_evidence\":"<<s.occlusionEvidence<<",\"mahalanobis_squared\":"<<s.physicsResidual<<",\"missed_frames\":"<<s.missedFrames<<"}";
}
inline std::string serialize(const TableCalibration& c,const TrajectoryTracker& t,double time,const std::string& ballHash,const std::string& sceneHash,size_t historyLimit=180){
    std::ostringstream out;out<<std::fixed<<std::setprecision(6);
    out<<"{\"schema_version\":1,\"provenance\":{\"position\":\"FILTERED_TABLE_PLANE_PROJECTION\",\"height\":\"UNOBSERVED\",\"covariance\":\"MODEL_BASED_NOT_COVERAGE_CALIBRATED\",\"quality\":\"HEURISTIC_NOT_PROBABILITY\",\"ball_model_sha256\":"<<std::quoted(ballHash)<<",\"scene_model_sha256\":"<<std::quoted(sceneHash)<<",\"acceleration_sigma_px_s2\":"<<t.accelerationSigma()<<",\"measurement_sigma_scale\":"<<t.measurementSigmaScale()<<"},\"calibration\":{\"state\":"<<std::quoted(c.state)<<",\"usable\":"<<(c.valid(time)?"true":"false")<<",\"quality\":"<<c.quality<<",\"corner_residual_px\":"<<c.cornerResidualPx<<",\"corner_sigma_px\":"<<c.cornerSigmaPx<<",\"border_support\":"<<c.borderSupport<<",\"age_s\":"<<(c.lastMeasured<0?0:std::max(0.,time-c.lastMeasured))<<",\"epoch\":"<<c.epoch<<",\"net_landmark\":\"UNOBSERVED\",\"center_line_landmark\":\"UNOBSERVED\",\"reprojection_error_px\":null},\"trajectory\":[";
    bool first=true;size_t skip=t.history.size()>historyLimit?t.history.size()-historyLimit:0;for(const auto& s:t.history){if(skip){--skip;continue;}if(!first)out<<',';first=false;sampleJson(out,s);}out<<"]}";return out.str();
}
} // namespace intelligence
