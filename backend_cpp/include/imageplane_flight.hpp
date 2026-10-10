#pragma once
#include "analysis_journal.hpp"
#include <opencv2/core.hpp>
#include <deque>
#include <optional>
#include <cmath>

// A research baseline for neural forecasting, not reconstructed 3D physics.
// Only supplied observations enter the fit. Forecasts never become detections.
class ImagePlaneFlight {
public:
    struct Point {int frame;double time;cv::Point2f pixel;};
    struct Fit {cv::Vec3d x{},y{};double origin=0,scale=1,residual=0;};
    static constexpr double horizon=.050;
    void reset(const std::string& reason="SOURCE_RESET"){points_.clear();lastFrame_=0;lastTime_=-1;reason_=reason;++segment_;}
    static std::optional<Fit> fit(const std::deque<Point>& points){
        if(points.size()<5)return {};
        Fit result;result.origin=points.back().time;result.scale=result.origin-points.front().time;
        if(result.scale<.04)return {};
        cv::Mat design(int(points.size()),3,CV_64F),values(int(points.size()),2,CV_64F);
        for(int i=0;i<int(points.size());++i){const double t=(points[i].time-result.origin)/result.scale;
            design.at<double>(i,0)=1;design.at<double>(i,1)=t;design.at<double>(i,2)=t*t;
            values.at<double>(i,0)=points[i].pixel.x;values.at<double>(i,1)=points[i].pixel.y;}
        cv::Mat coefficients;if(!cv::solve(design,values,coefficients,cv::DECOMP_SVD))return {};
        for(int j=0;j<3;++j){result.x[j]=coefficients.at<double>(j,0);result.y[j]=coefficients.at<double>(j,1);}
        const auto error=design*coefficients-values;result.residual=cv::norm(error,cv::NORM_L2)/std::sqrt(double(points.size()));
        if(!std::isfinite(result.residual))return {};
        return result;
    }
    static cv::Point2d position(const Fit& fit,double time){
        const double t=(time-fit.origin)/fit.scale;const cv::Vec3d basis(1,t,t*t);
        return {fit.x.dot(basis),fit.y.dot(basis)};
    }
    std::string update(int frame,double time,cv::Size size,std::optional<cv::Point2f> measured,bool contactCandidate=false){
        if(frame<1||!std::isfinite(time)||size.width<1||size.height<1)throw std::runtime_error("Invalid flight frame, time or image size");
        if(frame<=lastFrame_||time<=lastTime_|| (lastTime_>=0&&time-lastTime_>.075))reset("SOURCE_TIME_DISCONTINUITY");
        if(contactCandidate)reset("CONTACT_CANDIDATE");
        lastFrame_=frame;lastTime_=time;
        const double gate=std::max(6.,std::hypot(size.width,size.height)*.004);
        if(measured){
            if(!std::isfinite(measured->x)||!std::isfinite(measured->y))throw std::runtime_error("Nonfinite flight observation");
            if(auto previous=fit(points_);previous&&cv::norm(position(*previous,time)-cv::Point2d(*measured))>gate){
                points_.clear();++segment_;reason_="IMAGE_MOTION_DISCONTINUITY";
            }
            points_.push_back({frame,time,*measured});while(points_.size()>9)points_.pop_front();
        }
        if(!points_.empty()&&time-points_.back().time>horizon){points_.clear();++segment_;reason_="OBSERVATION_EXPIRED";}
        auto estimate=fit(points_);if(estimate&&estimate->residual>gate){estimate.reset();reason_="FIT_RESIDUAL_TOO_LARGE";}
        std::string support="[",forecasts="[";bool first=true;
        for(const auto& p:points_){if(!first)support+=",";first=false;support+=std::to_string(p.frame);}support+="]";
        if(estimate){
            for(int i=0;i<3;++i){const double t=time+horizon*i/2;
                if(t-points_.back().time>horizon+1e-9)break;
                const auto pixel=position(*estimate,t);if(!std::isfinite(pixel.x)||!std::isfinite(pixel.y))break;
                if(forecasts.size()>1)forecasts+=",";
                forecasts+=recording::json({{"timestamp",std::to_string(t)},{"pixel","["+std::to_string(pixel.x)+","+std::to_string(pixel.y)+"]"}});
            }
        }
        forecasts+="]";
        return recording::json({{"status",recording::quote(estimate?"EXPERIMENTAL_FORECAST":"UNAVAILABLE")},
            {"reason",recording::quote(estimate?(measured?"OBSERVED_FLIGHT_FIT":"BOUNDED_MISSING_OBSERVATION"):reason_)},
            {"method",recording::quote("recent_observations_quadratic_v1")},{"coordinate_basis",recording::quote("source_pixels")},
            {"reference_frame",std::to_string(frame)},{"reference_timestamp",std::to_string(time)},{"segment_id",std::to_string(segment_)},
            {"support_frames",support},{"maximum_horizon_s","0.05"},{"fit_rmse_px",estimate?std::to_string(estimate->residual):"null"},
            {"fit_origin",estimate?std::to_string(estimate->origin):"null"},{"fit_time_scale",estimate?std::to_string(estimate->scale):"null"},
            {"fit_coefficients",estimate?"[["+std::to_string(estimate->x[0])+","+std::to_string(estimate->x[1])+","+std::to_string(estimate->x[2])+"],["+std::to_string(estimate->y[0])+","+std::to_string(estimate->y[1])+","+std::to_string(estimate->y[2])+"]]":"null"},
            {"uncertainty","null"},{"uncertainty_reason",recording::quote("FORECAST_COVERAGE_NOT_CALIBRATED")},{"points",forecasts}});
    }
private:
    std::deque<Point> points_;int lastFrame_=0,segment_=0;double lastTime_=-1;std::string reason_="INSUFFICIENT_CONTINUOUS_OBSERVATIONS";
};
