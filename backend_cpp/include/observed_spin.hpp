#pragma once
#include <opencv2/opencv.hpp>
#include <deque>
#include <algorithm>

struct ObservedSpin {bool available=false;float rpm=0,quality=0;std::string reason="Ball too small or texture unresolved";};

// Measures apparent surface rotation about the camera axis. It cannot recover
// full 3D spin or disambiguate rotations hidden between exposures.
class ObservedSpinEstimator {
public:
    void reset(){previous_.release();rates_.clear();}
    ObservedSpin update(const cv::Mat& frame,cv::Rect box,float dt) {
        ObservedSpin result;
        box &= cv::Rect(0,0,frame.cols,frame.rows);
        if(box.width<24 || box.height<24 || dt<=0 || dt>.1f){reset();return result;}
        cv::Mat hsv,white;
        cv::cvtColor(frame(box),hsv,cv::COLOR_BGR2HSV);
        cv::inRange(hsv,cv::Scalar(0,0,140),cv::Scalar(180,95,255),white);
        cv::morphologyEx(white,white,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,{3,3}));
        std::vector<std::vector<cv::Point>> contours;cv::findContours(white,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        cv::Rect sphere;double best=0;
        for(const auto& contour:contours){
            auto r=cv::boundingRect(contour);double area=cv::contourArea(contour),perimeter=cv::arcLength(contour,true);
            if(r.width<24 || r.height<24 || double(r.width)/r.height<.8 || double(r.width)/r.height>1.25)continue;
            if(area<r.area()*.55 || perimeter<=0 || 4*CV_PI*area/(perimeter*perimeter)<.65)continue;
            if(!r.contains(cv::Point(box.width/2,box.height/2)))continue;
            if(area>best){best=area;sphere=r;}
        }
        if(best==0){reset();return result;}
        sphere.x+=box.x;sphere.y+=box.y;
        cv::Mat crop;cv::cvtColor(frame(sphere),crop,cv::COLOR_BGR2GRAY);cv::resize(crop,crop,{64,64});
        cv::Mat mask=cv::Mat::zeros(64,64,CV_8U);cv::circle(mask,{32,32},23,cv::Scalar(255),-1);
        if(previous_.empty()){previous_=crop;result.reason="Collecting surface observations";return result;}
        std::vector<cv::Point2f> before,after;cv::goodFeaturesToTrack(previous_,before,32,.08,5,mask);
        if(before.size()<6){previous_=crop;rates_.clear();result.reason="No resolvable surface markings";return result;}
        std::vector<unsigned char> status;std::vector<float> error;
        cv::calcOpticalFlowPyrLK(previous_,crop,before,after,status,error,{15,15},2);
        std::vector<cv::Point2f> a,b;
        for(size_t i=0;i<before.size();++i)if(status[i] && error[i]<15 && cv::norm(after[i]-cv::Point2f(32,32))<25){a.push_back(before[i]);b.push_back(after[i]);}
        previous_=crop;
        if(a.size()<6){rates_.clear();result.reason="Surface tracking uncertain";return result;}
        cv::Mat inliers;auto affine=cv::estimateAffinePartial2D(a,b,inliers,cv::RANSAC,1.5);
        if(affine.empty()){rates_.clear();return result;}
        const float fraction=float(cv::countNonZero(inliers))/a.size();
        const double scale=std::hypot(affine.at<double>(0,0),affine.at<double>(1,0));
        const double angle=std::atan2(affine.at<double>(1,0),affine.at<double>(0,0));
        if(fraction<.8f || scale<.95 || scale>1.05 || std::abs(angle)>CV_PI/4){rates_.clear();result.reason="Rotation ambiguous between frames";return result;}
        rates_.push_back(float(angle*60/(2*CV_PI*dt)));if(rates_.size()>5)rates_.pop_front();
        if(rates_.size()<3){result.reason="Collecting surface observations";return result;}
        auto sorted=std::vector<float>(rates_.begin(),rates_.end());std::sort(sorted.begin(),sorted.end());
        const float median=sorted[sorted.size()/2];
        if(sorted.back()-sorted.front()>std::max(30.f,std::abs(median)*.5f)){result.reason="Rotation estimate inconsistent";return result;}
        result.available=true;result.rpm=std::abs(median);result.quality=fraction;
        result.reason="Camera-axis component only; fast rotation may alias";return result;
    }
private:
    cv::Mat previous_;std::deque<float> rates_;
};
