#pragma once
#include "tensorrt_engine.hpp"

// Refine the learned table ROI to a visible tabletop contour. A rectangle or
// default trapezoid is never substituted when the contour is ambiguous.
inline std::vector<cv::Point2f> findTableSurface(const cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& d : detections) {
        if (d.class_id != 1 || d.confidence < .6f) continue;
        cv::Rect roi(static_cast<int>(d.x*frame.cols), static_cast<int>(d.y*frame.rows),
                     static_cast<int>(d.w*frame.cols), static_cast<int>(d.h*frame.rows));
        roi &= cv::Rect(0,0,frame.cols,frame.rows);
        if (roi.area() < frame.total() * .02) continue;
        cv::Mat hsv, mask;
        cv::cvtColor(frame(roi),hsv,cv::COLOR_BGR2HSV);
        cv::inRange(hsv,cv::Scalar(35,65,25),cv::Scalar(140,255,255),mask);
        cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT,{7,7}));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        std::sort(contours.begin(),contours.end(),[](const auto& a,const auto& b){return cv::contourArea(a)>cv::contourArea(b);});
        for (const auto& contour: contours) {
            std::vector<cv::Point> hull, quad;
            cv::convexHull(contour,hull);
            cv::approxPolyDP(hull,quad,.025*cv::arcLength(hull,true),true);
            if (quad.size()!=4 || !cv::isContourConvex(quad) || cv::contourArea(quad)<roi.area()*.4) continue;
            std::vector<cv::Point2f> points;
            for (const auto& p:quad) points.emplace_back(static_cast<float>(p.x+roi.x),static_cast<float>(p.y+roi.y));
            cv::Point2f center(0,0); for(const auto& p:points) center+=p; center*=.25f;
            std::sort(points.begin(),points.end(),[&](const auto&a,const auto&b){return std::atan2(a.y-center.y,a.x-center.x)<std::atan2(b.y-center.y,b.x-center.x);});
            auto tl=std::min_element(points.begin(),points.end(),[](const auto&a,const auto&b){return a.x+a.y<b.x+b.y;});
            std::rotate(points.begin(),tl,points.end());
            return points;
        }
    }
    return {};
}
