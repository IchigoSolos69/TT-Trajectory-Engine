#pragma once
#include <opencv2/opencv.hpp>
#include <deque>
#include <optional>
#include <vector>
#include <cmath>
#include <cstdint>

// Candidate-only, causal ROI policy. An inferred recovery never extends the
// detector anchor lifetime. No ground-truth position enters this class.
class TemporalRecoveryWindow {
public:
    struct Request {
        cv::Point2f predicted;
        cv::Point origin;
        int missedFrames;
        std::vector<float> input;
    };
    std::optional<Request> update(uint64_t fid, double time, const cv::Mat& frame,
                                std::optional<cv::Point2f> detector) {
        if(frame.empty() || frame.type()!=CV_8UC3 || !std::isfinite(time))
            throw std::runtime_error("Invalid temporal source frame");
        if(!frames_.empty() && (fid!=lastFrame_+1 || time<=lastTime_ ||
            time-lastTime_>.05 || frame.size()!=frames_.back().size())) reset();
        lastFrame_=fid;lastTime_=time;
        frames_.push_back(frame.clone());if(frames_.size()>5)frames_.pop_front();
        if(detector && (!std::isfinite(detector->x)||!std::isfinite(detector->y)||
            detector->x<0||detector->y<0||detector->x>=frame.cols||detector->y>=frame.rows))
            detector.reset();
        if(detector){
            previous_=latest_;latest_=Anchor{fid,time,*detector};return std::nullopt;
        }
        if(!latest_ || !previous_ || frames_.size()!=5) return std::nullopt;
        auto missed=fid-latest_->frame;
        const double dt=latest_->time-previous_->time,age=time-latest_->time;
        if(missed>4 || age>.05 || dt<=0 || dt>.05) return std::nullopt;
        cv::Point2f velocity=(latest_->pixel-previous_->pixel)/float(dt);
        if(cv::norm(velocity)>6*frame.cols) return std::nullopt;
        const auto predicted=latest_->pixel+velocity*float(age);
        if(predicted.x<0||predicted.y<0||predicted.x>=frame.cols||predicted.y>=frame.rows)
            return std::nullopt;
        // Match training: integer truncation, grey border, bilinear RGB resize,
        // temporal-major contiguous CHW, oldest to current.
        cv::Point origin(int(predicted.x-96),int(predicted.y-96));
        std::vector<float> input(15*128*128);
        int t=0;
        for(const auto& image:frames_){
            cv::Mat padded(192,192,CV_8UC3,cv::Scalar(114,114,114));
            const cv::Rect source=cv::Rect(origin.x,origin.y,192,192)&cv::Rect(0,0,image.cols,image.rows);
            if(source.area()>0) image(source).copyTo(padded(cv::Rect(source.x-origin.x,source.y-origin.y,source.width,source.height)));
            cv::Mat resized;cv::resize(padded,resized,{128,128});
            for(int y=0;y<128;++y)for(int x=0;x<128;++x){
                auto pixel=resized.at<cv::Vec3b>(y,x);
                for(int c=0;c<3;++c)input[(t*3+c)*128*128+y*128+x]=pixel[2-c]/255.f;
            }++t;
        }
        return Request{predicted,origin,int(missed),std::move(input)};
    }
    void reset(){frames_.clear();latest_.reset();previous_.reset();lastFrame_=0;lastTime_=0;}
private:
    struct Anchor{uint64_t frame;double time;cv::Point2f pixel;};
    std::deque<cv::Mat> frames_;
    std::optional<Anchor> latest_,previous_;
    uint64_t lastFrame_=0;double lastTime_=0;
};
