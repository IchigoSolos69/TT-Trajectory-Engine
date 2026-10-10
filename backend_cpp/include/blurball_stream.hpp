#pragma once
#include "blurball_engine.hpp"
#include <deque>
#include <optional>

// Source-frame policy is deliberately separate from the TensorRT model.
// Recordings collect every eligible window (two-frame lookahead); cameras only
// consume channel 2 of a past/present window. Neither policy pads missing frames.
class BlurBallStream {
public:
    struct Frame {
        cv::Mat image;
        int id=0;
        double timestamp=0;
        std::vector<BlurBallObservation> candidates;
        std::optional<BlurBallObservation> selected;
        float inferMs=0, gpuMs=0;
    };
    explicit BlurBallStream(BlurBallEngine& engine):engine_(engine){}
    void reset(){pending_.clear();history_.clear();anchor_.reset();decoded_=0;eof_=false;lastLiveId_=0;}
    bool nextRecording(cv::VideoCapture& capture,double fps,Frame& output){
        float workMs=0,gpuMs=0;
        while(pending_.size()<3&&!eof_){
            cv::Mat image;
            if(!capture.read(image)||image.empty()){eof_=true;break;}
            Frame f;f.image=image.clone();f.id=++decoded_;f.timestamp=(f.id-1)/fps;
            pending_.push_back(std::move(f));history_.push_back(image.clone());
            if(history_.size()>3)history_.pop_front();
            if(history_.size()==3){
                const auto result=engine_.infer({history_[0],history_[1],history_[2]});
                workMs+=result.wallMs;gpuMs+=result.gpuMs;
                for(int channel=0;channel<3;++channel){
                    const int sourceId=decoded_-2+channel;
                    for(auto& queued:pending_)if(queued.id==sourceId)
                        queued.candidates.insert(queued.candidates.end(),result.observations[channel].begin(),result.observations[channel].end());
                }
            }
        }
        if(pending_.empty())return false;
        output=std::move(pending_.front());pending_.pop_front();
        output.inferMs=workMs;output.gpuMs=gpuMs;select(output);return true;
    }
    Frame causal(const cv::Mat& image,int id,double timestamp){
        if(lastLiveId_&& (id!=lastLiveId_+1||(!history_.empty()&&history_.back().size()!=image.size())))reset();
        lastLiveId_=id;history_.push_back(image.clone());if(history_.size()>3)history_.pop_front();
        Frame output;output.image=image;output.id=id;output.timestamp=timestamp;
        if(history_.size()==3){const auto result=engine_.infer({history_[0],history_[1],history_[2]});output.candidates=result.observations[2];output.inferMs=result.wallMs;output.gpuMs=result.gpuMs;}
        select(output);return output;
    }
private:
    void select(Frame& frame){
        // Upstream's component-mass ranking and strict 300-source-pixel gate.
        // It is a heuristic, not calibrated uncertainty or a physical speed gate.
        for(const auto& candidate:frame.candidates){
            if(anchor_&&cv::norm(candidate.pixelCenter-*anchor_)>=300.f)continue;
            if(!frame.selected||candidate.componentMass>frame.selected->componentMass)frame.selected=candidate;
        }
        if(frame.selected)anchor_=frame.selected->pixelCenter;else anchor_.reset();
    }
    BlurBallEngine& engine_;
    std::deque<Frame> pending_;
    std::deque<cv::Mat> history_;
    std::optional<cv::Point2f> anchor_;
    int decoded_=0,lastLiveId_=0;bool eof_=false;
};
