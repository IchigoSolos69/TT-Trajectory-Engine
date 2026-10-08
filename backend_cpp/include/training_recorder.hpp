#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

// Clean source images, never detector annotations. A bounded worker keeps disk
// writes off the capture thread. Each JPEG remains usable after an interruption.
class TrainingRecorder {
public:
    struct Status { bool active=false, failed=false; int saved=0, dropped=0; std::string clip; };
    explicit TrainingRecorder(std::filesystem::path root, int intervalMs=200, int limitMs=60000)
        : root_(std::move(root)), intervalMs_(intervalMs), limitMs_(limitMs) {}
    ~TrainingRecorder() { stop(); }
    TrainingRecorder(const TrainingRecorder&)=delete;
    TrainingRecorder& operator=(const TrainingRecorder&)=delete;

    bool start(int camera, int width, int height, int burstFrames=1) {
        stop();
        std::lock_guard<std::mutex> lock(mutex_);
        status_={}; queue_.clear(); stopping_=false;
        try {
            if(burstFrames!=1&&burstFrames!=5)throw std::runtime_error("Unsupported capture burst length");
            burstFrames_=burstFrames;burstRemaining_=0;burstId_=0;previousFrame_=0;
            std::filesystem::create_directories(root_);
            const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            bool created=false;
            for(int attempt=0;attempt<100 && !created;++attempt) {
                status_.clip="live_"+std::to_string(stamp)+"_"+std::to_string(attempt);
                directory_=root_/status_.clip;
                created=std::filesystem::create_directory(directory_);
            }
            if(!created) throw std::runtime_error("Cannot allocate a unique recording directory");
            std::ofstream meta(directory_/"capture.json");
            meta << "{\"source\":\"live\",\"camera_index\":"<<camera
                 <<",\"width\":"<<width<<",\"height\":"<<height
                 <<",\"sample_interval_ms\":"<<intervalMs_<<",\"limit_ms\":"<<limitMs_
                 <<",\"burst_frames\":"<<burstFrames_<<",\"burst_interval_ms\":"<<(burstFrames_>1?1000:intervalMs_)
                 <<",\"contains_ground_truth\":false,\"frames\":\"frames.jsonl\"}\n";
            meta.close(); if(!meta) throw std::runtime_error("Cannot write capture metadata");
            started_=std::chrono::steady_clock::now(); nextMs_=0;
            status_.active=true;
            worker_=std::thread([this]{writeFrames();});
            return true;
        } catch(const std::exception& e) {
            status_.active=false; status_.failed=true;
            std::cerr<<"[TrainingRecorder] "<<e.what()<<std::endl;
            return false;
        }
    }
    void offer(const cv::Mat& frame, int sourceFrame, const std::string& reviewMetadata="") {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!status_.active) return;
        const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now()-started_).count();
        if(elapsed>=limitMs_) {status_.active=false;stopping_=true;ready_.notify_one();return;}
        if(burstRemaining_ && sourceFrame!=previousFrame_+1)burstRemaining_=0;
        if(!burstRemaining_){
            if(elapsed<nextMs_) return;
            nextMs_=elapsed+(burstFrames_>1?1000:intervalMs_);
            burstRemaining_=burstFrames_;++burstId_;
        }
        if(queue_.size()>=8) {++status_.dropped;burstRemaining_=0;return;}
        const int burstIndex=burstFrames_-burstRemaining_;
        queue_.push_back({frame.clone(),sourceFrame,elapsed,reviewMetadata,burstId_,burstIndex});
        previousFrame_=sourceFrame;--burstRemaining_;ready_.notify_one();
    }
    void expire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if(status_.active && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started_).count()>=limitMs_){
            status_.active=false;stopping_=true;ready_.notify_one();
        }
    }
    void stop() {
        {std::lock_guard<std::mutex> lock(mutex_);status_.active=false;stopping_=true;}
        ready_.notify_one();
        if(worker_.joinable()) worker_.join();
    }
    Status status() const {std::lock_guard<std::mutex> lock(mutex_);return status_;}
private:
    struct Item {cv::Mat frame;int sourceFrame;long long elapsedMs;std::string reviewMetadata;int burstId,burstIndex;};
    void writeFrames() {
        try {
            std::ofstream index(directory_/"frames.jsonl");
            if(!index) throw std::runtime_error("Cannot create frame index");
            int count=0;
            while(true) {
                Item item;
                {std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock,[this]{return stopping_ || !queue_.empty();});
                    if(queue_.empty()) break;
                    item=std::move(queue_.front());queue_.pop_front();}
                const auto name="frame_"+std::to_string(++count)+".jpg";
                if(!cv::imwrite((directory_/name).string(),item.frame,{cv::IMWRITE_JPEG_QUALITY,95}))
                    throw std::runtime_error("Cannot write training image");
                index<<"{\"file\":\""<<name<<"\",\"source_frame\":"<<item.sourceFrame
                     <<",\"elapsed_ms\":"<<item.elapsedMs<<",\"burst_id\":"<<item.burstId
                     <<",\"burst_index\":"<<item.burstIndex<<",\"burst_length\":"<<burstFrames_;
                if(!item.reviewMetadata.empty())index<<",\"review_metadata\":"<<item.reviewMetadata;
                index<<"}\n";
                index.flush(); if(!index) throw std::runtime_error("Cannot write frame index");
                {std::lock_guard<std::mutex> lock(mutex_);status_.saved=count;}
            }
        } catch(const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex_);
            status_.active=false;status_.failed=true;queue_.clear();
            std::cerr<<"[TrainingRecorder] "<<e.what()<<std::endl;
        }
    }
    std::filesystem::path root_,directory_;
    int intervalMs_,limitMs_,burstFrames_=1,burstRemaining_=0,burstId_=0,previousFrame_=0;long long nextMs_=0;
    std::chrono::steady_clock::time_point started_;
    mutable std::mutex mutex_;std::condition_variable ready_;
    std::deque<Item> queue_;std::thread worker_;bool stopping_=false;Status status_;
};
