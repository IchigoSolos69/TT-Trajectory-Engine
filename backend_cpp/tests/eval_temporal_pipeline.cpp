#include "temporal_recovery.hpp"
#include "temporal_recovery_window.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

int main(int argc,char** argv)try{
    if(argc<5||argc>6||(argc==6&&std::string(argv[5])!="--requests-only"))
        throw std::runtime_error("Usage: predictions.csv video.mp4 temporal.engine output.csv [--requests-only]");
    const bool requestsOnly=argc==6;
    std::ifstream predictions(argv[1]);std::ofstream output(argv[4]);
    if(!predictions||!output)throw std::runtime_error("Cannot open temporal evaluation files");
    cv::VideoCapture video(argv[2],cv::CAP_FFMPEG);
    if(!video.isOpened())throw std::runtime_error("Cannot open temporal evaluation video");
    double fps=video.get(cv::CAP_PROP_FPS);if(!std::isfinite(fps)||fps<=0)throw std::runtime_error("Invalid video FPS");
    std::unique_ptr<TemporalRecoveryEngine> engine;
    if(!requestsOnly)engine=std::make_unique<TemporalRecoveryEngine>(argv[3]);
    TemporalRecoveryWindow window;
    std::string line;std::getline(predictions,line);cv::Mat image;int current=0,rows=0,calls=0,accepted=0;
    output<<"frame,baseline_x,baseline_y,baseline_score,recovery_x,recovery_y,recovery_score,recovery_accepted,missed_frames,roi_x,roi_y,gpu_ms\n"<<std::setprecision(9);
    while(std::getline(predictions,line)){
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        int fid,count;float x,y,score;
        if(!(row>>fid>>x>>y>>score>>count)||fid<1||!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(score))
            throw std::runtime_error("Invalid detector CSV row");
        if(rows==0&&fid>1){if(!video.set(cv::CAP_PROP_POS_FRAMES,fid-1))throw std::runtime_error("Video seek failed");current=fid-1;}
        if(fid<=current)throw std::runtime_error("Detector frame IDs must increase");
        while(current<fid){if(!video.read(image))throw std::runtime_error("Video ended early");++current;}
        std::optional<cv::Point2f> detection;
        if(score>=.25f&&x>=0&&y>=0&&x<image.cols&&y<image.rows)detection=cv::Point2f(x,y);
        auto request=window.update(fid,(fid-1)/fps,image,detection);
        float rx=-1,ry=-1,rs=0,gpuMs=0;bool use=false;
        if(request && engine){
            auto result=engine->infer(request->input);++calls;gpuMs=result.gpuMs;
            const auto peak=std::max_element(result.logits.begin(),result.logits.begin()+1024)-result.logits.begin();
            rx=request->origin.x+(float(peak%32)+.5f)*6;
            ry=request->origin.y+(float(peak/32)+.5f)*6;
            rs=1.f/(1.f+std::exp(-result.logits.back()));
            use=rs>=.95f&&rx>=0&&ry>=0&&rx<image.cols&&ry<image.rows&&
                cv::norm(cv::Point2f(rx,ry)-request->predicted)<=96;
            if(use)++accepted;
        }
        output<<fid<<','<<(detection?x:-1)<<','<<(detection?y:-1)<<','<<score<<','<<rx<<','<<ry<<','<<rs<<','<<use<<','
              <<(request?request->missedFrames:0)<<','<<(request?request->origin.x:0)<<','<<(request?request->origin.y:0)<<','<<gpuMs<<'\n';
        if(++rows%2000==0)std::cerr<<rows<<" source frames evaluated\n";
    }
    if(rows==0||!output.good())throw std::runtime_error("Empty or failed evaluation output");
    std::cerr<<"Completed "<<rows<<" source frames, "<<calls<<" native recovery calls, "<<accepted<<" accepted hypotheses; no ground truth used\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
