#include "blurball_stream.hpp"
#include "imageplane_flight.hpp"
#include "recording_controller.hpp"
#include <iostream>
#include <iomanip>

// Offline replay of the live detector policy: only channel 2 of three past/
// present source frames is eligible. No labels, future channels or camera input.
int main(int argc,char** argv) try {
    if(argc<4)throw std::runtime_error("Usage: eval_imageplane_forecast engine video new_output_directory [--limit N] [--overlay-from N --overlay-to N]");
    int limit=0,overlayFrom=0,overlayTo=0;
    for(int i=4;i<argc;++i){
        const std::string option=argv[i];
        if(i+1>=argc)throw std::runtime_error("Missing evaluation argument: "+option);
        const int value=std::stoi(argv[++i]);if(value<1)throw std::runtime_error("Evaluation frame arguments must be positive");
        if(option=="--limit")limit=value;
        else if(option=="--overlay-from")overlayFrom=value;
        else if(option=="--overlay-to")overlayTo=value;
        else throw std::runtime_error("Unknown evaluation argument: "+option);
    }
    if((overlayFrom==0)!=(overlayTo==0)||overlayTo<overlayFrom)throw std::runtime_error("Invalid overlay interval");
    const auto source=std::filesystem::u8path(argv[2]),directory=std::filesystem::u8path(argv[3]);
    if(std::filesystem::exists(directory))throw std::runtime_error("Refusing to overwrite evaluation output");
    cv::VideoCapture capture(source.u8string(),cv::CAP_FFMPEG);
    if(!capture.isOpened())throw std::runtime_error("Cannot decode the original source recording");
    const double fps=capture.get(cv::CAP_PROP_FPS);
    const int declared=int(capture.get(cv::CAP_PROP_FRAME_COUNT));
    if(!std::isfinite(fps)||fps<=0||declared<1)throw std::runtime_error("Source recording has no valid nominal FPS/frame count");
    const int expected=limit?std::min(limit,declared):declared;
    BlurBallEngine engine(argv[1]);BlurBallStream stream(engine);ImagePlaneFlight flight;
    const std::string model=recording::ballProvenance("blurball",engine.getSha256Hash(),true);
    const auto manifest=recording::json({{"schema_version","1"},{"source",recording::quote(source.u8string())},
        {"source_sha256",recording::quote(recording::sha256(source))},{"source_fps",std::to_string(fps)},
        {"source_declared_frames",std::to_string(declared)},{"total_frames",std::to_string(expected)},
        {"source_interval",recording::quote(limit?"CHRONOLOGICAL_PREFIX":"COMPLETE_RECORDING")},
        {"ball_model",model},{"ball_model_sha256",recording::quote(engine.getSha256Hash())},
        {"runtime_sha256",recording::quote(recording::sha256(std::filesystem::u8path(argv[0])))},
        {"experimental_intelligence","false"},{"contact_detection",recording::quote("UNKNOWN_NO_SCENE_OR_CONTACT_REFERENCE")},
        {"evaluation_policy",recording::quote("CAUSAL_SOURCE_REPLAY_NOT_LIVE_LATENCY")}});
    recording::JournalWriter journal(directory,manifest);
    int count=0,observed=0;double adapterMs=0,gpuMs=0;
    while(count<expected){
        cv::Mat image;if(!capture.read(image)||image.empty())break;
        const int frame=++count;const double time=(frame-1)/fps;
        const auto observation=stream.causal(image,frame,time);
        adapterMs+=observation.inferMs;gpuMs+=observation.gpuMs;
        std::optional<cv::Point2f> center;
        std::string ball="null",blur="null";
        if(observation.selected){
            ++observed;const auto& selected=*observation.selected;center=selected.pixelCenter;
            ball=recording::json({{"x",std::to_string(center->x/image.cols)},{"y",std::to_string(center->y/image.rows)},
                {"confidence",std::to_string(selected.heatmapPeak)}});
            blur=recording::json({{"center_px","["+std::to_string(center->x)+","+std::to_string(center->y)+"]"},
                {"component_mass",std::to_string(selected.componentMass)},{"heatmap_peak",std::to_string(selected.heatmapPeak)},
                {"extent_px",std::to_string(selected.blurExtentSourcePx)},{"axis_deg",std::to_string(selected.blurAxisDeg)}});
        }
        journal.append(frame,recording::json({{"frame_id",std::to_string(frame)},{"source_time",std::to_string(time)},
            {"source_width",std::to_string(image.cols)},{"source_height",std::to_string(image.rows)},
            {"ball",ball},{"ball_blur",blur},{"ball_model",model},
            {"ball_flight",flight.update(frame,time,image.size(),center)}}));
        if(overlayFrom&&frame>=overlayFrom&&frame<=overlayTo){
            cv::Mat overlay=image.clone();
            if(observation.selected){const auto& selected=*observation.selected;
                cv::line(overlay,selected.blurEndpoints[0],selected.blurEndpoints[1],cv::Scalar(200,100,230),2,cv::LINE_AA);
                cv::circle(overlay,selected.pixelCenter,10,cv::Scalar(0,200,255),2,cv::LINE_AA);
            }
            cv::putText(overlay,"CAUSAL FRAME "+std::to_string(frame)+(center?" OBSERVED":" NO ACCEPTED BALL"),
                {20,40},cv::FONT_HERSHEY_SIMPLEX,.7,cv::Scalar(0,200,255),2,cv::LINE_AA);
            if(!cv::imwrite((directory/("frame_"+std::to_string(frame)+".jpg")).u8string(),overlay))throw std::runtime_error("Cannot save source overlay");
        }
    }
    journal.finish(expected);
    if(count!=expected)throw std::runtime_error("Source decode ended early; partial journal retained for diagnostics");
    std::cout<<"{\"source_frames\":"<<count<<",\"observed_frames\":"<<observed
        <<",\"warmup_frames\":2,\"lookahead_frames\":0,\"offline_adapter_total_ms\":"<<adapterMs
        <<",\"offline_gpu_total_ms\":"<<gpuMs<<",\"live_latency_measured\":false}\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"Causal forecast evaluation failed: "<<error.what()<<'\n';return 1;}
