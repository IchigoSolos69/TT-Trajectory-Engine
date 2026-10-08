#include "training_recorder.hpp"
#include <cassert>

static void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
int main() try {
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root=std::filesystem::absolute("artifacts/recorder_test")/std::to_string(stamp);
    TrainingRecorder recorder(root,10,90);
    cv::Mat frame(120,160,CV_8UC3,cv::Scalar(30,100,200));
    recorder.offer(frame,1);
    require(!std::filesystem::exists(root),"Recording wrote data before explicit start");
    require(recorder.start(2,160,120),"Recording did not start");
    recorder.offer(frame,17,"{\"reviewed\":false,\"ball_model_sha256\":\"test-hash\"}");frame.setTo(cv::Scalar(0,0,0));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    recorder.offer(frame,18);recorder.stop();
    auto state=recorder.status();
    require(!state.active && !state.failed && state.saved==2,"Stop did not drain queued frames");
    const auto firstClip=state.clip;
    auto image=cv::imread((root/state.clip/"frame_1.jpg").string());
    require(!image.empty() && image.cols==160 && image.rows==120,"Saved resolution differs from source");
    auto mean=cv::mean(image);
    require(std::abs(mean[0]-30)<3 && std::abs(mean[2]-200)<3,"Source frame was not cloned before queueing");
    std::ifstream index(root/state.clip/"frames.jsonl");std::string line;int lines=0;
    while(std::getline(index,line)) {++lines;require(line.find("source_frame")!=std::string::npos,"Missing source index");if(lines==1)require(line.find("test-hash")!=std::string::npos,"Missing hard-example provenance");}
    require(lines==2,"Frame manifest differs from saved files");
    require(recorder.start(2,160,120),"Second clip failed");
    require(recorder.status().clip!=firstClip,"Recording overwrote an earlier clip");
    recorder.offer(frame,20);std::this_thread::sleep_for(std::chrono::milliseconds(110));recorder.offer(frame,21);
    require(!recorder.status().active,"Time limit did not stop recording");recorder.stop();
    require(recorder.status().saved==1,"Frame saved past duration limit");
    require(recorder.start(2,160,120),"Empty hard-example capture failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(110));recorder.expire();
    require(!recorder.status().active,"Capture must expire even when no difficult frames are selected");recorder.stop();
    require(recorder.status().saved==0,"Empty hard-example capture must not fabricate samples");
    TrainingRecorder bursts(root,200,2000);
    require(bursts.start(2,160,120,5),"Temporal burst capture failed");
    for(int fid=100;fid<106;++fid)bursts.offer(frame,fid);
    bursts.stop();
    require(bursts.status().saved==5,"Burst must save exactly five consecutive frames without sparse throttling");
    std::ifstream burstIndex(root/bursts.status().clip/"frames.jsonl");
    int burstRows=0;
    while(std::getline(burstIndex,line)){
        require(line.find("\"source_frame\":"+std::to_string(100+burstRows))!=std::string::npos,"Temporal burst source frames changed");
        require(line.find("\"burst_index\":"+std::to_string(burstRows))!=std::string::npos,"Burst ordering missing");
        require(line.find("\"burst_length\":5")!=std::string::npos,"Burst length missing");++burstRows;
    }
    require(burstRows==5,"Burst manifest incomplete");
    require(bursts.start(2,160,120,5),"Gap test capture failed");
    bursts.offer(frame,200);bursts.offer(frame,202);bursts.stop();
    require(bursts.status().saved==1,"A source-frame gap must not be represented as consecutive burst data");
    const auto blocked=root/"regular_file";{std::ofstream out(blocked);out<<"not a directory";}
    TrainingRecorder failing(blocked);
    require(!failing.start(0,160,120) && failing.status().failed,"Write failure was not surfaced");
    std::cout<<"PASS: explicit start, clean full-resolution frames, source timestamps, bounded duration, flush on stop, unique clips, write errors\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
