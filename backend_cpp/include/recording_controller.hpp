#pragma once
#include "analysis_journal.hpp"
#include "telemetry_server.hpp"
#include "video_picker.hpp"
#include <future>
#include <thread>
#include <chrono>
#include <algorithm>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>

namespace recording {
inline std::string sha256(const std::filesystem::path& path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot hash source: "+path.u8string());
    HCRYPTPROV provider=0;HCRYPTHASH hash=0;
    if(!CryptAcquireContextW(&provider,nullptr,nullptr,PROV_RSA_AES,CRYPT_VERIFYCONTEXT))throw std::runtime_error("Cannot initialize source hash provider");
    if(!CryptCreateHash(provider,CALG_SHA_256,0,0,&hash)){CryptReleaseContext(provider,0);throw std::runtime_error("Cannot initialize source hash");}
    std::vector<char> buffer(1024*1024);bool ok=true;
    while(in){in.read(buffer.data(),buffer.size());if(in.gcount()&&!CryptHashData(hash,reinterpret_cast<const BYTE*>(buffer.data()),DWORD(in.gcount()),0)){ok=false;break;}}
    BYTE digest[32];DWORD size=32;ok=ok&&in.eof()&&CryptGetHashParam(hash,HP_HASHVAL,digest,&size,0);
    CryptDestroyHash(hash);CryptReleaseContext(provider,0);if(!ok)throw std::runtime_error("Source hashing failed");
    std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);return out.str();
}
inline std::wstring commandArgument(const std::wstring& value){
    std::wstring result=L"\"";int slashes=0;
    for(wchar_t c:value){if(c==L'\\'){++slashes;continue;}if(c==L'"'){result.append(slashes*2+1,L'\\');result+=c;}else{result.append(slashes,L'\\');result+=c;}slashes=0;}
    result.append(slashes*2,L'\\');return result+L"\"";
}
class AnalysisChild {
public:
    ~AnalysisChild(){stop();if(job_)CloseHandle(job_);}
    void start(const std::vector<std::wstring>& arguments,const std::filesystem::path& log){
        stop();wchar_t executable[32768];const DWORD length=GetModuleFileNameW(nullptr,executable,32768);
        if(!length||length>=32768)throw std::runtime_error("Cannot resolve analysis executable");
        std::wstring command=commandArgument(executable);for(const auto& arg:arguments)command+=L" "+commandArgument(arg);
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
        HANDLE output=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&security,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(output==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot create analysis worker log");
        STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES;
        startup.hStdOutput=output;startup.hStdError=output;startup.hStdInput=nullptr;
        PROCESS_INFORMATION process{};
        if(!job_){job_=CreateJobObjectW(nullptr,nullptr);JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};limit.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if(!job_||!SetInformationJobObject(job_,JobObjectExtendedLimitInformation,&limit,sizeof(limit))){CloseHandle(output);throw std::runtime_error("Cannot create analysis lifetime job");}}
        const BOOL created=CreateProcessW(executable,command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&startup,&process);
        CloseHandle(output);if(!created)throw std::runtime_error("Cannot start analysis worker");
        if(!AssignProcessToJobObject(job_,process.hProcess)){TerminateProcess(process.hProcess,2);CloseHandle(process.hProcess);CloseHandle(process.hThread);throw std::runtime_error("Cannot bind analysis worker lifetime");}
        ResumeThread(process.hThread);
        process_=process.hProcess;CloseHandle(process.hThread);
    }
    bool running()const{return process_&&WaitForSingleObject(process_,0)==WAIT_TIMEOUT;}
    void stop(){if(process_){if(running()){TerminateProcess(process_,2);WaitForSingleObject(process_,5000);}CloseHandle(process_);process_=nullptr;}}
private:HANDLE process_=nullptr,job_=nullptr;
};
inline std::string previewJpeg(const cv::Mat& image){
    cv::Mat scaled;double ratio=std::min(1.,960./image.cols);cv::resize(image,scaled,{},ratio,ratio,cv::INTER_AREA);
    std::vector<unsigned char> bytes;if(!cv::imencode(".jpg",scaled,bytes,{cv::IMWRITE_JPEG_QUALITY,80}))throw std::runtime_error("Preview encoding failed");
    constexpr char chars[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";std::string result;result.reserve((bytes.size()+2)/3*4);
    for(size_t i=0;i<bytes.size();i+=3){unsigned n=unsigned(bytes[i])<<16;if(i+1<bytes.size())n|=unsigned(bytes[i+1])<<8;if(i+2<bytes.size())n|=bytes[i+2];result+=chars[n>>18];result+=chars[(n>>12)&63];result+=i+1<bytes.size()?chars[(n>>6)&63]:'=';result+=i+2<bytes.size()?chars[n&63]:'=';}return result;
}

inline int runController(std::string video,const std::string& ballEngine,const std::string& sceneEngine,
                         int port,float threshold,bool intelligence,int maxFrames,const std::filesystem::path& root){
    using Clock=std::chrono::steady_clock;
    TelemetryServer server(port);if(!server.start())throw std::runtime_error("Recording server failed to start");
    AnalysisChild child;std::unique_ptr<JournalReader> journal;std::filesystem::path directory;
    cv::VideoCapture playback;cv::Mat image;int total=0,displayed=0,epoch=0,pendingSeek=0;double fps=0,playhead=1;
    std::string analysisId,error,jpeg;Object shown;std::future<VideoSelection> picker;
    std::shared_ptr<void> sourceLock;
    const auto cacheRoot=root/"artifacts"/"analyses";std::filesystem::create_directories(cacheRoot);
    const std::string ballHash=sha256(std::filesystem::u8path(ballEngine)),sceneHash=sha256(std::filesystem::u8path(sceneEngine));
    wchar_t runningExecutable[32768];if(!GetModuleFileNameW(nullptr,runningExecutable,32768))throw std::runtime_error("Cannot fingerprint analysis runtime");
    const std::string runtimeHash=sha256(runningExecutable);
    const auto open=[&](const std::string& path){
        const auto source=std::filesystem::absolute(std::filesystem::u8path(path));auto ext=source.extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(!std::filesystem::is_regular_file(source)||(ext!=".mp4"&&ext!=".avi"&&ext!=".mkv"&&ext!=".mov"&&ext!=".webm"&&ext!=".m4v"))throw std::runtime_error("Select a supported local video recording");
        // Keep the exact hashed bytes stable during analysis and replay. A changed
        // recording must be selected again and receives a different cache identity.
        HANDLE sourceHandle=CreateFileW(source.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(sourceHandle==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot lock recording for read-only analysis; close its writer first");
        auto candidateLock=std::shared_ptr<void>(sourceHandle,[](void* handle){CloseHandle(handle);});
        cv::VideoCapture candidate(source.u8string(),cv::CAP_FFMPEG);cv::Mat first;
        const double rate=candidate.get(cv::CAP_PROP_FPS);const int count=int(candidate.get(cv::CAP_PROP_FRAME_COUNT));
        if(!candidate.isOpened()||!std::isfinite(rate)||rate<=0||count<1||!candidate.read(first)||!candidate.set(cv::CAP_PROP_POS_FRAMES,0))throw std::runtime_error("Recording cannot be decoded; current recording is unchanged");
        Object identity{{"schema_version","2"},{"algorithm_revision",quote("complete-recording-v1")},{"source_sha256",quote(sha256(source))},
            {"source_path",quote(source.u8string())},{"total_frames",std::to_string(count)},{"source_fps",std::to_string(rate)},
            {"ball_model_sha256",quote(ballHash)},{"scene_model_sha256",quote(sceneHash)},
            {"runtime_sha256",quote(runtimeHash)},{"confidence_threshold",std::to_string(threshold)},{"experimental_intelligence",intelligence?"true":"false"},{"scene_interval_frames","1"}};
        const auto request=json(identity);std::filesystem::path reusable;
        if(!maxFrames)for(const auto& entry:std::filesystem::directory_iterator(cacheRoot)){
            if(!entry.is_directory()||!std::filesystem::exists(entry.path()/"completion.json")||!std::filesystem::exists(entry.path()/"request.json"))continue;
            try{if(readText(entry.path()/"request.json")==request&&object(readText(entry.path()/"completion.json")).at("status")==quote("COMPLETE")){
                JournalReader cached(entry.path());
                if(cached.available()!=count)throw std::runtime_error("Completed cache has incomplete frame coverage");
                cached.read(1);cached.read(count);
                reusable=entry.path();break;
            }}
            catch(const std::exception& e){std::cerr<<"[Analysis] Ignoring invalid cache: "<<e.what()<<std::endl;}
        }
        std::filesystem::path next=reusable;
        if(next.empty()){
            const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            next=cacheRoot/(std::to_string(stamp)+"_"+std::to_string(GetCurrentProcessId()));std::filesystem::create_directory(next);writeText(next/"request.json",request);
        }
        // Validation and hashing completed before replacing the previous source.
        child.stop();playback=std::move(candidate);sourceLock=std::move(candidateLock);video=source.u8string();directory=next;
        total=count;fps=rate;analysisId=next.filename().string();journal=std::make_unique<JournalReader>(next);
        displayed=0;playhead=1;pendingSeek=0;++epoch;shown.clear();jpeg.clear();error.clear();server.playbackPaused=false;
        if(reusable.empty()){
            std::vector<std::wstring> args{L"--sim",L"--video",source.wstring(),L"--analysis-worker",next.wstring(),
                L"--data-root",root.wstring(),
                std::filesystem::u8path(ballEngine).wstring(),L"--scene-engine",std::filesystem::u8path(sceneEngine).wstring(),L"--confidence",std::to_wstring(threshold)};
            if(intelligence)args.push_back(L"--experimental-intelligence");
            if(maxFrames){args.push_back(L"--max-frames");args.push_back(std::to_wstring(maxFrames));}
            child.start(args,next/"worker.log");
        }
        std::cout<<"[Source] Opened "<<video<<"; analysis="<<analysisId<<(reusable.empty()?" NEW":" CACHED")<<std::endl;
    };
    open(video);auto previous=Clock::now(),lastPublish=previous-std::chrono::seconds(1);bool force=true;
    while(!server.shutdownRequested){
        const auto now=Clock::now();double dt=std::chrono::duration<double>(now-previous).count();previous=now;
        auto requested=server.takeVideoRequest();
        if(server.videoPickerRequested.exchange(false)&&!picker.valid())picker=std::async(std::launch::async,selectLocalVideo);
        if(picker.valid()&&picker.wait_for(std::chrono::milliseconds(0))==std::future_status::ready){auto selection=picker.get();error=selection.error;if(!selection.path.empty())requested=selection.path;force=true;}
        if(!requested.empty()){try{open(requested);}catch(const std::exception& e){error=e.what();std::cerr<<"[Source] "<<e.what()<<std::endl;}force=true;dt=0;}
        if(server.restartRequested.exchange(false)){playhead=1;pendingSeek=0;++epoch;server.playbackPaused=false;force=true;dt=0;}
        int seek=server.seekFrameRequested.exchange(-1);
        const int step=server.frameStepRequested.exchange(0);if(step)seek=std::clamp(displayed+step,1,total);
        if(seek>0){if(seek>total)error="Requested frame is outside this recording";else{pendingSeek=seek;server.playbackPaused=true;++epoch;error.clear();}force=true;}
        const int available=journal->available();const bool complete=std::filesystem::exists(directory/"completion.json");
        const bool analysisFailed=!child.running()&&!complete;
        if(analysisFailed)error="Analysis worker stopped before completion; inspect its worker.log";
        const bool waiting=pendingSeek>available||available==0;
        if(pendingSeek&&pendingSeek<=available){playhead=pendingSeek;pendingSeek=0;dt=0;force=true;}
        if(!server.playbackPaused&&available>0){playhead+=dt*fps*server.playbackRate.load();playhead=std::min(playhead,double(available));}
        const int target=available?std::clamp(int(playhead),1,std::min(total,available)):0;
        if(force||now-lastPublish>=std::chrono::milliseconds(33)){
            if(target>0&&(target!=displayed||shown.empty())){
                // Sequential decoding when possible; exact seek only for jumps.
                int next=int(playback.get(cv::CAP_PROP_POS_FRAMES))+1;
                if(next!=target&&!playback.set(cv::CAP_PROP_POS_FRAMES,target-1))throw std::runtime_error("Replay seek failed");
                if(!playback.read(image)||image.empty()||int(playback.get(cv::CAP_PROP_POS_FRAMES))!=target)throw std::runtime_error("Replay decoded the wrong source frame");
                shown=journal->read(target);jpeg=previewJpeg(image);displayed=target;
            }
            auto packet=shown;
            if(packet.empty()){FramePresentation empty;empty.totalFrames=total;packet=object(TelemetryServer::serializeToJSON(0,{},float(fps),"simulation",nullptr,0,0,nullptr,std::filesystem::u8path(video).filename().u8string(),&empty));}
            packet["schema_version"]="2";packet["analysis_id"]=quote(analysisId);packet["source_id"]=std::to_string(epoch);packet["presentation_epoch"]=std::to_string(epoch);
            packet["preview_jpeg"]=quote(jpeg);packet["video_error"]=quote(error);packet["video_picker_open"]=picker.valid()?"true":"false";
            packet["paused"]=server.playbackPaused?"true":"false";packet["ended"]=(complete&&displayed==total)?"true":"false";
            packet["playback_rate"]=std::to_string(server.playbackRate.load());packet["total_frames"]=std::to_string(total);packet["skipped_frames"]="0";
            packet["analysis"]=json({{"status",quote(analysisFailed?"ERROR":complete?(available==total?"COMPLETE":"PARTIAL"):"RUNNING")},
                {"analyzed_frames",std::to_string(available)},{"total_frames",std::to_string(total)},{"waiting_for_analysis",waiting?"true":"false"},
                {"requested_frame",std::to_string(pendingSeek)},{"coverage",available==total?quote("ALL_SOURCE_FRAMES"):quote("CONSECUTIVE_PREFIX")}});
            packet["service"]=json({{"application",quote("TT_TRAJECTORY_ENGINE")},{"mode",quote("simulation")},{"pid",std::to_string(GetCurrentProcessId())},{"root",quote(root.u8string())}});
            server.broadcastJSON(json(packet));lastPublish=now;force=false;
        }
        if(maxFrames&&!child.running())return complete?0:1;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return 0;
}
} // namespace recording
