#include "tensorrt_engine.hpp"
#include "bytetrack.hpp"
#include "telemetry_server.hpp"
#include "physics_engine.hpp"
#include "scene_geometry.hpp"
#include "training_recorder.hpp"
#include "video_picker.hpp"
#include "trajectory_intelligence.hpp"
#include "rally_intelligence.hpp"
#include "hard_sample_selector.hpp"
#include "recording_controller.hpp"
#include <future>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <cctype>

static std::string base64(const std::vector<unsigned char>& bytes) {
    static constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result; result.reserve((bytes.size()+2)/3*4);
    for(size_t i=0;i<bytes.size();i+=3) {
        unsigned n=static_cast<unsigned>(bytes[i])<<16;
        if(i+1<bytes.size()) n|=static_cast<unsigned>(bytes[i+1])<<8;
        if(i+2<bytes.size()) n|=bytes[i+2];
        result+=alphabet[(n>>18)&63]; result+=alphabet[(n>>12)&63];
        result+=i+1<bytes.size()?alphabet[(n>>6)&63]:'='; result+=i+2<bytes.size()?alphabet[n&63]:'=';
    }
    return result;
}

int runEngine(int argc,char** argv) try {
    bool isLiveMode = false;
    std::string enginePath = "models/yolov11_ttgames.engine";
    std::string videoPath;
    std::string scenePath = "models/scene.engine";
    bool modeSpecified = false;
    int maxFrames = 0;
    int serverPort=9002;
    bool experimentalIntelligence=false;
    std::filesystem::path analysisDirectory;
    float confidenceThreshold = 0.25f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--live" || arg == "--sim") {
            if (modeSpecified) throw std::runtime_error("Specify exactly one source mode: --sim or --live.");
            modeSpecified = true;
            isLiveMode = arg == "--live";
        } else if (arg == "--experimental-intelligence") {
            experimentalIntelligence=true;
        } else if(arg=="--analysis-worker" && i+1<argc){
            analysisDirectory=std::filesystem::u8path(argv[++i]);
        } else if (arg == "--port" && i+1<argc) {
            serverPort=std::stoi(argv[++i]);if(serverPort<1024||serverPort>65535)throw std::runtime_error("Invalid telemetry port");
        } else if (arg == "--scene-engine" && i + 1 < argc) {
            scenePath = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            videoPath = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            maxFrames = std::stoi(argv[++i]);
            if (maxFrames <= 0) throw std::runtime_error("--max-frames must be positive.");
        } else if (arg == "--confidence" && i + 1 < argc) {
            confidenceThreshold = std::stof(argv[++i]);
            if (!(confidenceThreshold > 0 && confidenceThreshold < 1)) throw std::runtime_error("--confidence must be between 0 and 1.");
        } else if (arg.rfind(".engine") != std::string::npos) {
            enginePath = arg;
            if (!std::filesystem::is_regular_file(enginePath)) throw std::runtime_error("Explicit engine file does not exist: " + enginePath);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (isLiveMode && !videoPath.empty()) throw std::runtime_error("--video is only permitted with --sim.");
    if(isLiveMode&&!analysisDirectory.empty())throw std::runtime_error("Recording analysis cannot open a live camera");


    cv::VideoCapture cap;
    if(!isLiveMode) {
        if(videoPath.empty()) {
            for(const auto& prefix:{"","../","../../","../../../"}) {
                const auto path=std::string(prefix)+"python_tools/dataset/raw_videos/game_1.mp4";
                if(std::filesystem::is_regular_file(path)){videoPath=path;break;}
            }
        }
        if(!std::filesystem::is_regular_file(std::filesystem::u8path(videoPath)) || !cap.open(videoPath,cv::CAP_FFMPEG))
            throw std::runtime_error("Simulation requires readable dataset video: "+videoPath);
    }
    auto projectRoot=std::filesystem::current_path();
    for(int depth=0;depth<5&&!std::filesystem::is_directory(projectRoot/"backend_cpp");++depth)projectRoot=projectRoot.parent_path();
    if(!isLiveMode&&analysisDirectory.empty()){
        cap.release();
        return recording::runController(videoPath,TensorRTEngine::resolveEnginePath(enginePath),TensorRTEngine::resolveEnginePath(scenePath),serverPort,confidenceThreshold,experimentalIntelligence,maxFrames,projectRoot);
    }
    const bool analysisWorker=!analysisDirectory.empty();
    TensorRTEngine ballEngine(enginePath,640,640), sceneEngine(scenePath,640,640,2);
    if(!ballEngine.init() || !sceneEngine.init()) throw std::runtime_error("Both ball and scene detectors must initialize; no fallback detections.");
    TelemetryServer server(serverPort); if(!analysisWorker&&!server.start()) throw std::runtime_error("Telemetry server failed to start.");
    std::unique_ptr<recording::JournalWriter> journal;
    if(analysisWorker)journal=std::make_unique<recording::JournalWriter>(analysisDirectory,recording::readText(analysisDirectory/"request.json"));
    const float lowConfidence=std::min(.1f,confidenceThreshold*.5f);
    ByteTracker tracker(confidenceThreshold,lowConfidence,30);
    PhysicsEngine physics;
    intelligence::TableCalibration tableCalibration;
    intelligence::TrajectoryTracker trajectoryTracker;
    intelligence::RallyIntelligence rallyIntelligence;
    HardSampleSelector hardSamples;
    ObservedSpinEstimator spinEstimator;
    std::future<VideoSelection> videoPicker;
    int sourceId=1;
    float liveTime=0;
    using Clock=std::chrono::steady_clock;
    auto lastCapture=Clock::now(), lastPreview=Clock::now()-std::chrono::seconds(1), throughputStart=Clock::now();
    int throughputFrames=0;
    auto playbackAnchorTime=Clock::now();
    double playbackAnchorFrame=0;
    float previousRate=1;
    int activeCamera=-1, frameId=0, processed=0, previousFrame=0;
    float sourceFps=isLiveMode?0.f:static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    if(!isLiveMode && (!(sourceFps>0) || !std::isfinite(sourceFps))) throw std::runtime_error("Invalid source FPS");
    int totalFrames=isLiveMode?0:static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    bool ended=false;
    const std::string mode=isLiveMode?"live":"simulation";
    FramePresentation preview; preview.totalFrames=totalFrames;
    auto recordingRoot=std::filesystem::current_path();
    for(int depth=0;depth<5 && !std::filesystem::is_directory(recordingRoot/"backend_cpp");++depth)
        recordingRoot=recordingRoot.parent_path();
    TrainingRecorder recorder(recordingRoot/"training_data"/"live");
    bool recordingSession=false;
    const auto refreshRecording=[&]{
        const auto recording=recorder.status();
        preview.recordingActive=recording.active;preview.recordingFailed=recording.failed;
        preview.recordingSaved=recording.saved;preview.recordingDropped=recording.dropped;preview.recordingClip=recording.clip;
    };
    std::vector<TrackedObject> tracks;
    std::vector<std::pair<float,float>> normalizedCorners;
    cv::Mat homography, frame;
    PhysicsState state{}; bool measuredPhysics=false;
    int publishedFrameId=0;
    FramePresentation publishedPreview;
    std::vector<TrackedObject> publishedTracks;
    std::vector<std::pair<float,float>> publishedCorners;
    PhysicsState publishedState{}; bool publishedPhysics=false;
    float publishedInferMs=0, publishedTotalMs=0;
    float inferMs=0,totalMs=0, captureFps=0;
    float previousImageX=0,previousImageY=0;bool hadImageBall=false;
    std::cout<<"[Main] "<<mode<<" source="<<(isLiveMode?"selected camera (standby)":videoPath)<<std::endl;
    while(!server.shutdownRequested) {
        bool seekOneFrame=false;
        const auto start=Clock::now();
        auto requestedVideo=server.takeVideoRequest();
        if(server.videoPickerRequested.exchange(false)) {
            if(isLiveMode)preview.videoError="Switch the engine to simulation mode to open a recording.";
            else if(!videoPicker.valid()) {
                videoPicker=std::async(std::launch::async,selectLocalVideo);
                preview.videoPickerOpen=true;preview.videoError.clear();
            }
        }
        if(videoPicker.valid() && videoPicker.wait_for(std::chrono::milliseconds(0))==std::future_status::ready) {
            auto selected=videoPicker.get();preview.videoPickerOpen=false;preview.videoError=selected.error;
            if(!selected.path.empty())requestedVideo=selected.path;
        }
        if(!requestedVideo.empty()) {
            if(isLiveMode)preview.videoError="Recordings can only be opened in simulation mode.";
            else try {
                const auto path=std::filesystem::u8path(requestedVideo);
                auto ext=path.extension().string();std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return char(std::tolower(c));});
                if(!std::filesystem::is_regular_file(path) || (ext!=".mp4" && ext!=".avi" && ext!=".mkv" && ext!=".mov" && ext!=".webm" && ext!=".m4v"))
                    throw std::runtime_error("Select an existing MP4, AVI, MKV, MOV, WebM or M4V recording.");
                cv::VideoCapture candidate(requestedVideo,cv::CAP_FFMPEG);cv::Mat first;
                const float fps=static_cast<float>(candidate.get(cv::CAP_PROP_FPS));
                const int count=static_cast<int>(candidate.get(cv::CAP_PROP_FRAME_COUNT));
                if(!candidate.isOpened() || !std::isfinite(fps) || fps<=0 || count<=0 || !candidate.read(first) || first.empty() || !candidate.set(cv::CAP_PROP_POS_FRAMES,0))
                    throw std::runtime_error("This recording could not be decoded. The current recording is unchanged.");
                cap=candidate;videoPath=requestedVideo;sourceFps=fps;totalFrames=count;
                preview=FramePresentation{};preview.totalFrames=totalFrames;preview.sourceId=++sourceId;
                server.restartRequested=true;
                std::cout<<"[Source] Opened "<<videoPath<<" ("<<totalFrames<<" frames, "<<sourceFps<<" FPS)"<<std::endl;
            }catch(const std::exception& e){preview.videoError=e.what();std::cerr<<"[Source] "<<e.what()<<std::endl;}
        }
        if(isLiveMode) {
            if(server.seekFrameRequested.exchange(-1)>=0)preview.videoError="Seeking is only available for recordings.";
            if(cap.isOpened() && (!server.cameraAuthorized || activeCamera!=server.targetCameraIndex)) {
                server.trainingRecordingRequested=false;recorder.stop();recordingSession=false;
                cap.release(); activeCamera=-1; frame.release(); homography.release(); normalizedCorners.clear();
                physics.reset(); tracker=ByteTracker(confidenceThreshold,lowConfidence,30); tracks.clear(); preview=FramePresentation{};
                physics.resetRally();state={};hadImageBall=false;frameId=previousFrame=0;
                tableCalibration.reset();trajectoryTracker.reset();rallyIntelligence.reset();
                spinEstimator.reset();preview.sourceId=++sourceId;liveTime=0;
            }
            if(server.cameraAuthorized && !cap.isOpened()) {
                activeCamera=server.targetCameraIndex;
                if(!cap.open(activeCamera,cv::CAP_DSHOW)) {std::cerr<<"[Hardware] Selected camera failed: "<<activeCamera<<std::endl;server.cameraAuthorized=false;}
                else {cap.set(cv::CAP_PROP_BUFFERSIZE,1); lastCapture=Clock::now();}
            }
            if(!cap.isOpened()) {
                server.trainingRecordingRequested=false;recorder.stop();recordingSession=false;refreshRecording();
                preview.cameraActive=false;preview.cameraIndex=server.targetCameraIndex;
                throughputStart=Clock::now();throughputFrames=0;
                server.broadcastFrame(frameId,{},0,mode,nullptr,0,0,nullptr,"",&preview);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));continue;
            }
        } else if(!analysisWorker) {
            // Recording is exclusively an explicit action on an authorized live camera.
            server.trainingRecordingRequested=false;
            const int requestedFrame=server.seekFrameRequested.exchange(-1);
            if(requestedFrame>=1){
                if(requestedFrame>totalFrames)preview.videoError="Requested frame is outside this recording.";
                else if(!cap.set(cv::CAP_PROP_POS_FRAMES,requestedFrame-1))preview.videoError="The recording could not seek to this frame.";
                else{
                    seekOneFrame=true;ended=false;frameId=previousFrame=requestedFrame-1;preview.sourceId=++sourceId;
                    trajectoryTracker.reset();tableCalibration.reset();rallyIntelligence.seek();physics.reset();physics.resetRally();state={};
                    tracker=ByteTracker(confidenceThreshold,lowConfidence,30);tracks.clear();normalizedCorners.clear();homography.release();
                    preview.impacts.clear();preview.imageImpacts.clear();preview.strikes.clear();preview.videoError.clear();spinEstimator.reset();hadImageBall=false;
                    server.playbackPaused=false;lastPreview=Clock::now()-std::chrono::seconds(1);
                    playbackAnchorFrame=requestedFrame-1;playbackAnchorTime=Clock::now();
                }
            }
            if(server.restartRequested.exchange(false)) {
                if(!cap.set(cv::CAP_PROP_POS_FRAMES,0)) throw std::runtime_error("Cannot restart dataset");
                frameId=previousFrame=0; ended=false; physics.reset(); tracks.clear(); homography.release(); normalizedCorners.clear();
                physics.resetRally();state={};hadImageBall=false;preview.impacts.clear();preview.imageImpacts.clear();preview.skippedFrames=0;
                tableCalibration.reset();trajectoryTracker.reset();rallyIntelligence.reset();
                spinEstimator.reset();preview.strikes.clear();preview.sourceId=++sourceId;
                publishedPreview=FramePresentation{};publishedTracks.clear();publishedCorners.clear();publishedPhysics=false;publishedFrameId=0;
                lastPreview=Clock::now()-std::chrono::seconds(1);server.playbackPaused=false;
                throughputStart=Clock::now();throughputFrames=0;preview.processingFps=0;
                playbackAnchorFrame=0;playbackAnchorTime=Clock::now();
                tracker=ByteTracker(confidenceThreshold,lowConfidence,30);
            }
            if(server.playbackPaused || ended) {
                throughputStart=Clock::now();throughputFrames=0;
                playbackAnchorFrame=frameId;playbackAnchorTime=Clock::now();
                auto pausedPreview=publishedPreview;
                pausedPreview.videoError=preview.videoError;pausedPreview.videoPickerOpen=preview.videoPickerOpen;
                pausedPreview.paused=server.playbackPaused;pausedPreview.ended=ended;pausedPreview.playbackRate=server.playbackRate;
                server.broadcastFrame(publishedFrameId,publishedTracks,sourceFps,mode,publishedPhysics?&publishedState:nullptr,publishedTotalMs,publishedInferMs,
                    publishedCorners.size()==4?&publishedCorners:nullptr,std::filesystem::path(videoPath).filename().string(),&pausedPreview);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));continue;
            }
            const float rate=server.playbackRate;
            if(rate!=previousRate) {playbackAnchorFrame=frameId;playbackAnchorTime=Clock::now();previousRate=rate;}
            // Keep footage on its playback clock when inference is slower than the source.
            // Every displayed frame is still inferred; skipped frames are reported explicitly.
            const int desired=seekOneFrame?requestedFrame:std::min(totalFrames,1+static_cast<int>(playbackAnchorFrame+
                std::chrono::duration<double>(Clock::now()-playbackAnchorTime).count()*sourceFps*rate));
            int next=static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES))+1;
            if(desired-next>sourceFps*.5f) {
                if(!cap.set(cv::CAP_PROP_POS_FRAMES,desired-1))throw std::runtime_error("Playback seek failed");
                preview.skippedFrames+=desired-next;
            } else while(next<desired) {
                if(!cap.grab())throw std::runtime_error("Playback skip decoding failed");
                ++next; ++preview.skippedFrames;
            }
        }
        if(!cap.read(frame) || frame.empty()) {
            if(isLiveMode) {std::cerr<<"[Hardware] Camera read failed; releasing source"<<std::endl;server.cameraAuthorized=false;continue;}
            if(frameId<totalFrames) throw std::runtime_error("Dataset decoding ended early at "+std::to_string(frameId));
            if(analysisWorker||maxFrames>0) break;
            ended=true; continue;
        }
        const auto captured=Clock::now();
        float dt=isLiveMode?std::chrono::duration<float>(captured-lastCapture).count():1.f/sourceFps;
        lastCapture=captured; captureFps=1.f/std::max(dt,.001f);
        frameId=isLiveMode?frameId+1:static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
        if(!isLiveMode) dt=(frameId-previousFrame)/sourceFps;
        previousFrame=frameId;
        if(isLiveMode)liveTime+=dt;
        preview.sourceTime=isLiveMode?liveTime:(frameId-1)/sourceFps;
        if(isLiveMode) {
            if(server.trainingRecordingRequested && !recordingSession) {
                if(server.activeLearningRequested&&!experimentalIntelligence){server.trainingRecordingRequested=false;preview.videoError="Hard-example capture requires --experimental-intelligence.";}
            }
            if(server.trainingRecordingRequested && !recordingSession) {
                hardSamples.reset();
                recordingSession=recorder.start(activeCamera,frame.cols,frame.rows,server.trainingBurstRequested?5:1);
                if(!recordingSession) server.trainingRecordingRequested=false;
            }
            if(!server.trainingRecordingRequested && recordingSession) {recorder.stop();recordingSession=false;}
            if(recordingSession) {
                recorder.expire();
                if(!server.activeLearningRequested)recorder.offer(frame,frameId);
                if(!recorder.status().active) {server.trainingRecordingRequested=false;recorder.stop();recordingSession=false;}
            }
            refreshRecording();
        }
        auto input=TensorRTEngine::preprocessOpenCVFrame(frame);
        const auto inferStart=Clock::now();
        const auto detections=ballEngine.inferFrame(input,frame.cols,frame.rows,lowConfidence);
        inferMs=std::chrono::duration<float,std::milli>(Clock::now()-inferStart).count();
        tracks=tracker.update(detections);
        preview.measuredBall=!detections.empty() && detections.front().confidence>=confidenceThreshold;
        if(preview.measuredBall)preview.observedBall=detections.front();
        // Weak observations help association but never bypass the display threshold.
        tracks.erase(std::remove_if(tracks.begin(),tracks.end(),[&](const auto& t){return t.confidence<confidenceThreshold;}),tracks.end());
        preview.imageVx=preview.imageVy=0;
        if(!detections.empty() && detections.front().confidence>=confidenceThreshold) {
            const auto& d=detections.front();const float x=d.x+d.w*.5f,y=d.y+d.h*.5f;
            if(hadImageBall && dt>0 && dt<=.1f) {preview.imageVx=(x-previousImageX)/dt;preview.imageVy=(y-previousImageY)/dt;}
            previousImageX=x;previousImageY=y;hadImageBall=true;
        }else hadImageBall=false;
        const bool publish=analysisWorker||Clock::now()-lastPreview>=std::chrono::milliseconds(33) || (!isLiveMode && frameId==totalFrames);
        const bool sceneDue=analysisWorker?(frameId-1)%4==0:publish;
        if(sceneDue) {
            preview.scene=sceneEngine.inferFrame(input,frame.cols,frame.rows,.4f);
            const auto corners=findTableSurface(frame,preview.scene);
            if(experimentalIntelligence){
                const int calibrationEpoch=tableCalibration.epoch;
                tableCalibration.update(frame,corners,preview.sourceTime);
                if(tableCalibration.epoch!=calibrationEpoch){trajectoryTracker.reset();physics.reset();hadImageBall=false;preview.impacts.clear();}
                homography=tableCalibration.valid(preview.sourceTime)?tableCalibration.H:cv::Mat{};
                normalizedCorners.clear();
                if(!homography.empty())for(const auto& p:tableCalibration.filtered)normalizedCorners.emplace_back(p.x/frame.cols,p.y/frame.rows);
            }else{
                float displacement=0;if(normalizedCorners.size()==4&&corners.size()==4)for(int i=0;i<4;++i)displacement+=cv::norm(corners[i]-cv::Point2f(normalizedCorners[i].first*frame.cols,normalizedCorners[i].second*frame.rows))/4;
                const bool changed=homography.empty()||corners.size()!=4||displacement>std::max(3.f,frame.cols*.004f);
                if(changed){normalizedCorners.clear();homography.release();physics.reset();hadImageBall=false;}
                if(changed&&corners.size()==4){
                    for(const auto& p:corners)normalizedCorners.emplace_back(p.x/frame.cols,p.y/frame.rows);
                    const bool horizontal=cv::norm(corners[1]-corners[0])+cv::norm(corners[2]-corners[3])>cv::norm(corners[3]-corners[0])+cv::norm(corners[2]-corners[1]);
                    const std::vector<cv::Point2f> world=horizontal?std::vector<cv::Point2f>{{0,0},{0,2.74f},{1.525f,2.74f},{1.525f,0}}:std::vector<cv::Point2f>{{0,0},{1.525f,0},{1.525f,2.74f},{0,2.74f}};
                    homography=cv::getPerspectiveTransform(corners,world);
                }
            }
        }
        auto trajectory=intelligence::TrajectorySample{};
        if(experimentalIntelligence)trajectory=trajectoryTracker.update(frameId,preview.sourceTime,frame,detections,preview.scene,tableCalibration);
        else if(preview.measuredBall&&!homography.empty()){
            const auto& d=preview.observedBall;trajectory.pixel={(d.x+d.w*.5f)*frame.cols,(d.y+d.h*.5f)*frame.rows};
            std::vector<cv::Point2f> points;cv::perspectiveTransform(std::vector<cv::Point2f>{trajectory.pixel},points,homography);
            trajectory.table=points.front();trajectory.positionValid=std::isfinite(trajectory.table.x)&&std::isfinite(trajectory.table.y);trajectory.accepted=true;
        }
        if(experimentalIntelligence)rallyIntelligence.update(trajectory,preview.scene,frame.size());
        if(isLiveMode&&recordingSession&&server.activeLearningRequested){
            const auto metadata=hardSamples.select(frame,trajectory,detections,preview.scene,tableCalibration.valid(preview.sourceTime),ballEngine.getSha256Hash(),sceneEngine.getSha256Hash());
            if(!metadata.empty())recorder.offer(frame,frameId,metadata);
        }
        measuredPhysics=false;
        preview.positionValid=false;
        state.isStrike=false;state.isTableBounce=false;state.velocityKmh=0;state.spinRpm=0;
        if(experimentalIntelligence){
            // One authoritative interpretation: the experimental HUD consumes
            // the same trajectory/event graph as its timeline, never legacy
            // two-frame strike/bounce rules applied to the filtered samples.
            const int oldStrikes=state.strokeCount;
            state={};state.strokeCount=rallyIntelligence.strikeCount;
            state.maxVelocityKmh=rallyIntelligence.peakStrikeKmh;
            state.elapsedTimeSec=float(rallyIntelligence.observedDuration);
            state.isStrike=state.strokeCount>oldStrikes;
            if(trajectory.positionValid){
                preview.tableX=trajectory.table.x;preview.tableY=trajectory.table.y;preview.positionValid=true;
                state.velocityKmh=float(cv::norm(trajectory.velocity))*3.6f;
            }
            preview.strikes.clear();preview.impacts.clear();
            for(const auto& event:rallyIntelligence.events)if(event.type=="STRIKE"&&event.status=="RULE_SUPPORTED")
                preview.strikes.push_back({event.id,event.frame,float(event.time),float(cv::norm(event.outgoing))*3.6f});
            if(preview.strikes.size()>100)preview.strikes.erase(preview.strikes.begin(),preview.strikes.end()-100);
        }else if(trajectory.accepted && trajectory.positionValid) {
            const float px=trajectory.pixel.x,py=trajectory.pixel.y;
            preview.tableX=trajectory.table.x;preview.tableY=trajectory.table.y;
            preview.positionValid=true;
            if(dt>.1f)physics.reset();
            if(preview.positionValid) {state=physics.update(preview.tableX,preview.tableY,dt,px,py); measuredPhysics=true;}
            else physics.reset();
            if(state.isStrike) {
                preview.strikes.push_back({state.strokeCount,frameId,preview.sourceTime,state.velocityKmh});
                if(preview.strikes.size()>100)preview.strikes.erase(preview.strikes.begin());
            }
            if(state.isTableBounce && state.bounceX>=0 && state.bounceX<=1.525f && state.bounceY>=0 && state.bounceY<=2.74f) {
                preview.impacts.emplace_back(state.bounceX,state.bounceY);
                if(preview.impacts.size()>200)preview.impacts.erase(preview.impacts.begin());
            }
        } else physics.reset();
        if(state.isStrike || state.isTableBounce)spinEstimator.reset();
        preview.observedSpin=ObservedSpin{};
        if(preview.measuredBall && preview.observedBall.confidence>=.6f) {
            const auto& d=preview.observedBall;
            preview.observedSpin=spinEstimator.update(frame,cv::Rect(int(d.x*frame.cols),int(d.y*frame.rows),int(d.w*frame.cols),int(d.h*frame.rows)),dt);
        }else spinEstimator.reset();
        if(publish) {
            if(experimentalIntelligence){
                preview.intelligenceJson=intelligence::serialize(tableCalibration,trajectoryTracker,preview.sourceTime,ballEngine.getSha256Hash(),sceneEngine.getSha256Hash(),analysisWorker?1:180);
                preview.intelligenceJson.pop_back();preview.intelligenceJson+=",\"rally_graph\":"+rallyIntelligence.json()+"}";
            }
            // Cumulative impacts survive preview throttling and dropped network packets.
            preview.imageImpacts.clear();
            if(!homography.empty() && !preview.impacts.empty()) {
                std::vector<cv::Point2f> world,image;
                for(const auto& p:preview.impacts)world.emplace_back(p.first,p.second);
                cv::perspectiveTransform(world,image,homography.inv());
                for(const auto& p:image)preview.imageImpacts.emplace_back(p.x/frame.cols,p.y/frame.rows);
            }
            if(!analysisWorker)preview.jpegBase64=recording::previewJpeg(frame);
            preview.width=frame.cols;preview.height=frame.rows;
            preview.cameraActive=isLiveMode;preview.cameraIndex=isLiveMode?activeCamera:0;
            if(seekOneFrame)server.playbackPaused=true;
            preview.playbackRate=isLiveMode?1.f:server.playbackRate.load();preview.paused=seekOneFrame;preview.ended=false;
            totalMs=std::chrono::duration<float,std::milli>(Clock::now()-start).count();
            preview.previewFps=1.f/std::max(.001f,std::chrono::duration<float>(Clock::now()-lastPreview).count());
            if(analysisWorker){
                const auto packet=TelemetryServer::serializeToJSON(frameId,tracks,sourceFps,mode,&state,totalMs,inferMs,
                    normalizedCorners.size()==4?&normalizedCorners:nullptr,std::filesystem::u8path(videoPath).filename().u8string(),&preview);
                journal->append(frameId,packet);
            }else server.broadcastFrame(frameId,tracks,isLiveMode?captureFps:sourceFps,mode,&state,totalMs,inferMs,
                normalizedCorners.size()==4?&normalizedCorners:nullptr,"",&preview);
            publishedFrameId=frameId;publishedTracks=tracks;publishedCorners=normalizedCorners;
            publishedPreview=preview;publishedState=state;publishedPhysics=true;
            publishedInferMs=inferMs;publishedTotalMs=totalMs;
            lastPreview=Clock::now();
        }
        ++throughputFrames;
        const float throughputSeconds=std::chrono::duration<float>(Clock::now()-throughputStart).count();
        if(throughputSeconds>=1.f) {
            preview.processingFps=throughputFrames/throughputSeconds;
            throughputStart=Clock::now(); throughputFrames=0;
        }
        if(++processed==maxFrames) break;
        if(processed%240==0) std::cout<<"[Main] frame="<<frameId<<" source="<<mode<<" ball_candidates="<<detections.size()<<" scene="<<preview.scene.size()<<" calibrated="<<!homography.empty()<<" total_ms="<<totalMs<<" ball_infer_ms="<<inferMs<<std::endl;
        if(!isLiveMode&&!analysisWorker) {
            const auto deadline=start+std::chrono::duration<double>(1./(sourceFps*server.playbackRate.load()));
            std::this_thread::sleep_until(deadline);
        }
    }
    if(journal)journal->finish(totalFrames);
    cap.release();
    return 0;
} catch(const std::exception& error) {
    for(int i=1;i+1<argc;++i)if(std::string(argv[i])=="--analysis-worker"){
        try{recording::writeText(std::filesystem::u8path(argv[i+1])/"failure.json",recording::json({{"error",recording::quote(error.what())}}));}
        catch(const std::exception& saveError){std::cerr<<"[Analysis] Cannot save failure: "<<saveError.what()<<std::endl;}break;
    }
    std::cerr<<"[Main] FATAL: "<<error.what()<<std::endl;return 1;
}

// Windows CRT narrow argv uses the active code page. Convert wide command-line
// arguments explicitly so FFmpeg and filesystem::u8path receive actual UTF-8.
#ifdef _WIN32
int wmain(int argc,wchar_t** arguments){
    std::vector<std::string> utf8;utf8.reserve(argc);
    for(int i=0;i<argc;++i){
        const int size=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,arguments[i],-1,nullptr,0,nullptr,nullptr);
        if(size<=0){std::cerr<<"Invalid Unicode command line\n";return 1;}
        std::string value(size,'\0');WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,arguments[i],-1,value.data(),size,nullptr,nullptr);value.pop_back();utf8.push_back(std::move(value));
    }
    std::vector<char*> argv;for(auto& value:utf8)argv.push_back(value.data());return runEngine(argc,argv.data());
}
#else
int main(int argc,char** argv){return runEngine(argc,argv);}
#endif
