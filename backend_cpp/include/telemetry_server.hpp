#ifndef TELEMETRY_SERVER_HPP
#define TELEMETRY_SERVER_HPP

#include "bytetrack.hpp"
#include "physics_engine.hpp"
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include "observed_spin.hpp"

struct StrikeMeasurement {int id=0,frame=0;double time=0;float speedKmh=0;};

struct FramePresentation {
    std::string intelligenceJson;
    std::string ballModelJson,ballBlurJson,ballFlightJson;
    int sourceId=1;
    bool measuredBall=false,positionValid=false,videoPickerOpen=false;
    Detection observedBall{};
    std::string ballObservationState="DETECTOR_ONLY",ballObservationReason="DETECTOR_SCORE_ONLY";
    int sceneFrameId=0;
    std::vector<Detection> suppressedPersonParts;
    ObservedSpin observedSpin;
    std::string videoError;
    std::vector<StrikeMeasurement> strikes;
    std::string jpegBase64;
    int width = 0, height = 0, totalFrames = 0;
    int skippedFrames = 0;
    bool cameraActive = false;
    int cameraIndex = 0;
    bool recordingActive = false, recordingFailed = false;
    int recordingSaved = 0, recordingDropped = 0;
    std::string recordingClip;
    double sourceTime = 0;
    float playbackRate = 1, processingFps = 0, previewFps = 0;
    bool paused = false, ended = false;
    float tableX = 0, tableY = 0;
    float imageVx = 0, imageVy = 0;
    std::vector<Detection> scene;
    std::vector<std::pair<float,float>> impacts, imageImpacts;
};

class TelemetryServer {
public:
    TelemetryServer(int port = 9002);
    ~TelemetryServer();

    bool start();
    void stop();
    void broadcastJSON(const std::string& json);
    void broadcastFrame(int frameId, const std::vector<TrackedObject>& trackedObjects, float fps = 120.0f, const std::string& mode = "simulation", const PhysicsState* pState = nullptr, float totalMs = 0.0f, float inferMs = 0.0f, const std::vector<std::pair<float, float>>* corners = nullptr, const std::string& sourceVideo = "", const FramePresentation* presentation = nullptr);
    
    static std::string serializeToJSON(int frameId, const std::vector<TrackedObject>& tracks, float fps, const std::string& mode = "simulation", const PhysicsState* pState = nullptr, float totalMs = 0.0f, float inferMs = 0.0f, const std::vector<std::pair<float, float>>* corners = nullptr, const std::string& sourceVideo = "", const FramePresentation* presentation = nullptr);
    static std::string serializeTracksToJSON(const std::vector<TrackedObject>& tracks);

    void handleMessage(const std::string& message);
    void handleClose();

    std::atomic<bool> cameraAuthorized{false};
    std::atomic<bool> shutdownRequested{false};
    std::atomic<int> targetCameraIndex{0};
    std::atomic<float> playbackRate{1.0f};
    std::atomic<bool> playbackPaused{false};
    std::atomic<bool> restartRequested{false};
    std::atomic<int> seekFrameRequested{-1};
    std::atomic<int> frameStepRequested{0};
    std::atomic<bool> trainingRecordingRequested{false};
    std::atomic<bool> activeLearningRequested{false};
    std::atomic<bool> trainingBurstRequested{false};
    std::atomic<bool> videoPickerRequested{false};
    std::string takeVideoRequest();

private:
    std::mutex videoMutex_;
    std::string pendingVideo_;
    int port_;
    std::atomic<bool> isRunning_{false};
    void serverLoop();
    class Impl;
    Impl* impl_;
};

#endif // TELEMETRY_SERVER_HPP
