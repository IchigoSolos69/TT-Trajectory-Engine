#ifndef TELEMETRY_SERVER_HPP
#define TELEMETRY_SERVER_HPP

#include "bytetrack.hpp"
#include "physics_engine.hpp"
#include <string>
#include <vector>
#include <atomic>

class TelemetryServer {
public:
    TelemetryServer(int port = 9002);
    ~TelemetryServer();

    bool start();
    void broadcastFrame(int frameId, const std::vector<TrackedObject>& trackedObjects, float fps = 120.0f, const std::string& mode = "simulation", const PhysicsState* pState = nullptr);
    
    static std::string serializeToJSON(int frameId, const std::vector<TrackedObject>& tracks, float fps, const std::string& mode = "simulation", const PhysicsState* pState = nullptr);
    static std::string serializeTracksToJSON(const std::vector<TrackedObject>& tracks);

    void handleMessage(const std::string& message);
    void handleClose();

    std::atomic<bool> cameraAuthorized{false};

private:
    int port_;
    bool isRunning_;
};

#endif // TELEMETRY_SERVER_HPP
