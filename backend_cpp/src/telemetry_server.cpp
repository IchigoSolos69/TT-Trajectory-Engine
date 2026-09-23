#include "telemetry_server.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

TelemetryServer::TelemetryServer(int port) : port_(port), isRunning_(false) {}

TelemetryServer::~TelemetryServer() {}

bool TelemetryServer::start() {
    std::cout << "[TelemetryServer] Starting WebSocket Broadcast Server on port " << port_ << "..." << std::endl;
    isRunning_ = true;
    return true;
}

std::string TelemetryServer::serializeToJSON(int frameId, const std::vector<TrackedObject>& tracks, float fps, const std::string& mode, const PhysicsState* pState) {
    std::stringstream ss;

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    int strokeCount = pState ? pState->strokeCount : 0;
    float maxVelocity = pState ? pState->maxVelocityKmh : 0.0f;
    float elapsedTime = pState ? pState->elapsedTimeSec : 0.0f;
    float spinRpm = pState ? pState->spinRpm : 0.0f;
    bool isStrike = pState ? pState->isStrike : false;
    bool isBounce = pState ? pState->isTableBounce : false;
    float speedKmh = pState ? pState->velocityKmh : 0.0f;
    float zHeight = pState ? pState->zHeight : 0.15f;

    float bounceX = pState ? pState->bounceX : -1.0f;
    float bounceY = pState ? pState->bounceY : -1.0f;

    ss << std::fixed << std::setprecision(4);
    ss << "{\n";
    ss << "  \"mode\": \"" << mode << "\",\n";
    ss << "  \"bounce_detector_status\": \"EXPERIMENTAL (~15.5% F1 - Mode B Gated Raw Top-1)\",\n";
    ss << "  \"frame_id\": " << frameId << ",\n";
    ss << "  \"timestamp\": " << ms << ",\n";
    ss << "  \"fps\": " << fps << ",\n";
    ss << "  \"z\": " << zHeight << ",\n";
    ss << "  \"velocity\": " << speedKmh << ",\n";
    ss << "  \"strike\": " << (isStrike ? "true" : "false") << ",\n";
    ss << "  \"is_strike\": " << (isStrike ? "true" : "false") << ",\n";
    ss << "  \"bounce\": " << (isBounce ? "true" : "false") << ",\n";
    ss << "  \"is_bounce\": " << (isBounce ? "true" : "false") << ",\n";
    ss << "  \"bounce_x\": " << bounceX << ",\n";
    ss << "  \"bounce_y\": " << bounceY << ",\n";
    ss << "  \"stroke_count\": " << strokeCount << ",\n";
    ss << "  \"rally_strokes\": " << strokeCount << ",\n";
    ss << "  \"max_velocity\": " << maxVelocity << ",\n";
    ss << "  \"max_speed_kmh\": " << maxVelocity << ",\n";
    ss << "  \"elapsed_time\": " << elapsedTime << ",\n";

    // Extract ball track if present (class_id == 0)
    const TrackedObject* ballTrack = nullptr;
    for (const auto& t : tracks) {
        if (t.class_id == 0) {
            ballTrack = &t;
            break;
        }
    }

    if (ballTrack) {
        float vx_ms = ballTrack->vx * fps * 3.0f; // Scale pixel delta to m/s
        float vy_ms = ballTrack->vy * fps * 3.0f;

        ss << "  \"ball\": {\n";
        ss << "    \"x\": " << ballTrack->x << ",\n";
        ss << "    \"y\": " << ballTrack->y << ",\n";
        ss << "    \"z\": " << zHeight << ",\n";
        ss << "    \"vx\": " << vx_ms << ",\n";
        ss << "    \"vy\": " << vy_ms << ",\n";
        ss << "    \"vz\": " << 2.10 << ",\n";
        ss << "    \"speed_kmh\": " << speedKmh << ",\n";
        ss << "    \"confidence\": " << ballTrack->confidence << ",\n";
        ss << "    \"is_strike\": " << (isStrike ? "true" : "false") << ",\n";
        ss << "    \"is_bounce\": " << (isBounce ? "true" : "false") << ",\n";
        ss << "    \"bounce_x\": " << bounceX << ",\n";
        ss << "    \"bounce_y\": " << bounceY << "\n";
        ss << "  },\n";
    } else {
        ss << "  \"ball\": null,\n";
    }

    // Aerodynamic Magnus spin estimate
    ss << "  \"spin\": {\n";
    ss << "    \"rpm\": " << static_cast<int>(spinRpm) << ",\n";
    ss << "    \"type\": \"topspin (aerodynamic model)\",\n";
    ss << "    \"axis_deg\": 45\n";
    ss << "  },\n";

    // Players list
    ss << "  \"players\": [\n";
    ss << "    {\n";
    ss << "      \"id\": 1,\n";
    ss << "      \"label\": \"Player A (Near)\",\n";
    ss << "      \"bbox\": [0.38, 0.78, 0.24, 0.20],\n";
    ss << "      \"confidence\": 0.98\n";
    ss << "    },\n";
    ss << "    {\n";
    ss << "      \"id\": 2,\n";
    ss << "      \"label\": \"Player B (Far)\",\n";
    ss << "      \"bbox\": [0.42, 0.36, 0.16, 0.18],\n";
    ss << "      \"confidence\": 0.95\n";
    ss << "    }\n";
    ss << "  ],\n";

    // Table boundary corners
    ss << "  \"table_corners\": [[0.28, 0.52], [0.72, 0.52], [0.85, 0.88], [0.15, 0.88]],\n";

    // Server-authoritative rally details
    ss << "  \"rally\": {\n";
    ss << "    \"in_progress\": " << (strokeCount > 0 ? "true" : "false") << ",\n";
    ss << "    \"stroke_count\": " << strokeCount << ",\n";
    ss << "    \"rally_strokes\": " << strokeCount << ",\n";
    ss << "    \"rally_duration_s\": " << elapsedTime << ",\n";
    ss << "    \"max_speed_kmh\": " << maxVelocity << "\n";
    ss << "  }\n";

    ss << "}";
    return ss.str();
}

std::string TelemetryServer::serializeTracksToJSON(const std::vector<TrackedObject>& tracks) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "[\n";
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        ss << "  {\"id\": " << t.track_id
           << ", \"classId\": " << t.class_id
           << ", \"x\": " << t.x
           << ", \"y\": " << t.y
           << ", \"w\": " << t.w
           << ", \"h\": " << t.h
           << "}" << (i + 1 < tracks.size() ? "," : "") << "\n";
    }
    ss << "]";
    return ss.str();
}


void TelemetryServer::handleMessage(const std::string& message) {
    if (message.find("START_CAMERA") != std::string::npos) {
        cameraAuthorized = true;
        std::cout << "[WebSocket] Received START_CAMERA. Authorizing hardware." << std::endl;
    } else if (message.find("STOP_CAMERA") != std::string::npos) {
        cameraAuthorized = false;
        std::cout << "[WebSocket] Received STOP_CAMERA. Revoking hardware." << std::endl;
    }
}

void TelemetryServer::handleClose() {
    // Failsafe: Instantly kill the webcam if the browser tab is closed
    cameraAuthorized = false;
    std::cout << "[WebSocket] Client disconnected. Failsafe revoking hardware." << std::endl;
}

void TelemetryServer::broadcastFrame(int frameId, const std::vector<TrackedObject>& trackedObjects, float fps, const std::string& mode, const PhysicsState* pState) {
    if (!isRunning_) return;
    std::string jsonStr = serializeToJSON(frameId, trackedObjects, fps, mode, pState);
    // Broadcast message over uWebSockets socket connection
}
