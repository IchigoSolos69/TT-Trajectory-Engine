#include "telemetry_server.hpp"
#include "analysis_journal.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <wincrypt.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "crypt32.lib")
#endif

#include <algorithm>
#include <cctype>
#include <memory>
#include <map>
#include <cstdlib>
#include <cmath>

class TelemetryServer::Impl {
public:
    SOCKET listenSocket = INVALID_SOCKET;
    std::vector<SOCKET> clientSockets;
    std::mutex socketMutex;
    std::thread serverThread;
    std::shared_ptr<std::vector<uint8_t>> latest;
    uint64_t sequence = 0;
    struct Output { std::shared_ptr<std::vector<uint8_t>> packet; size_t offset=0; uint64_t sequence=0; };
    std::map<SOCKET, Output> outgoing;
    std::map<SOCKET, std::string> incoming;

    static std::string toLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    }

    std::string computeAcceptKey(const std::string& clientKey) {
        std::string magic = clientKey + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        HCRYPTPROV hProv = 0;
        HCRYPTHASH hHash = 0;
        BYTE sha1Hash[20];
        DWORD hashLen = 20;

        if (CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            if (CryptCreateHash(hProv, CALG_SHA1, 0, 0, &hHash)) {
                CryptHashData(hHash, (BYTE*)magic.c_str(), (DWORD)magic.length(), 0);
                CryptGetHashParam(hHash, HP_HASHVAL, sha1Hash, &hashLen, 0);
                CryptDestroyHash(hHash);
            }
            CryptReleaseContext(hProv, 0);
        }

        char base64Str[128] = {0};
        DWORD b64Len = sizeof(base64Str);
        CryptBinaryToStringA(sha1Hash, 20, CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, base64Str, &b64Len);
        std::string result(base64Str);
        // Strip trailing \r or \n if present
        result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
        result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
        return result;
    }
};

TelemetryServer::TelemetryServer(int port) : port_(port), isRunning_(false), impl_(new Impl()) {}

TelemetryServer::~TelemetryServer() {
    stop();
    delete impl_;
}

bool TelemetryServer::start() {
    std::cout << "[TelemetryServer] Starting WebSocket Broadcast Server on port " << port_ << "..." << std::endl;

#ifdef _WIN32
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        std::cerr << "[TelemetryServer] WSAStartup failed: " << res << std::endl;
        return false;
    }
#endif

    impl_->listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (impl_->listenSocket == INVALID_SOCKET) {
        std::cerr << "[TelemetryServer] Failed to create socket." << std::endl;
        return false;
    }

    int opt = 1;
    setsockopt(impl_->listenSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in service;
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    service.sin_port = htons(port_);

    if (bind(impl_->listenSocket, (SOCKADDR*)&service, sizeof(service)) == SOCKET_ERROR) {
        std::cerr << "[TelemetryServer] Bind failed on port " << port_ << std::endl;
        closesocket(impl_->listenSocket);
        impl_->listenSocket = INVALID_SOCKET;
        return false;
    }

    if (listen(impl_->listenSocket, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "[TelemetryServer] Listen failed." << std::endl;
        closesocket(impl_->listenSocket);
        impl_->listenSocket = INVALID_SOCKET;
        return false;
    }

    isRunning_ = true;
    impl_->serverThread = std::thread(&TelemetryServer::serverLoop, this);
    std::cout << "[TelemetryServer] Listening for WebSocket clients on ws://localhost:" << port_ << std::endl;
    return true;
}

void TelemetryServer::stop() {
    if (!isRunning_) return;
    isRunning_ = false;

    // The worker owns socket I/O. Join before closing its descriptors.
    if (impl_->serverThread.joinable()) impl_->serverThread.join();

    if (impl_->listenSocket != INVALID_SOCKET) {
        closesocket(impl_->listenSocket);
        impl_->listenSocket = INVALID_SOCKET;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->socketMutex);
        for (SOCKET s : impl_->clientSockets) {
            if (s != INVALID_SOCKET) {
                closesocket(s);
            }
        }
        impl_->clientSockets.clear();
    }

#ifdef _WIN32
    WSACleanup();
#endif
}

void TelemetryServer::serverLoop() {
    while (isRunning_) {
        fd_set readfds, writefds;
        FD_ZERO(&writefds);
        FD_ZERO(&readfds);
        FD_SET(impl_->listenSocket, &readfds);

        std::vector<SOCKET> currentClients;
        {
            std::lock_guard<std::mutex> lock(impl_->socketMutex);
            currentClients = impl_->clientSockets;
            for (SOCKET client : currentClients) {
                auto& pending = impl_->outgoing[client];
                if (!pending.packet && pending.sequence != impl_->sequence) {
                    pending.packet = impl_->latest; pending.sequence = impl_->sequence; pending.offset = 0;
                }
                if (pending.packet) FD_SET(client, &writefds);
            }
        }

        for (SOCKET s : currentClients) {
            FD_SET(s, &readfds);
        }

        timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 2000; // 2ms network service interval

        int activity = select(0, &readfds, &writefds, NULL, &timeout);
        if (activity <= 0) {
            continue;
        }

        // 1. New incoming connection on listenSocket
        if (FD_ISSET(impl_->listenSocket, &readfds)) {
            SOCKET client = accept(impl_->listenSocket, NULL, NULL);
            if (client != INVALID_SOCKET) {
                std::cout << "[TelemetryServer] Incoming TCP connection accepted." << std::endl;

                // Bound an incomplete HTTP handshake; it runs only on the network thread.
                DWORD receiveTimeout = 1000;
                setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&receiveTimeout), sizeof(receiveTimeout));
                char buffer[4096] = {0};
                int bytesRead = recv(client, buffer, sizeof(buffer) - 1, 0);
                if (bytesRead > 0) {
                    std::string req(buffer, bytesRead);
                    std::string lowerReq = impl_->toLower(req);

                    size_t keyPos = lowerReq.find("sec-websocket-key:");
                    if (keyPos != std::string::npos) {
                        size_t valStart = keyPos + 18;
                        while (valStart < req.length() && (req[valStart] == ' ' || req[valStart] == '\t')) {
                            valStart++;
                        }
                        size_t keyEnd = req.find_first_of("\r\n", valStart);
                        if (keyEnd == std::string::npos) keyEnd = req.length();

                        std::string clientKey = req.substr(valStart, keyEnd - valStart);
                        std::string acceptKey = impl_->computeAcceptKey(clientKey);

                        std::stringstream response;
                        response << "HTTP/1.1 101 Switching Protocols\r\n"
                                 << "Upgrade: websocket\r\n"
                                 << "Connection: Upgrade\r\n"
                                 << "Sec-WebSocket-Accept: " << acceptKey << "\r\n\r\n";

                        std::string respStr = response.str();
                        send(client, respStr.c_str(), (int)respStr.length(), 0);
                        std::cout << "[TelemetryServer] WebSocket Handshake successful!" << std::endl;

                        {
                            std::lock_guard<std::mutex> lock(impl_->socketMutex);
                            u_long nonblocking = 1; ioctlsocket(client, FIONBIO, &nonblocking);
                            int yes = 1; setsockopt(client, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&yes), sizeof(yes));
                            impl_->clientSockets.push_back(client);
                        }
                    } else {
                        std::cerr << "[TelemetryServer] Handshake failed: Sec-WebSocket-Key header missing." << std::endl;
                        closesocket(client);
                    }
                } else {
                    closesocket(client);
                }
            }
        }

        // 2. Incoming frame data or close from existing clients
        std::vector<SOCKET> socketsToRemove;
        for (SOCKET client : currentClients) {
            if (FD_ISSET(client, &readfds)) {
                uint8_t inBuffer[4096] = {0};
                int cBytes = recv(client, (char*)inBuffer, sizeof(inBuffer), 0);
                if (cBytes <= 0) {
                    std::cout << "[TelemetryServer] Client connection lost." << std::endl;
                    socketsToRemove.push_back(client);
                    continue;
                }

                auto& received = impl_->incoming[client];
                received.append(reinterpret_cast<char*>(inBuffer), cBytes);
                while (received.size() >= 2) {
                    auto byte = [&](size_t n){ return static_cast<uint8_t>(received[n]); };
                    const auto opcode = byte(0) & 15;
                    size_t length = byte(1) & 127, offset = 2;
                    if (!(byte(0)&128) || !(byte(1)&128) || length == 127) { socketsToRemove.push_back(client); break; }
                    if (length == 126) { if(received.size()<4) break; length=(byte(2)<<8)|byte(3); offset=4; }
                    if (length>4096) { socketsToRemove.push_back(client); break; }
                    if (received.size()<offset+4+length) break;
                    std::string msg(length, '\0');
                    for(size_t i=0;i<length;++i) msg[i]=byte(offset+4+i)^byte(offset+i%4);
                    received.erase(0,offset+4+length);
                    if(opcode==8) { socketsToRemove.push_back(client); break; }
                    if(opcode==1) handleMessage(msg);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(impl_->socketMutex);
            for (SOCKET client : currentClients) {
                if (!FD_ISSET(client, &writefds)) continue;
                auto& pending = impl_->outgoing[client];
                if (!pending.packet) continue;
                if (pending.offset == 0 && pending.sequence != impl_->sequence) {
                    pending.packet=impl_->latest; pending.sequence=impl_->sequence;
                }
                int sent=send(client, reinterpret_cast<const char*>(pending.packet->data()+pending.offset),
                    static_cast<int>(pending.packet->size()-pending.offset), 0);
                if(sent>0) {pending.offset+=sent; if(pending.offset==pending.packet->size()) {pending.packet.reset(); pending.offset=0;}}
                else if(sent==0 || WSAGetLastError()!=WSAEWOULDBLOCK) socketsToRemove.push_back(client);
            }
        }
        if (!socketsToRemove.empty()) {
            std::lock_guard<std::mutex> lock(impl_->socketMutex);
            std::sort(socketsToRemove.begin(), socketsToRemove.end());
            socketsToRemove.erase(std::unique(socketsToRemove.begin(), socketsToRemove.end()), socketsToRemove.end());
            for (SOCKET deadSocket : socketsToRemove) {
                impl_->outgoing.erase(deadSocket); impl_->incoming.erase(deadSocket);
                closesocket(deadSocket);
                auto it = std::find(impl_->clientSockets.begin(), impl_->clientSockets.end(), deadSocket);
                if (it != impl_->clientSockets.end()) {
                    impl_->clientSockets.erase(it);
                }
            }
            if (impl_->clientSockets.empty()) {
                handleClose();
            }
        }
    }
}

std::string TelemetryServer::serializeToJSON(int frameId, const std::vector<TrackedObject>& tracks, float fps, const std::string& mode, const PhysicsState* pState, float totalMs, float inferMs, const std::vector<std::pair<float, float>>* corners, const std::string& sourceVideo, const FramePresentation* presentation) {
    std::stringstream ss;

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    int strokeCount = pState ? pState->strokeCount : 0;
    float maxVelocity = pState ? pState->maxVelocityKmh : 0.0f;
    float elapsedTime = pState ? pState->elapsedTimeSec : 0.0f;
    bool isStrike = pState ? pState->isStrike : false;
    bool isBounce = pState ? pState->isTableBounce : false;
    float speedKmh = pState ? pState->velocityKmh : 0.0f;
    float zHeight = 0.0f; // Airborne height is unobserved; no synthetic height in telemetry.

    float bounceX = pState ? pState->bounceX : -1.0f;
    float bounceY = pState ? pState->bounceY : -1.0f;

    bool hasLiveCorners = corners && corners->size() == 4;

    ss << std::fixed << std::setprecision(4);
    ss << "{\n";
    static const auto root=[] {if(const char* data=std::getenv("TT_DATA_ROOT"))return std::filesystem::absolute(std::filesystem::u8path(data)).u8string();auto path=std::filesystem::current_path();for(int i=0;i<5&&!std::filesystem::is_directory(path/"backend_cpp");++i)path=path.parent_path();return path.u8string();}();
    ss << "\"service\":{\"application\":\"TT_TRAJECTORY_ENGINE\",\"mode\":"<<recording::quote(mode)<<",\"pid\":"<<GetCurrentProcessId()<<",\"root\":"<<recording::quote(root)<<"},\n";
    ss << "  \"mode\": \"" << mode << "\",\n";
    ss << "  \"source_video\": " << std::quoted(sourceVideo) << ",\n";
    ss << "  \"bounce_detector_status\": \"EXPERIMENTAL (~15.5% F1 - Mode B Gated Raw Top-1)\",\n";
    ss << "  \"table_calibration_status\": \"" << (hasLiveCorners ? "MODEL_TABLE_WITH_CONTOUR_REFINEMENT" : "UNAVAILABLE") << "\",\n";
    ss << "  \"frame_id\": " << frameId << ",\n";
    ss << "  \"timestamp\": " << ms << ",\n";
    ss << "  \"fps\": " << fps << ",\n";
    ss << "  \"latency_ms\": " << totalMs << ",\n";
    ss << "  \"infer_ms\": " << inferMs << ",\n";
    ss << "  \"z\": null,\n";
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

    const TrackedObject* ballTrack = nullptr;
    for (const auto& t : tracks) {
        if (t.class_id == 0 && t.is_active && t.age == 0 &&
            (!ballTrack || t.confidence > ballTrack->confidence)) {
            ballTrack = &t;
        }
    }

    TrackedObject measured{};
    if(presentation) {
        ballTrack=nullptr;
        if(presentation->measuredBall) {
            const auto& d=presentation->observedBall;
            measured.x=d.x;measured.y=d.y;measured.w=d.w;measured.h=d.h;measured.confidence=d.confidence;
            ballTrack=&measured;
        }
    }
    if (ballTrack) {
        float vx_ms = presentation ? presentation->imageVx : ballTrack->vx * fps;
        float vy_ms = presentation ? presentation->imageVy : ballTrack->vy * fps;

        ss << "  \"ball\": {\n";
        ss << "    \"x\": " << ballTrack->x + ballTrack->w / 2.0f << ",\n";
        ss << "    \"y\": " << ballTrack->y + ballTrack->h / 2.0f << ",\n";
        ss << "    \"z\": null,\n";
        ss << "    \"vx\": " << vx_ms << ",\n";
        ss << "    \"vy\": " << vy_ms << ",\n";
        ss << "    \"vz\": null,\n";
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

    const auto spin=presentation?presentation->observedSpin:ObservedSpin{};
    ss << "  \"spin\": {\"rpm\":";
    if(spin.available)ss<<spin.rpm;else ss<<"null";
    ss << ",\"status\":\""<<(spin.available?"estimated":"unavailable")<<"\",\"type\":\"camera_axis\",\"axis_deg\":null,\"quality\":"<<spin.quality
       <<",\"reason\":"<<std::quoted(spin.reason)<<"},\n";

    ss << "  \"players\": [],\n";

    if (hasLiveCorners) {
        ss << "  \"table_corners\": [[" << (*corners)[0].first << ", " << (*corners)[0].second << "], ["
           << (*corners)[1].first << ", " << (*corners)[1].second << "], ["
           << (*corners)[2].first << ", " << (*corners)[2].second << "], ["
           << (*corners)[3].first << ", " << (*corners)[3].second << "]],\n";
    } else {
        ss << "  \"table_corners\": [],\n";
    }

    ss << "  \"rally\": {\n";
    ss << "    \"in_progress\": " << (strokeCount > 0 ? "true" : "false") << ",\n";
    ss << "    \"stroke_count\": " << strokeCount << ",\n";
    ss << "    \"rally_strokes\": " << strokeCount << ",\n";
    ss << "    \"rally_duration_s\": " << elapsedTime << ",\n";
    ss << "    \"max_speed_kmh\": " << maxVelocity << "\n";
    ss << "  }\n";

    if (presentation) {
        if(!presentation->intelligenceJson.empty())ss<<",\"intelligence\":"<<presentation->intelligenceJson;
        ss << ",\"source_id\":"<<presentation->sourceId
           <<",\"ball_observation\":{\"state\":"<<std::quoted(presentation->ballObservationState)<<",\"reason\":"<<std::quoted(presentation->ballObservationReason)<<"}"
           <<",\"scene_frame_id\":"<<presentation->sceneFrameId
           <<",\"position_basis\":\"table_plane_projection\",\"height_observed\":false"
           <<",\"video_picker_open\":"<<(presentation->videoPickerOpen?"true":"false")
           <<",\"video_error\":"<<std::quoted(presentation->videoError);
        ss << ",\"strikes\":[";
        for(size_t i=0;i<presentation->strikes.size();++i) {
            const auto& strike=presentation->strikes[i];if(i)ss<<",";
            ss<<"{\"id\":"<<strike.id<<",\"frame_id\":"<<strike.frame<<",\"source_time\":"<<strike.time<<",\"speed_kmh\":"<<strike.speedKmh<<"}";
        }
        ss<<"]";
        ss << ",\n  \"preview_jpeg\": " << std::quoted(presentation->jpegBase64)
           << ",\n  \"source_width\": " << presentation->width
           << ",\n  \"source_height\": " << presentation->height
           << ",\n  \"source_time\": " << presentation->sourceTime
           << ",\n  \"total_frames\": " << presentation->totalFrames
           << ",\n  \"playback_rate\": " << presentation->playbackRate
           << ",\n  \"processing_fps\": " << presentation->processingFps
           << ",\n  \"preview_fps\": " << presentation->previewFps
           << ",\n  \"skipped_frames\": " << presentation->skippedFrames
           << ",\n  \"camera_active\": " << (presentation->cameraActive ? "true" : "false")
           << ",\n  \"camera_index\": " << presentation->cameraIndex
           << ",\n  \"training_recording\": {\"active\":" << (presentation->recordingActive?"true":"false")
           << ",\"failed\":" << (presentation->recordingFailed?"true":"false")
           << ",\"saved_frames\":" << presentation->recordingSaved
           << ",\"dropped_frames\":" << presentation->recordingDropped
           << ",\"clip\":\"" << presentation->recordingClip << "\"}"
           << ",\n  \"paused\": " << (presentation->paused ? "true" : "false")
           << ",\n  \"ended\": " << (presentation->ended ? "true" : "false");
        if (hasLiveCorners && ballTrack && presentation->positionValid) ss << ",\n  \"table_position\": [" << presentation->tableX << "," << presentation->tableY << "]";
        ss << ",\n  \"scene_detections\": [";
        bool first=true;
        for (const auto& d: presentation->scene) {
            if(!first) ss << ","; first=false;
            ss << "{\"class_id\":" << d.class_id << ",\"confidence\":" << d.confidence
               << ",\"bbox\":[" << d.x << "," << d.y << "," << d.w << "," << d.h << "]}";
        }
        ss << "]\n";
        ss << ",\"suppressed_scene_detections\":[";
        for(size_t i=0;i<presentation->suppressedPersonParts.size();++i){
            if(i)ss<<",";const auto& d=presentation->suppressedPersonParts[i];
            ss<<"{\"class_id\":"<<d.class_id<<",\"confidence\":"<<d.confidence<<",\"reason\":\"CONTAINED_LOWER_PERSON_FRAGMENT\",\"bbox\":["<<d.x<<","<<d.y<<","<<d.w<<","<<d.h<<"]}";
        }
        ss<<"]";
        ss << ",\"impact_points\":[";
        for(size_t i=0;i<presentation->impacts.size();++i) {
            if(i)ss<<",";ss<<"["<<presentation->impacts[i].first<<","<<presentation->impacts[i].second<<"]";
        }
        ss << "],\"image_impact_points\":[";
        for(size_t i=0;i<presentation->imageImpacts.size();++i) {
            if(i)ss<<",";ss<<"["<<presentation->imageImpacts[i].first<<","<<presentation->imageImpacts[i].second<<"]";
        }
        ss << "]\n";
    }
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
    if(message=="SHUTDOWN") {cameraAuthorized=false;trainingRecordingRequested=false;shutdownRequested=true;}
    else if (message == "START_CAMERA") cameraAuthorized = true;
    else if (message == "STOP_CAMERA") {cameraAuthorized = false;trainingRecordingRequested=false;}
    else if (message == "START_TRAINING_RECORDING" && cameraAuthorized) {activeLearningRequested=false;trainingBurstRequested=false;trainingRecordingRequested=true;}
    else if (message == "START_TEMPORAL_RECORDING" && cameraAuthorized) {activeLearningRequested=false;trainingBurstRequested=true;trainingRecordingRequested=true;}
    else if (message == "START_ACTIVE_LEARNING" && cameraAuthorized) {activeLearningRequested=true;trainingBurstRequested=false;trainingRecordingRequested=true;}
    else if (message == "STOP_TRAINING_RECORDING") trainingRecordingRequested=false;
    else if (message == "OPEN_VIDEO_DIALOG") videoPickerRequested=true;
    else if (message.rfind("LOAD_VIDEO:",0)==0) {std::lock_guard<std::mutex> lock(videoMutex_);pendingVideo_=message.substr(11);}
    else if (message == "PAUSE") playbackPaused = true;
    else if (message == "PLAY") playbackPaused = false;
    else if (message == "RESTART") { restartRequested = true; playbackPaused = false; }
    else if(message=="STEP_FORWARD") {frameStepRequested=1;playbackPaused=true;}
    else if(message=="STEP_BACK") {frameStepRequested=-1;playbackPaused=true;}
    else if(message.rfind("SEEK_FRAME:",0)==0){
        try{size_t used=0;auto text=message.substr(11);int frame=std::stoi(text,&used);if(used!=text.size()||frame<1)throw std::runtime_error("Invalid frame");seekFrameRequested=frame;}
        catch(const std::exception& e){std::cerr<<"[WebSocket] Rejected seek: "<<e.what()<<std::endl;}
    }
    else if (message.rfind("SET_CAMERA:",0)==0 || message.rfind("PLAYBACK_RATE:",0)==0) {
        try {
            const bool camera=message.rfind("SET_CAMERA:",0)==0;
            const auto value=message.substr(camera ? 11 : 14);
            size_t used=0; float number=std::stof(value,&used);
            if(used!=value.size() || !std::isfinite(number)) throw std::runtime_error("Invalid number");
            if(camera && number>=0 && number<=9 && number==std::floor(number)) {targetCameraIndex=static_cast<int>(number);trainingRecordingRequested=false;}
            else if(!camera && number>=.25f && number<=4.f) playbackRate=number;
            else throw std::runtime_error("Out of range");
        } catch(const std::exception& e) { std::cerr << "[WebSocket] Rejected " << message << ": " << e.what() << std::endl; }
    }
}

std::string TelemetryServer::takeVideoRequest() {
    std::lock_guard<std::mutex> lock(videoMutex_);
    auto path=std::move(pendingVideo_);pendingVideo_.clear();return path;
}

void TelemetryServer::handleClose() {
    cameraAuthorized = false;
    trainingRecordingRequested=false;
    std::cout << "[WebSocket] Client disconnected. Failsafe revoking hardware camera capture." << std::endl;
}

void TelemetryServer::broadcastFrame(int frameId, const std::vector<TrackedObject>& trackedObjects, float fps, const std::string& mode, const PhysicsState* pState, float totalMs, float inferMs, const std::vector<std::pair<float, float>>* corners, const std::string& sourceVideo, const FramePresentation* presentation) {
    broadcastJSON(serializeToJSON(frameId, trackedObjects, fps, mode, pState, totalMs, inferMs, corners, sourceVideo, presentation));
}

void TelemetryServer::broadcastJSON(const std::string& jsonStr) {
    if (!isRunning_) return;

    std::lock_guard<std::mutex> lock(impl_->socketMutex);
    if (impl_->clientSockets.empty()) return;

    size_t len = jsonStr.length();

    std::vector<uint8_t> frame;
    frame.push_back(0x81); // Text frame FIN

    if (len <= 125) {
        frame.push_back(static_cast<uint8_t>(len));
    } else if (len <= 65535) {
        frame.push_back(126);
        frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        frame.push_back(static_cast<uint8_t>(len & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--) {
            frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xFF));
        }
    }

    frame.insert(frame.end(), jsonStr.begin(), jsonStr.end());

    impl_->latest = std::make_shared<std::vector<uint8_t>>(std::move(frame));
    ++impl_->sequence;
}
