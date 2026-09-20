#ifndef BYTETRACK_HPP
#define BYTETRACK_HPP

#include "tensorrt_engine.hpp"
#include <vector>

struct TrackedObject {
    int track_id;
    int class_id;
    float confidence;
    float x, y, w, h;
    float vx, vy;        // Estimated velocity vector
    int age;
    bool is_active;
};

class ByteTracker {
public:
    ByteTracker(float highScoreThresh = 0.6f, float lowScoreThresh = 0.1f, int maxAge = 30);
    ~ByteTracker();

    std::vector<TrackedObject> update(const std::vector<Detection>& detections);

private:
    float highThresh_;
    float lowThresh_;
    int maxAge_;
    int nextTrackId_;

    std::vector<TrackedObject> activeTracks_;
    
    static float computeIoU(const Detection& a, const TrackedObject& b);
};

#endif // BYTETRACK_HPP
