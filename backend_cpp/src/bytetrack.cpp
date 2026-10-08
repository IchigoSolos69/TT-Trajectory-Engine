#include "bytetrack.hpp"
#include <algorithm>
#include <cmath>

ByteTracker::ByteTracker(float highScoreThresh, float lowScoreThresh, int maxAge)
    : highThresh_(highScoreThresh), lowThresh_(lowScoreThresh), maxAge_(maxAge), nextTrackId_(1) {}

ByteTracker::~ByteTracker() {}

float ByteTracker::computeIoU(const Detection& a, const TrackedObject& b) {
    // Detector boxes use normalized top-left coordinates, not centre coordinates.
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    float intersectionW = std::max(0.0f, x2 - x1);
    float intersectionH = std::max(0.0f, y2 - y1);
    float intersectionArea = intersectionW * intersectionH;

    float areaA = a.w * a.h;
    float areaB = b.w * b.h;
    float unionArea = areaA + areaB - intersectionArea;

    if (unionArea <= 0.0f) return 0.0f;
    return intersectionArea / unionArea;
}

std::vector<TrackedObject> ByteTracker::update(const std::vector<Detection>& detections) {
    // 1. Separate detections into high confidence and low confidence sets
    std::vector<Detection> highDets;
    std::vector<Detection> lowDets;

    for (const auto& det : detections) {
        if (det.confidence >= highThresh_) {
            highDets.push_back(det);
        } else if (det.confidence >= lowThresh_) {
            lowDets.push_back(det);
        }
    }

    // 2. Constant-velocity prediction for existing tracks.
    for (auto& track : activeTracks_) {
        track.x += track.vx;
        track.y += track.vy;
        track.age++;
    }

    // 3. First association: Match highDets with activeTracks via IoU
    std::vector<bool> detMatched(highDets.size(), false);
    std::vector<bool> trackMatched(activeTracks_.size(), false);

    for (size_t i = 0; i < highDets.size(); ++i) {
        float bestIoU = 0.2f; // Min IoU threshold
        int bestTrackIdx = -1;

        for (size_t j = 0; j < activeTracks_.size(); ++j) {
            if (trackMatched[j] || highDets[i].class_id != activeTracks_[j].class_id) continue;
            float iou = computeIoU(highDets[i], activeTracks_[j]);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestTrackIdx = (int)j;
            }
        }

        if (bestTrackIdx >= 0) {
            // Update Track
            auto& t = activeTracks_[bestTrackIdx];
            // Residual is relative to the predicted position. Preserve velocity
            // when a detection lands exactly on the prediction, including gaps.
            t.vx += (highDets[i].x - t.x) / std::max(1, t.age);
            t.vy += (highDets[i].y - t.y) / std::max(1, t.age);
            t.x = highDets[i].x;
            t.y = highDets[i].y;
            t.w = highDets[i].w;
            t.h = highDets[i].h;
            t.confidence = highDets[i].confidence;
            t.age = 0;

            detMatched[i] = true;
            trackMatched[bestTrackIdx] = true;
        }
    }

    // Second association: low-confidence observations may maintain an existing
    // identity but cannot create a new track.
    for (const auto& detection : lowDets) {
        int best = -1;
        float bestIoU = .2f;
        for (size_t j = 0; j < activeTracks_.size(); ++j) {
            if (trackMatched[j] || detection.class_id != activeTracks_[j].class_id) continue;
            const float overlap = computeIoU(detection, activeTracks_[j]);
            if (overlap > bestIoU) { bestIoU = overlap; best = static_cast<int>(j); }
        }
        if (best < 0) continue;
        auto& t = activeTracks_[best];
        t.vx += (detection.x-t.x)/std::max(1,t.age);
        t.vy += (detection.y-t.y)/std::max(1,t.age);
        t.x=detection.x;t.y=detection.y;t.w=detection.w;t.h=detection.h;
        t.confidence=detection.confidence;t.age=0;trackMatched[best]=true;
    }

    // 4. Create new tracks for unmatched high-confidence detections
    for (size_t i = 0; i < highDets.size(); ++i) {
        if (!detMatched[i]) {
            TrackedObject newTrack;
            newTrack.track_id = nextTrackId_++;
            newTrack.class_id = highDets[i].class_id;
            newTrack.confidence = highDets[i].confidence;
            newTrack.x = highDets[i].x;
            newTrack.y = highDets[i].y;
            newTrack.w = highDets[i].w;
            newTrack.h = highDets[i].h;
            newTrack.vx = 0.0f;
            newTrack.vy = 0.0f;
            newTrack.age = 0;
            newTrack.is_active = true;

            activeTracks_.push_back(newTrack);
        }
    }

    // 5. Remove lost tracks exceeding maxAge
    activeTracks_.erase(
        std::remove_if(activeTracks_.begin(), activeTracks_.end(),
                       [this](const TrackedObject& t) { return t.age > maxAge_; }),
        activeTracks_.end());

    return activeTracks_;
}
