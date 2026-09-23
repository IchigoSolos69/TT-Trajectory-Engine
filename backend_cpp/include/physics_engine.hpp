#pragma once
#include <cmath>
#include <vector>

struct PhysicsState {
    float velocityKmh;
    float zHeight;          // Estimated ballistic elevation (meters)
    bool isStrike;          // Vision-based paddle deflection
    bool isTableBounce;     // Vision-detected table bounce
    float bounceX;          // Impact X position in physical table meters
    float bounceY;          // Impact Y position in physical table meters
    int strokeCount;        // Server-authoritative cumulative rally stroke count
    float maxVelocityKmh;   // Server-authoritative peak velocity in km/h
    float elapsedTimeSec;   // Server-authoritative active rally flight duration
    float spinRpm;          // Aerodynamic Magnus model estimated spin RPM
};

class PhysicsEngine {
public:
    PhysicsEngine();
    PhysicsState update(float x, float y, float deltaTime, float pixelX = -1.0f, float pixelY = -1.0f);
    void reset();
    void resetRally();

private:
    float lastX, lastY;
    float lastPixelY;       // Last 2D image pixel Y coordinate
    float lastPixelVy;      // Velocity in 2D image pixel Y (pixels/sec)
    float smVy;             // Exponentially smoothed Y-axis velocity (m/s)
    float lastValidSpeedKmh;// Last accepted plausible speed
    float velocityZ;        // Vertical Z-axis velocity (m/s)
    float currentZ;         // Elevation above table (meters)
    bool hasLast;           // Tracking continuity flag
    int cooldown;           // Refractory period for strike detection (frames)
    int bounceCooldown;     // Refractory period for vision bounce detection (frames)
    bool bounceSignalArmed_; // Require trajectory signal to re-arm before next bounce trigger
    int dirSign;            // Latched longitudinal direction sign (+1, -1, or 0)
    int trackingLossFrames; // Frames elapsed without valid tracking sample
    int consecutiveTrackFrames; // Consecutive valid frames with non-noise tracking
    std::vector<float> vyHistory; // History of 2D vertical pixel velocities

    // Peak-picking state buffer
    bool candidateBounceActive_;
    int candidateBounceFrame_;
    float candidateMaxPixelY_;
    float candidateBounceX_;
    float candidateBounceY_;
    int candidateWaitCounter_;

    // Server-authoritative cumulative counters
    int strokeCount;
    float maxVelocityKmh;
    float elapsedTimeSec;
    bool rallyActive;       // True when an active rally is in progress
};

