#include "physics_engine.hpp"
#include <algorithm>
#include <cmath>

PhysicsEngine::PhysicsEngine() {
    reset();
    resetRally();
}

void PhysicsEngine::reset() {
    lastX = 0.0f;
    lastY = 0.0f;
    lastPixelY = -1.0f;
    lastPixelVy = 0.0f;
    smVy = 0.0f;
    lastValidSpeedKmh = 0.0f;
    velocityZ = 0.0f;
    currentZ = 0.15f; // Initial elevation above table (0.15m)
    hasLast = false;
    dirSign = 0;
    cooldown = 0;
    bounceCooldown = 0;
    bounceSignalArmed_ = true;
    consecutiveTrackFrames = 0;
    candidateBounceActive_ = false;
    candidateBounceFrame_ = 0;
    candidateMaxPixelY_ = -1.0f;
    candidateBounceX_ = 0.0f;
    candidateBounceY_ = 0.0f;
    candidateWaitCounter_ = 0;
    vyHistory.clear();
}

void PhysicsEngine::resetRally() {
    strokeCount = 0;
    maxVelocityKmh = 0.0f;
    elapsedTimeSec = 0.0f;
    dirSign = 0;
    smVy = 0.0f;
    cooldown = 0;
    bounceCooldown = 0;
    bounceSignalArmed_ = true;
    trackingLossFrames = 0;
    consecutiveTrackFrames = 0;
    candidateBounceActive_ = false;
    candidateBounceFrame_ = 0;
    candidateMaxPixelY_ = -1.0f;
    candidateBounceX_ = 0.0f;
    candidateBounceY_ = 0.0f;
    candidateWaitCounter_ = 0;
    rallyActive = false;
    vyHistory.clear();
}

PhysicsState PhysicsEngine::update(float x, float y, float deltaTime, float pixelX, float pixelY) {
    PhysicsState state;
    state.isStrike = false;
    state.isTableBounce = false;
    state.bounceX = -1.0f;
    state.bounceY = -1.0f;

    // Preserve actual camera/sample timing down to 4 FPS. Main resets continuity
    // for long gaps; forcing a 15 FPS camera to 1/30s doubled estimated speeds.
    float dt = std::max(1.0f / 240.0f, std::min(0.25f, deltaTime));

    // 1. First-frame / tracking re-entry check
    if (!hasLast || pixelY < 0.0f) {
        lastX = x;
        lastY = y;
        lastPixelY = pixelY;
        smVy = 0.0f;
        lastValidSpeedKmh = 0.0f;
        hasLast = (pixelY >= 0.0f);
        consecutiveTrackFrames = 0;

        state.velocityKmh = 0.0f;
        state.zHeight = currentZ;
        state.strokeCount = strokeCount;
        state.maxVelocityKmh = maxVelocityKmh;
        state.elapsedTimeSec = elapsedTimeSec;
        state.spinRpm = 0.0f;
        return state;
    }

    // 2. Tracking Timeout & Rally Reset Check
    trackingLossFrames = 0;
    consecutiveTrackFrames++;

    // Accumulate total flight time for active rally
    if (rallyActive || strokeCount > 0) {
        elapsedTimeSec += dt;
        rallyActive = true;
    }

    // 3. Compute 2D displacement and physical velocities (m/s)
    float dx = x - lastX;
    float dy = y - lastY;

    float vx = dx / dt;
    float vy = dy / dt; // Longitudinal velocity along table length (Y-axis)
    float rawSpeedMs = std::sqrt(vx * vx + vy * vy);
    float rawSpeedKmh = rawSpeedMs * 3.6f;

    // Reject implausible tracking jumps (> 150 km/h)
    if (rawSpeedKmh > 150.0f) {
        hasLast = false;
        consecutiveTrackFrames = 0;
        state.velocityKmh = lastValidSpeedKmh;
        state.zHeight = currentZ;
        state.strokeCount = strokeCount;
        state.maxVelocityKmh = maxVelocityKmh;
        state.elapsedTimeSec = elapsedTimeSec;
        state.spinRpm = 0.0f;
        return state;
    }

    // Valid frame sample
    lastX = x;
    lastY = y;
    lastValidSpeedKmh = rawSpeedKmh;
    state.velocityKmh = rawSpeedKmh;

    if (rawSpeedKmh > maxVelocityKmh) {
        maxVelocityKmh = rawSpeedKmh;
    }

    // 4. Vision-Based 2D Pixel Discontinuity Table Bounce Detector with Peak Picking & Gating
    if (bounceCooldown > 0) bounceCooldown--;

    if (pixelY >= 0.0f && lastPixelY >= 0.0f) {
        float pixelVy = (pixelY - lastPixelY) / dt; // positive = downward in camera view
        float pixelAccelSpike = (lastPixelVy - pixelVy) / dt;

        // Sanity check on pixel velocity (ignore noise jumps > 4000 px/s)
        if (std::abs(pixelVy) < 4000.0f) {
            vyHistory.push_back(pixelVy);
            if (vyHistory.size() > 4) {
                vyHistory.erase(vyHistory.begin());
            }

            // Gating 1: Table ROI Check (in normalized 3D table space or 1080p pixel space)
            bool isTableRectangle = (x >= -0.1f && x <= 1.625f && y >= -0.1f && y <= 2.84f);
            // Gating 2: Minimum Track Continuity (require >= 1 tracked frame so motion-blur dropouts at impact don't kill recall)
            bool hasTrackContinuity = (consecutiveTrackFrames >= 1);

            float maxInbound = -9999.0f;
            for (size_t i = 0; i + 1 < vyHistory.size(); ++i) {
                if (vyHistory[i] > maxInbound) maxInbound = vyHistory[i];
            }
            float deltaVy = maxInbound - pixelVy;

            // Suppress table bounce trigger if paddle strike occurred recently
            bool strikeActive = (cooldown >= 11);

            bool conditionSignFlip = (lastPixelVy > 80.0f && pixelVy < -20.0f && pixelAccelSpike > 1500.0f);
            bool conditionWindowed = (maxInbound > 60.0f && pixelVy <= 0.0f && deltaVy > 400.0f);

            // Candidate Peak-Picking: Buffer turning point and pick local maximum Y (bottom-most point of arc)
            if (bounceCooldown == 0 && bounceSignalArmed_ && !strikeActive && vyHistory.size() >= 3 && isTableRectangle && hasTrackContinuity) {
                if (conditionSignFlip || conditionWindowed) {
                    if (!candidateBounceActive_) {
                        candidateBounceActive_ = true;
                        candidateMaxPixelY_ = pixelY;
                        candidateBounceX_ = x;
                        candidateBounceY_ = y;
                        candidateWaitCounter_ = 2; // Buffer 2 additional frames to find true turning point peak
                    } else if (pixelY > candidateMaxPixelY_) {
                        // Found lower point (higher Y on screen) during turning window
                        candidateMaxPixelY_ = pixelY;
                        candidateBounceX_ = x;
                        candidateBounceY_ = y;
                    }
                }
            }

            // Process active candidate window
            if (candidateBounceActive_) {
                if (pixelY > candidateMaxPixelY_) {
                    candidateMaxPixelY_ = pixelY;
                    candidateBounceX_ = x;
                    candidateBounceY_ = y;
                }

                candidateWaitCounter_--;
                if (candidateWaitCounter_ <= 0) {
                    // Fire Peak-Picked Bounce!
                    state.isTableBounce = true;
                    state.bounceX = candidateBounceX_;
                    state.bounceY = candidateBounceY_;
                    currentZ = 0.02f; // Re-anchor elevation at physical impact site
                    velocityZ = std::abs(velocityZ) * 0.85f; // Table surface restitution
                    bounceCooldown = 15; // 15-frame refractory period
                    bounceSignalArmed_ = false; // Disarm until trajectory signal clears
                    candidateBounceActive_ = false;
                }
            }

            // Re-arm signal once vertical trajectory is no longer meeting trigger conditions
            if (!conditionSignFlip && !conditionWindowed && !candidateBounceActive_) {
                bounceSignalArmed_ = true;
            }
            lastPixelVy = pixelVy;
        }
    }
    lastPixelY = pixelY;

    // 5. Direction Sign-Reversal Paddle Strike Detector with Dead-Band Hysteresis
    smVy = 0.5f * smVy + 0.5f * vy; // Exponential smoothing (m/s)
    constexpr float kMin = 1.5f;   // Dead-band threshold (~5.4 km/h)
    int currentSign = (smVy > kMin) - (smVy < -kMin); // +1, 0, or -1

    if (cooldown > 0) {
        cooldown--;
    }

    if (currentSign != 0) {
        if (dirSign != 0 && currentSign != dirSign && cooldown == 0) {
            state.isStrike = true;
            strokeCount++;
            rallyActive = true;
            velocityZ = std::max(velocityZ + 2.5f, 3.2f); // Paddle strike vertical lift boost
            cooldown = 15; // 15-frame refractory period (~125ms at 120 FPS)
        }
        dirSign = currentSign; // Update latched direction only on confident reading
    }

    // 6. Parabolic Z-Axis Gravity Simulation (-9.81 m/s^2)
    velocityZ -= 9.81f * dt;
    currentZ += velocityZ * dt;

    // Floor/Table surface fallback elevation decay at z <= 0.02m (pure elevation floor, no heatmap emission)
    if (currentZ <= 0.02f) {
        currentZ = 0.02f;
        // Table surface restitution e = 0.85
        velocityZ = std::abs(velocityZ) * 0.85f;
    }

    state.zHeight = currentZ;
    state.strokeCount = strokeCount;
    state.maxVelocityKmh = maxVelocityKmh;
    state.elapsedTimeSec = elapsedTimeSec;
    // Spin cannot be inferred from translational speed. Surface observations
    // are handled by ObservedSpinEstimator and may be unavailable.
    state.spinRpm = 0.0f;

    return state;
}
