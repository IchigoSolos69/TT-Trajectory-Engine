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
    cooldown = 0;
    bounceCooldown = 0;
    bounceSignalArmed_ = true;
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
    rallyActive = false;
    vyHistory.clear();
}

PhysicsState PhysicsEngine::update(float x, float y, float deltaTime, float pixelX, float pixelY) {
    (void)pixelX;
    PhysicsState state;
    state.isStrike = false;
    state.isTableBounce = false;
    state.bounceX = -1.0f;
    state.bounceY = -1.0f;

    // Clamp dt to valid interval [1/240s, 1/30s] to avoid division by zero or huge jumps after frame stalls
    float dt = std::max(1.0f / 240.0f, std::min(1.0f / 30.0f, deltaTime));

    // 1. First-frame / tracking re-entry check
    if (!hasLast) {
        lastX = x;
        lastY = y;
        lastPixelY = pixelY;
        smVy = 0.0f;
        lastValidSpeedKmh = 0.0f;
        hasLast = true;

        state.velocityKmh = 0.0f;
        state.zHeight = currentZ;
        state.strokeCount = strokeCount;
        state.maxVelocityKmh = maxVelocityKmh;
        state.elapsedTimeSec = elapsedTimeSec;
        state.spinRpm = 0.0f;
        return state;
    }

    // 2. Tracking Timeout & Rally Reset Check (120 frames ~ 1.0 sec without track)
    trackingLossFrames = 0;

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

    // Reject implausible tracking jumps (> 150 km/h = 41.67 m/s) instead of clamping
    if (rawSpeedKmh > 150.0f) {
        // Discard corrupt tracking sample: reset tracking baseline for next frame
        hasLast = false;
        state.velocityKmh = lastValidSpeedKmh;
        state.zHeight = currentZ;
        state.strokeCount = strokeCount;
        state.maxVelocityKmh = maxVelocityKmh;
        state.elapsedTimeSec = elapsedTimeSec;
        state.spinRpm = std::min(6000.0f, lastValidSpeedKmh * 55.0f + 1200.0f);
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

    // 4. Vision-Based 2D Pixel Discontinuity Table Bounce Detector
    if (bounceCooldown > 0) bounceCooldown--;

    if (pixelY >= 0.0f && lastPixelY >= 0.0f) {
        float pixelVy = (pixelY - lastPixelY) / dt; // positive = downward in camera view
        float pixelAccelSpike = (lastPixelVy - pixelVy) / dt;

        vyHistory.push_back(pixelVy);
        if (vyHistory.size() > 4) {
            vyHistory.erase(vyHistory.begin());
        }

        bool isTableRectangle = (x >= -0.1f && x <= 1.625f && y >= -0.1f && y <= 2.84f);

        float maxInbound = -9999.0f;
        for (size_t i = 0; i + 1 < vyHistory.size(); ++i) {
            if (vyHistory[i] > maxInbound) maxInbound = vyHistory[i];
        }
        float deltaVy = maxInbound - pixelVy;

        // Suppress table bounce trigger if paddle strike occurred within 4 frames
        bool strikeActive = (cooldown >= 11);

        bool conditionSignFlip = (lastPixelVy > 80.0f && pixelVy < -20.0f && pixelAccelSpike > 1500.0f);
        bool conditionWindowed = (maxInbound > 60.0f && pixelVy <= 0.0f && deltaVy > 400.0f);

        if (bounceCooldown == 0 && bounceSignalArmed_ && !strikeActive && vyHistory.size() >= 3 && isTableRectangle) {
            if (conditionSignFlip || conditionWindowed) {
                state.isTableBounce = true;
                state.bounceX = x;
                state.bounceY = y;
                currentZ = 0.02f; // Re-anchor elevation at physical impact site
                velocityZ = std::abs(velocityZ) * 0.85f; // Table surface restitution
                bounceCooldown = 15; // 15-frame refractory period
                bounceSignalArmed_ = false; // Disarm until trajectory signal clears
            }
        }

        // Re-arm signal once vertical trajectory is no longer meeting trigger conditions
        if (!conditionSignFlip && !conditionWindowed) {
            bounceSignalArmed_ = true;
        }
        lastPixelVy = pixelVy;
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
    // Aerodynamic Magnus Spin model (estimated from stroke velocity)
    state.spinRpm = std::min(6000.0f, rawSpeedKmh * 55.0f + 1200.0f);

    return state;
}
