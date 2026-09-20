#undef NDEBUG
#include "physics_engine.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>

inline void TEST_ASSERT(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "[TEST FAIL] Assertion Failed: " << msg << std::endl;
        std::exit(1);
    }
}

void test_synthetic_rally() {
    std::cout << "[Test] Running Synthetic Rally Physics Test..." << std::endl;
    PhysicsEngine physics;

    float dt = 1.0f / 120.0f; // 120 FPS frame delta (0.00833s)
    float currentY = 0.5f;
    float currentX = 0.7625f;

    int strikeCountDetected = 0;
    int bounceCountDetected = 0;

    // Phase 1: Ball moving in +Y direction at +8.0 m/s for 40 frames
    float v1 = 8.0f;
    for (int i = 0; i < 40; ++i) {
        currentY += v1 * dt;
        PhysicsState state = physics.update(currentX, currentY, dt);
        if (state.isStrike) strikeCountDetected++;
        if (state.isTableBounce) bounceCountDetected++;
    }

    // Phase 2: Paddle Strike #1 - Ball reverses to -5.0 m/s for 40 frames
    float v2 = -5.0f;
    for (int i = 0; i < 40; ++i) {
        currentY += v2 * dt;
        PhysicsState state = physics.update(currentX, currentY, dt);
        if (state.isStrike) strikeCountDetected++;
        if (state.isTableBounce) bounceCountDetected++;
    }

    // Phase 3: Paddle Strike #2 - Ball reverses to +8.0 m/s for 40 frames
    float v3 = 8.0f;
    for (int i = 0; i < 40; ++i) {
        currentY += v3 * dt;
        PhysicsState state = physics.update(currentX, currentY, dt);
        if (state.isStrike) strikeCountDetected++;
        if (state.isTableBounce) bounceCountDetected++;
    }

    // Phase 4: Ball falls to simulated table surface and bounces for 60 frames
    for (int i = 0; i < 60; ++i) {
        currentY += 1.0f * dt;
        PhysicsState state = physics.update(currentX, currentY, dt);
        if (state.isStrike) strikeCountDetected++;
        if (state.isTableBounce) bounceCountDetected++;
    }

    PhysicsState finalState = physics.update(currentX, currentY, dt);

    std::cout << " -> Final Stroke Count: " << finalState.strokeCount << " (Expected: 2)" << std::endl;
    std::cout << " -> Final Max Speed: " << finalState.maxVelocityKmh << " km/h (Expected: <= 29.0 km/h)" << std::endl;
    std::cout << " -> Floor Bounces Triggered: " << bounceCountDetected << std::endl;

    TEST_ASSERT(finalState.strokeCount == 2, "strokeCount must be exactly 2");
    TEST_ASSERT(finalState.maxVelocityKmh <= 29.0f, "maxVelocityKmh must be <= 29.0 km/h");
    TEST_ASSERT(finalState.maxVelocityKmh >= 28.0f, "maxVelocityKmh must be >= 28.0 km/h");
    TEST_ASSERT(bounceCountDetected == 0, "bounceCountDetected must be ZERO for floor decay");
    std::cout << "[Test PASS] Synthetic Rally Physics Test Passed Successfully!\n" << std::endl;
}

void test_tracking_spike_rejection() {
    std::cout << "[Test] Running Tracking Spike Rejection Test..." << std::endl;
    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;

    // Initial normal tracking
    physics.update(0.5f, 1.0f, dt);
    PhysicsState s1 = physics.update(0.5f, 1.05f, dt); // ~21.6 km/h

    // Injected 10-meter tracking jump in 1 frame (> 4300 km/h)
    PhysicsState sSpike = physics.update(0.5f, 11.05f, dt);

    std::cout << " -> Speed during tracking jump: " << sSpike.velocityKmh << " km/h (Expected: <= 30.0 km/h)" << std::endl;
    std::cout << " -> Max Speed Recorded: " << sSpike.maxVelocityKmh << " km/h (Expected: <= 30.0 km/h)" << std::endl;

    TEST_ASSERT(sSpike.maxVelocityKmh <= 30.0f, "sSpike.maxVelocityKmh must be <= 30.0 km/h");
    std::cout << "[Test PASS] Tracking Spike Rejection Test Passed Successfully!\n" << std::endl;
}

void test_vision_bounce_detection() {
    std::cout << "[Test] Running Vision 2D Pixel Discontinuity Bounce Test..." << std::endl;
    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;

    float pixelY = 200.0f;
    float physicalX = 0.7625f;
    float physicalY = 1.37f;

    bool visionBounceDetected = false;

    // Ball falling downward in camera image (+pixelY): 300 pixels/sec
    for (int i = 0; i < 10; ++i) {
        pixelY += 300.0f * dt;
        physicalY += 2.0f * dt;
        PhysicsState s = physics.update(physicalX, physicalY, dt, 320.0f, pixelY);
        if (s.isTableBounce) visionBounceDetected = true;
    }

    // Ball impacts table surface & rebounds upward (-pixelY): -200 pixels/sec with sharp accel turning point
    for (int i = 0; i < 10; ++i) {
        pixelY -= 200.0f * dt;
        physicalY -= 1.5f * dt;
        PhysicsState s = physics.update(physicalX, physicalY, dt, 320.0f, pixelY);
        if (s.isTableBounce) visionBounceDetected = true;
    }

    std::cout << " -> Vision Bounce Detected: " << (visionBounceDetected ? "YES" : "NO") << " (Expected: YES)" << std::endl;
    TEST_ASSERT(visionBounceDetected == true, "visionBounceDetected must be TRUE");
    std::cout << "[Test PASS] Vision 2D Pixel Bounce Test Passed Successfully!\n" << std::endl;
}

void test_continuous_trigger_signal_rearming() {
    std::cout << "[Test] Running Continuous Trigger Signal Re-arming Test..." << std::endl;
    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;

    float pixelY = 200.0f;
    float physicalX = 0.7625f;
    float physicalY = 1.37f;

    int totalBounceCount = 0;

    // 1. Inbound downward trajectory: pixelVy = +500 px/s for 5 frames to build vyHistory
    for (int i = 0; i < 5; ++i) {
        pixelY += 500.0f * dt;
        PhysicsState s = physics.update(physicalX, physicalY, dt, 320.0f, pixelY);
        if (s.isTableBounce) totalBounceCount++;
    }

    // 2. Trigger point and continuous trigger zone: pixelVy = -10 px/s for 30 frames
    // This keeps conditionWindowed = true continuously past the 15-frame cooldown.
    for (int i = 0; i < 30; ++i) {
        pixelY -= 10.0f * dt;
        PhysicsState s = physics.update(physicalX, physicalY, dt, 320.0f, pixelY);
        if (s.isTableBounce) totalBounceCount++;
    }

    std::cout << " -> Continuous Signal Total Bounce Count: " << totalBounceCount << " (Expected: 1)" << std::endl;
    TEST_ASSERT(totalBounceCount == 1, "Continuous trigger signal past cooldown MUST fire exactly 1 bounce");
    std::cout << "[Test PASS] Continuous Trigger Signal Re-arming Test Passed Successfully!\n" << std::endl;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Table Tennis Physics Engine Unit Tests   " << std::endl;
    std::cout << "===========================================" << std::endl;

    test_synthetic_rally();
    test_tracking_spike_rejection();
    test_vision_bounce_detection();
    test_continuous_trigger_signal_rearming();

    std::cout << "All Physics Engine Tests PASSED!" << std::endl;
    return 0;
}
