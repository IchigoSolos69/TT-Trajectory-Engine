#ifndef HOMOGRAPHY_CALIBRATOR_HPP
#define HOMOGRAPHY_CALIBRATOR_HPP

#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>

#pragma once
#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#ifndef HAS_OPENCV
#define HAS_OPENCV
#endif
#endif


class HomographyCalibrator {
public:
    HomographyCalibrator();
    ~HomographyCalibrator();

    // Initialize Homography with 4 image pixel corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    bool calibrateTable(
        float tlX, float tlY,
        float trX, float trY,
        float brX, float brY,
        float blX, float blY,
        float tableWidthMeters = 1.525f,
        float tableLengthMeters = 2.74f,
        bool isDefaultGrid = false
    );

    // Transform 2D pixel coordinates (x, y) to real-world table meters (x_m, y_m)
    std::pair<float, float> pixelToMeters(float pixelX, float pixelY);

    // Compute Euclidean velocity in km/h across consecutive frames
    float computeVelocityKmh(float currentPixelX, float currentPixelY, float prevPixelX, float prevPixelY, float fps = 60.0f);

private:
    bool isCalibrated_;
    float tableW_;
    float tableL_;

#ifdef HAS_OPENCV
    cv::Mat homographyMatrix_;
#endif
};

#endif // HOMOGRAPHY_CALIBRATOR_HPP
