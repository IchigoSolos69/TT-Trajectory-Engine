#include "homography_calibrator.hpp"

HomographyCalibrator::HomographyCalibrator()
    : isCalibrated_(false), tableW_(1.525f), tableL_(2.74f) {}

HomographyCalibrator::~HomographyCalibrator() {}

bool HomographyCalibrator::calibrateTable(
    float tlX, float tlY,
    float trX, float trY,
    float brX, float brY,
    float blX, float blY,
    float tableWidthMeters,
    float tableLengthMeters,
    bool isDefaultGrid
) {
    tableW_ = tableWidthMeters;
    tableL_ = tableLengthMeters;

#ifdef HAS_OPENCV
    // 1. Table corners in video pixels (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    std::vector<cv::Point2f> imagePoints = {
        cv::Point2f(tlX, tlY), cv::Point2f(trX, trY),
        cv::Point2f(brX, brY), cv::Point2f(blX, blY)
    };

    // 2. Real-world table corners in meters (ITTF: 1.525m x 2.74m)
    std::vector<cv::Point2f> realWorldPoints = {
        cv::Point2f(0.0f, 0.0f),           cv::Point2f(tableW_, 0.0f),
        cv::Point2f(tableW_, tableL_),    cv::Point2f(0.0f, tableL_)
    };

    // 3. Generate Homography Matrix
    homographyMatrix_ = cv::findHomography(imagePoints, realWorldPoints);
    isCalibrated_ = !homographyMatrix_.empty();
    
    if (isCalibrated_) {
        if (isDefaultGrid) {
            std::cout << "[HomographyCalibrator] Default ITTF grid perspective matrix initialized (Standby)." << std::endl;
        } else {
            std::cout << "[HomographyCalibrator] Table calibrated successfully from live camera frame (1.525m x 2.74m grid)." << std::endl;
        }
    }
    return isCalibrated_;
#else
    isCalibrated_ = true;
    return true;
#endif
}

std::pair<float, float> HomographyCalibrator::pixelToMeters(float pixelX, float pixelY) {
#ifdef HAS_OPENCV
    if (!isCalibrated_) return { pixelX * 0.002f, pixelY * 0.002f };

    std::vector<cv::Point2f> pixelCoords = { cv::Point2f(pixelX, pixelY) };
    std::vector<cv::Point2f> meterCoords;

    cv::perspectiveTransform(pixelCoords, meterCoords, homographyMatrix_);
    return { meterCoords[0].x, meterCoords[0].y };
#else
    // Linear fallback approximation
    return { (pixelX / 1920.0f) * tableW_, (pixelY / 1080.0f) * tableL_ };
#endif
}

float HomographyCalibrator::computeVelocityKmh(
    float currentPixelX, float currentPixelY,
    float prevPixelX, float prevPixelY,
    float fps
) {
    auto currM = pixelToMeters(currentPixelX, currentPixelY);
    auto prevM = pixelToMeters(prevPixelX, prevPixelY);

    // 1. Calculate Euclidean distance in real-world meters
    float dx = currM.first - prevM.first;
    float dy = currM.second - prevM.second;
    float distanceMeters = std::sqrt(dx * dx + dy * dy);

    // 2. Time elapsed per frame
    float timeElapsed = 1.0f / std::max(fps, 1.0f); // seconds

    // 3. Calculate Speed (m/s)
    float speed_mps = distanceMeters / timeElapsed;

    // 4. Convert to km/h
    float speed_kmh = speed_mps * 3.6f;
    return speed_kmh;
}
