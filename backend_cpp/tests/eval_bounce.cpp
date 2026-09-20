#include "physics_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

struct GroundTruth {
    std::map<int, std::pair<float, float>> ballPositions;
    std::set<int> bounceFrames;
};

GroundTruth loadGroundTruth(const std::string& ballJsonPath, const std::string& eventJsonPath) {
    GroundTruth gt;

    // 1. Load Ball Markup
    std::ifstream bFile(ballJsonPath);
    if (bFile.is_open()) {
        std::stringstream buffer;
        buffer << bFile.rdbuf();
        std::string content = buffer.str();

        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos) {
            size_t endFramePos = content.find("\"", pos + 1);
            if (endFramePos == std::string::npos) break;
            std::string frameStr = content.substr(pos + 1, endFramePos - pos - 1);
            pos = endFramePos + 1;

            if (frameStr == "x" || frameStr == "y" || frameStr == "status") continue;

            try {
                int frameIdx = std::stoi(frameStr);
                size_t xPos = content.find("\"x\"", pos);
                if (xPos == std::string::npos || xPos > pos + 100) continue;
                size_t xValStart = content.find(":", xPos) + 1;
                size_t xValEnd = content.find_first_of(",}", xValStart);
                float xVal = std::stof(content.substr(xValStart, xValEnd - xValStart));

                size_t yPos = content.find("\"y\"", pos);
                if (yPos == std::string::npos || yPos > pos + 100) continue;
                size_t yValStart = content.find(":", yPos) + 1;
                size_t yValEnd = content.find_first_of(",}", yValStart);
                float yVal = std::stof(content.substr(yValStart, yValEnd - yValStart));

                if (xVal >= 0.0f && yVal >= 0.0f) {
                    gt.ballPositions[frameIdx] = {xVal, yVal};
                }
            } catch (...) {
                continue;
            }
        }
    }

    // 2. Load Events Markup
    std::ifstream eFile(eventJsonPath);
    if (eFile.is_open()) {
        std::stringstream buffer;
        buffer << eFile.rdbuf();
        std::string content = buffer.str();

        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos) {
            size_t endFramePos = content.find("\"", pos + 1);
            if (endFramePos == std::string::npos) break;
            std::string frameStr = content.substr(pos + 1, endFramePos - pos - 1);
            pos = endFramePos + 1;

            if (frameStr == "bounce" || frameStr == "net" || frameStr == "empty_event") continue;

            try {
                int frameIdx = std::stoi(frameStr);
                size_t valStart = content.find("\"", pos);
                if (valStart == std::string::npos || valStart > pos + 20) continue;
                size_t valEnd = content.find("\"", valStart + 1);
                std::string eventName = content.substr(valStart + 1, valEnd - valStart - 1);
                pos = valEnd + 1;

                if (eventName.find("bounce") != std::string::npos) {
                    gt.bounceFrames.insert(frameIdx);
                }
            } catch (...) {
                continue;
            }
        }
    }

    return gt;
}

struct EvalResult {
    float precision;
    float recall;
    float f1;
    int tp;
    int fp;
    int fn;
    int totalDetected;
};

EvalResult evaluateSingleRun(const GroundTruth& gt, float sigmaPx, float dropoutRate, unsigned int seed, int toleranceFrames = 3) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> noiseDist(0.0f, sigmaPx);
    std::uniform_real_distribution<float> dropDist(0.0f, 1.0f);

    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;

    std::vector<int> detectedBounceFrames;

    for (const auto& kv : gt.ballPositions) {
        int fIdx = kv.first;

        // Apply frame dropout
        if (dropoutRate > 0.0f && dropDist(gen) < dropoutRate) {
            continue;
        }

        float rawX = kv.second.first;
        float rawY = kv.second.second;

        // Apply Gaussian noise
        float px = rawX + (sigmaPx > 0.0f ? noiseDist(gen) : 0.0f);
        float py = rawY + (sigmaPx > 0.0f ? noiseDist(gen) : 0.0f);

        float metersX = (px / 1920.0f) * 1.525f;
        float metersY = (py / 1080.0f) * 2.74f;

        PhysicsState state = physics.update(metersX, metersY, dt, px, py);
        if (state.isTableBounce) {
            detectedBounceFrames.push_back(fIdx);
        }
    }

    // Evaluate TP, FP, FN with specified tolerance window
    std::set<int> matchedGt;
    int tp = 0, fp = 0;

    for (int det : detectedBounceFrames) {
        bool match = false;
        for (int realG : gt.bounceFrames) {
            if (std::abs(det - realG) <= toleranceFrames && matchedGt.find(realG) == matchedGt.end()) {
                matchedGt.insert(realG);
                tp++;
                match = true;
                break;
            }
        }
        if (!match) fp++;
    }

    int fn = static_cast<int>(gt.bounceFrames.size()) - static_cast<int>(matchedGt.size());
    float prec = (tp + fp > 0) ? (static_cast<float>(tp) / (tp + fp)) * 100.0f : 0.0f;
    float rec = (tp + fn > 0) ? (static_cast<float>(tp) / (tp + fn)) * 100.0f : 0.0f;
    float f1 = (prec + rec > 0.0f) ? (2.0f * prec * rec / (prec + rec)) : 0.0f;

    return {prec, rec, f1, tp, fp, fn, static_cast<int>(detectedBounceFrames.size())};
}

void printCleanResult(const std::string& seqName, const std::string& ballJson, const std::string& eventJson) {
    GroundTruth gt = loadGroundTruth(ballJson, eventJson);
    EvalResult r3 = evaluateSingleRun(gt, 0.0f, 0.0f, 42, 3);
    EvalResult r5 = evaluateSingleRun(gt, 0.0f, 0.0f, 42, 5);

    std::cout << "\n--- CLEAN UNCONTAMINATED EVALUATION (sigma=0, dropout=0): " << seqName << " ---" << std::endl;
    std::cout << "  GT Bounces Count : " << gt.bounceFrames.size() << std::endl;
    std::cout << "  [Tolerance +/-3 frames] -> Detected: " << r3.totalDetected << " | TP: " << r3.tp << " | FP: " << r3.fp << " | FN: " << r3.fn
              << " | Precision: " << r3.precision << "% | Recall: " << r3.recall << "% | F1: " << r3.f1 << "%" << std::endl;
    std::cout << "  [Tolerance +/-5 frames] -> Detected: " << r5.totalDetected << " | TP: " << r5.tp << " | FP: " << r5.fp << " | FN: " << r5.fn
              << " | Precision: " << r5.precision << "% | Recall: " << r5.recall << "% | F1: " << r5.f1 << "%" << std::endl;

    // Diagnostic check for inter-trigger spacing and false positives
    PhysicsEngine physics;
    float dt = 1.0f / 120.0f;
    std::vector<int> detFrames;
    for (const auto& kv : gt.ballPositions) {
        float px = kv.second.first, py = kv.second.second;
        float mx = (px / 1920.0f) * 1.525f, my = (py / 1080.0f) * 2.74f;
        PhysicsState st = physics.update(mx, my, dt, px, py);
        if (st.isTableBounce) detFrames.push_back(kv.first);
    }

    int minDelta = 999999;
    std::vector<int> closeDeltas;
    for (size_t i = 1; i < detFrames.size(); i++) {
        int delta = detFrames[i] - detFrames[i-1];
        if (delta < minDelta) minDelta = delta;
        if (delta < 15) closeDeltas.push_back(delta);
    }
    std::cout << "  [Cooldown Diagnostic] Min inter-trigger delta: " << minDelta << " frames. Triggers < 15 frames apart: " << closeDeltas.size() << std::endl;

    std::set<int> matchedGt;
    std::vector<int> fpFrames;
    for (int det : detFrames) {
        bool match = false;
        for (int realG : gt.bounceFrames) {
            if (std::abs(det - realG) <= 3 && matchedGt.find(realG) == matchedGt.end()) {
                matchedGt.insert(realG);
                match = true;
                break;
            }
        }
        if (!match) fpFrames.push_back(det);
    }

    std::cout << "  [False Positive Frames (tol +/-3)]: ";
    for (size_t i = 0; i < fpFrames.size(); i++) {
        std::cout << fpFrames[i] << (i + 1 < fpFrames.size() ? ", " : "");
    }
    std::cout << std::endl;
}

void runNoiseMatrix(const std::string& seqName, const std::string& ballJson, const std::string& eventJson) {
    GroundTruth gt = loadGroundTruth(ballJson, eventJson);
    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "  C++ NATIVE BOUNCE EVALUATOR NOISE MATRIX (MEAN ± STD ACROSS 5 SEEDS): " << seqName 
              << " (" << gt.ballPositions.size() << " frames, " << gt.bounceFrames.size() << " GT bounces)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Gaussian Noise (px) | Dropout (%) | Avg TP | Avg FP | Avg FN | Precision (%)      | Recall (%)         | F1 Score (%)" << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------------------" << std::endl;

    std::vector<float> sigmas = {0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    std::vector<float> dropouts = {0.0f, 0.05f, 0.10f, 0.15f, 0.20f};
    const int NUM_SEEDS = 5;

    for (float sig : sigmas) {
        for (float drop : dropouts) {
            std::vector<float> precs, recs, f1s;
            float sumTp = 0.0f, sumFp = 0.0f, sumFn = 0.0f;

            for (unsigned int seed = 42; seed < 42 + NUM_SEEDS; seed++) {
                EvalResult res = evaluateSingleRun(gt, sig, drop, seed, 3);
                precs.push_back(res.precision);
                recs.push_back(res.recall);
                f1s.push_back(res.f1);
                sumTp += res.tp;
                sumFp += res.fp;
                sumFn += res.fn;
            }

            float avgP = std::accumulate(precs.begin(), precs.end(), 0.0f) / NUM_SEEDS;
            float avgR = std::accumulate(recs.begin(), recs.end(), 0.0f) / NUM_SEEDS;
            float avgF = std::accumulate(f1s.begin(), f1s.end(), 0.0f) / NUM_SEEDS;

            auto calcStd = [](const std::vector<float>& vec, float mean) {
                float sqSum = 0.0f;
                for (float v : vec) sqSum += (v - mean) * (v - mean);
                return std::sqrt(sqSum / vec.size());
            };

            float stdP = calcStd(precs, avgP);
            float stdR = calcStd(recs, avgR);
            float stdF = calcStd(f1s, avgF);

            float avgTp = sumTp / NUM_SEEDS;
            float avgFp = sumFp / NUM_SEEDS;
            float avgFn = sumFn / NUM_SEEDS;

            printf("  %-17.1f | %-11.0f | %-6.1f | %-6.1f | %-6.1f | %6.2f ± %-5.2f | %6.2f ± %-5.2f | %6.2f ± %-5.2f\n",
                   sig, drop * 100.0f, avgTp, avgFp, avgFn, avgP, stdP, avgR, stdR, avgF, stdF);
        }
    }
}

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  NATIVE C++ PHYSICS ENGINE BOUNCE NOISE SENSITIVITY BENCHMARK (5 SEEDS)" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    printCleanResult("game_1 (Tuning Set)",
                     "python_tools/dataset/raw_markup/game_1/ball_markup.json",
                     "python_tools/dataset/raw_markup/game_1/events_markup.json");

    printCleanResult("test_1 (Dev Set)",
                     "python_tools/dataset/raw_markup/test_1/ball_markup.json",
                     "python_tools/dataset/raw_markup/test_1/events_markup.json");

    runNoiseMatrix("game_1 (Tuning Set)", 
                   "python_tools/dataset/raw_markup/game_1/ball_markup.json", 
                   "python_tools/dataset/raw_markup/game_1/events_markup.json");

    runNoiseMatrix("test_1 (Dev Set)", 
                   "python_tools/dataset/raw_markup/test_1/ball_markup.json", 
                   "python_tools/dataset/raw_markup/test_1/events_markup.json");

    return 0;
}
