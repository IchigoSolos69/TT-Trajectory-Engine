#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <cassert>

struct MatcherResult {
    int tp;
    int fp;
    int fn;
    float precision;
    float recall;
    float f1;
};

MatcherResult evaluateMatcher(const std::vector<int>& detected, const std::vector<int>& groundTruth, int tolerance = 3) {
    std::set<int> matchedGt;
    int tp = 0, fp = 0;

    for (int det : detected) {
        bool match = false;
        for (int gt : groundTruth) {
            if (std::abs(det - gt) <= tolerance && matchedGt.find(gt) == matchedGt.end()) {
                matchedGt.insert(gt);
                tp++;
                match = true;
                break;
            }
        }
        if (!match) fp++;
    }

    int fn = static_cast<int>(groundTruth.size()) - static_cast<int>(matchedGt.size());
    float prec = (tp + fp > 0) ? (static_cast<float>(tp) / (tp + fp)) * 100.0f : 0.0f;
    float rec = (tp + fn > 0) ? (static_cast<float>(tp) / (tp + fn)) * 100.0f : 0.0f;
    float f1 = (prec + rec > 0.0f) ? (2.0f * prec * rec / (prec + rec)) : 0.0f;

    return {tp, fp, fn, prec, rec, f1};
}

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "  UNIT TEST: C++ BOUNCE MATCHER EVALUATOR (tolerance = +/-3 frames)" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    // Case 1: Perfect detections
    {
        std::vector<int> gt = {100, 200, 300};
        std::vector<int> det = {100, 200, 300};
        MatcherResult res = evaluateMatcher(det, gt, 3);
        assert(res.tp == 3 && res.fp == 0 && res.fn == 0);
        assert(std::abs(res.precision - 100.0f) < 0.001f);
        assert(std::abs(res.recall - 100.0f) < 0.001f);
        std::cout << "  [PASS] Case 1: Perfect Detections (TP=3, FP=0, FN=0, Prec=100%, Rec=100%)" << std::endl;
    }

    // Case 2: Two detections near one ground-truth bounce (1-to-1 match enforcement)
    {
        std::vector<int> gt = {100};
        std::vector<int> det = {100, 102}; // frame 100 and 102 both within +/-3 of gt 100
        MatcherResult res = evaluateMatcher(det, gt, 3);
        assert(res.tp == 1 && res.fp == 1 && res.fn == 0);
        assert(std::abs(res.precision - 50.0f) < 0.001f);
        assert(std::abs(res.recall - 100.0f) < 0.001f);
        std::cout << "  [PASS] Case 2: Duplicate Near-GT Detections (TP=1, FP=1, FN=0, Prec=50%, Rec=100%)" << std::endl;
    }

    // Case 3: Detection outside tolerance window
    {
        std::vector<int> gt = {100};
        std::vector<int> det = {105}; // 5 frames away > 3 frame tolerance
        MatcherResult res = evaluateMatcher(det, gt, 3);
        assert(res.tp == 0 && res.fp == 1 && res.fn == 1);
        assert(std::abs(res.precision - 0.0f) < 0.001f);
        assert(std::abs(res.recall - 0.0f) < 0.001f);
        std::cout << "  [PASS] Case 3: Detection Outside Tolerance Window (TP=0, FP=1, FN=1, Prec=0%, Rec=0%)" << std::endl;
    }

    // Case 4: Zero detections (zero division safe)
    {
        std::vector<int> gt = {100, 200};
        std::vector<int> det = {};
        MatcherResult res = evaluateMatcher(det, gt, 3);
        assert(res.tp == 0 && res.fp == 0 && res.fn == 2);
        assert(std::abs(res.precision - 0.0f) < 0.001f);
        assert(std::abs(res.recall - 0.0f) < 0.001f);
        std::cout << "  [PASS] Case 4: Zero Detections (TP=0, FP=0, FN=2, Prec=0%, Rec=0%, Safe Division)" << std::endl;
    }

    std::cout << "=========================================================================" << std::endl;
    std::cout << "  ALL MATCHER UNIT TESTS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "=========================================================================\n" << std::endl;
    return 0;
}
