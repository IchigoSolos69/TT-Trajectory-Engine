#pragma once
#include "tensorrt_engine.hpp"
#include <algorithm>
#include <cmath>

// The scene head uses class 0=person, 1=table. IoU NMS alone can retain a
// lower-leg proposal contained within the full person's box because its area
// is small. Apply a conservative, class-specific containment check; preserve
// overlapping full people and the raw suppressed proposals for diagnostics.
struct ScenePostprocessResult {
    std::vector<Detection> accepted;
    std::vector<Detection> suppressedPersonParts;
};
inline ScenePostprocessResult postprocessScene(const std::vector<Detection>& detections) {
    ScenePostprocessResult result;
    for (size_t i=0;i<detections.size();++i) {
        const auto& part=detections[i]; bool suppressed=false;
        if(part.class_id==0 && part.w>0 && part.h>0) {
            const cv::Rect2f small(part.x,part.y,part.w,part.h);
            for(size_t j=0;j<detections.size();++j) {
                if(i==j)continue;
                const auto& full=detections[j];
                if(full.class_id!=0 || full.confidence-part.confidence<.15f || full.w<=0 || full.h<=0)continue;
                const cv::Rect2f large(full.x,full.y,full.w,full.h);
                // Lower-half fragments only. A nearby head or similarly sized
                // overlapping person is not evidence of a duplicate leg.
                if(small.area()>large.area()*.25f || part.w>full.w*.7f || part.h>full.h*.5f ||
                   part.y<full.y+full.h*.5f)continue;
                if((small&large).area()/small.area()>=.95f){suppressed=true;break;}
            }
        }
        (suppressed?result.suppressedPersonParts:result.accepted).push_back(part);
    }
    return result;
}
