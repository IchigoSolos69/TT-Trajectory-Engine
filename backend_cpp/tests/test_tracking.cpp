#include "bytetrack.hpp"
#include <stdexcept>
#include <iostream>
static void check(bool c,const char* m){if(!c)throw std::runtime_error(m);}
int main(){
    ByteTracker tracker(.6f,.1f,3);
    auto detection=[](float x,float score,int cls=0){return Detection{cls,score,x,.2f,.05f,.05f};};
    auto a=tracker.update({detection(.2f,.9f)});const auto id=a[0].track_id;
    a=tracker.update({detection(.205f,.9f)});
    a=tracker.update({detection(.21f,.9f)});
    check(a.size()==1&&a[0].track_id==id,"Stable identity");
    check(std::abs(a[0].vx-.005f)<1e-6,"Motion residual must not zero velocity");
    a=tracker.update({detection(.215f,.3f)});
    check(a[0].age==0&&a[0].track_id==id,"Low-confidence association");
    a=tracker.update({});check(a[0].age==1,"Occluded prediction must not count as measured");
    a=tracker.update({detection(.225f,.8f),detection(.9f,.2f)});
    check(a.size()==1&&a[0].track_id==id,"Recovery and no low-confidence new identities");
    a=tracker.update({detection(.23f,.9f,1)});
    check(a.size()==2&&a[0].age>0,"Different classes must not associate");
    for(int i=0;i<4;++i)a=tracker.update({});check(a.empty(),"Expired tracks removed");
    std::cout<<"PASS: track identity, velocity, second association, occlusion, class isolation, expiration"<<std::endl;
}
