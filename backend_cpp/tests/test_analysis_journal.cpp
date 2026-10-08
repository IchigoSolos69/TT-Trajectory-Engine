#include "analysis_journal.hpp"
#include <iostream>
#include <chrono>
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main()try{
    using namespace recording;
    auto root=std::filesystem::path("artifacts/analyses")/("journal_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    JournalWriter writer(root,"{\"schema_version\":2}");JournalReader reader(root);
    for(int frame=1;frame<=200;++frame){
        writer.append(frame,json({{"frame_id",std::to_string(frame)},{"strikes",frame<50?"[]":"[{\"id\":1,\"label\":\"quoted \\\"value\\\"\"}]"},
            {"intelligence","{\"trajectory\":[{\"frame_id\":"+std::to_string(frame)+"}],\"rally_graph\":{\"events\":[]}}"}}));
        require(reader.available()==frame,"Committed frame unavailable to concurrent reader");
    }
    writer.finish(200);
    auto selected=reader.read(100);require(object(selected.at("intelligence")).at("trajectory").find("\"frame_id\":100")!=std::string::npos,"Requested sample absent");
    require(reader.read(2).at("strikes")=="[]","Future event leaked into old frame");
    require(reader.read(100)==selected,"Backward/forward seeking changes historical analytics");
    bool rejects=false;try{reader.read(201);}catch(const std::runtime_error&){rejects=true;}require(rejects,"Seek-ahead must wait for analysis");
    rejects=false;try{writer.append(202,"{\"frame_id\":202}");}catch(const std::runtime_error&){rejects=true;}require(rejects,"Skipped analytical frame accepted");
    rejects=false;try{object("{\"x\":1,\"x\":2}");}catch(const std::runtime_error&){rejects=true;}require(rejects,"Duplicate JSON keys accepted");
    require(object(json({{"path",quote("folder\\file \"quoted\"\n")}})).at("path")==quote("folder\\file \"quoted\"\n"),"Escaped strings changed");
    std::cout<<"PASS: consecutive durable frames, historical event state, exact seek, bounded reconstruction, JSON escaping\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
