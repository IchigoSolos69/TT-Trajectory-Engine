#pragma once
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cctype>
#include <iomanip>

namespace recording {
inline std::string quote(const std::string& value){
    std::ostringstream out;out<<'"';
    for(unsigned char c:value){switch(c){case '"':out<<"\\\"";break;case '\\':out<<"\\\\";break;
        case '\n':out<<"\\n";break;case '\r':out<<"\\r";break;case '\t':out<<"\\t";break;
        default:if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}}
    out<<'"';return out.str();
}
// Preserve raw JSON values; only split top-level fields of native JSON objects.
// Keys are deliberately restricted to the ASCII schema keys used by the engine.
using Object=std::map<std::string,std::string>;
inline Object object(const std::string& json){
    Object result;size_t i=0;
    const auto white=[&]{while(i<json.size()&&std::isspace(static_cast<unsigned char>(json[i])))++i;};
    white();if(i==json.size()||json[i++]!='{')throw std::runtime_error("Expected JSON object");
    white();if(i<json.size()&&json[i]=='}'){++i;white();if(i!=json.size())throw std::runtime_error("Trailing JSON");return result;}
    while(i<json.size()){
        white();if(i==json.size()||json[i++]!='"')throw std::runtime_error("Expected JSON key");
        size_t keyStart=i;while(i<json.size()&&json[i]!='"'){if(json[i]=='\\'||static_cast<unsigned char>(json[i])<32)throw std::runtime_error("Invalid schema key");++i;}
        if(i==json.size())throw std::runtime_error("Unterminated key");
        std::string key=json.substr(keyStart,i-keyStart);++i;white();
        if(i==json.size()||json[i++]!=':')throw std::runtime_error("Expected JSON colon");
        white();size_t start=i;bool string=false,escape=false;int depth=0;
        for(;i<json.size();++i){char c=json[i];
            if(string){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')string=false;continue;}
            if(c=='"'){string=true;continue;}
            if(c=='{'||c=='[')++depth;
            else if(c=='}'||c==']'){if(depth==0)break;--depth;}
            else if(c==','&&depth==0)break;
        }
        if(string||depth!=0||i==json.size()||i==start)throw std::runtime_error("Incomplete JSON value");
        if(!result.emplace(key,json.substr(start,i-start)).second)throw std::runtime_error("Duplicate JSON field");
        if(json[i]=='}'){++i;white();if(i!=json.size())throw std::runtime_error("Trailing JSON");return result;}
        if(json[i++]!=',')throw std::runtime_error("Expected field separator");
    }throw std::runtime_error("Unterminated JSON object");
}
inline std::string json(const Object& fields){std::string out="{";for(const auto& [key,value]:fields){if(out.size()>1)out+=',';out+=quote(key)+":"+value;}return out+"}";}
inline std::string compact(const std::string& text){
    std::string result;bool string=false,escape=false;
    for(char c:text){if(string){result+=c;if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')string=false;}
        else if(c=='"'){string=true;result+=c;}else if(!std::isspace(static_cast<unsigned char>(c)))result+=c;}
    return result;
}
inline std::string readText(const std::filesystem::path& path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot read analysis file: "+path.u8string());return {std::istreambuf_iterator<char>(in),{}};}
inline void writeText(const std::filesystem::path& path,const std::string& text){std::ofstream out(path,std::ios::binary);out<<text;out.close();if(!out)throw std::runtime_error("Cannot write analysis file: "+path.u8string());}
inline void publishText(const std::filesystem::path& path,const std::string& text){
    auto temporary=path;temporary+=".tmp";writeText(temporary,text);std::filesystem::rename(temporary,path);
}

class JournalWriter {
public:
    explicit JournalWriter(std::filesystem::path root,const std::string& manifest):root_(std::move(root)){
        std::filesystem::create_directories(root_/"blocks");
        if(std::filesystem::exists(root_/"frames.idx"))throw std::runtime_error("Refusing to overwrite an analysis journal");
        records_.open(root_/"frames.jsonl",std::ios::binary);index_.open(root_/"frames.idx",std::ios::binary);
        if(!records_||!index_)throw std::runtime_error("Cannot create analysis journal");
        publishText(root_/"manifest.json",manifest);
    }
    void append(int frame,const std::string& telemetry){
        if(frame!=count_+1)throw std::runtime_error("Analysis journal requires every consecutive source frame");
        auto data=object(telemetry);
        if(std::stoi(data.at("frame_id"))!=frame)throw std::runtime_error("Frame identity mismatch");
        data.erase("preview_jpeg");
        for(const auto& key:{"strikes","impact_points","image_impact_points"})intern(data,key);
        if(data.count("intelligence")){
            auto intelligence=object(data.at("intelligence"));
            if(intelligence.count("rally_graph")){data["rally_graph"]=intelligence.at("rally_graph");intelligence.erase("rally_graph");intern(data,"rally_graph");}
            data["intelligence"]=json(intelligence);
        }
        const uint64_t offset=static_cast<uint64_t>(records_.tellp());
        records_<<compact(json(data))<<'\n';records_.flush();
        if(!records_)throw std::runtime_error("Analysis record write failed; no frame committed");
        // Publish the index only after all referenced data is readable.
        index_.write(reinterpret_cast<const char*>(&offset),sizeof(offset));index_.flush();
        if(!index_)throw std::runtime_error("Analysis index write failed");
        ++count_;
    }
    void finish(int expected){
        records_.flush();index_.flush();
        publishText(root_/"completion.json",json({{"status",quote(count_==expected?"COMPLETE":"PARTIAL")},{"analyzed_frames",std::to_string(count_)},{"total_frames",std::to_string(expected)}}));
    }
private:
    void intern(Object& data,const std::string& key){
        auto found=data.find(key);if(found==data.end())return;
        const auto value=compact(found->second);auto& previous=last_[key];
        if(previous.first!=value){previous={value,++blockId_};writeText(root_/"blocks"/(std::to_string(blockId_)+".json"),value);}
        data["_"+key+"_ref"]=std::to_string(previous.second);data.erase(key);
    }
    std::filesystem::path root_;std::ofstream records_,index_;int count_=0,blockId_=0;
    std::map<std::string,std::pair<std::string,int>> last_;
};

class JournalReader {
public:
    explicit JournalReader(std::filesystem::path root):root_(std::move(root)){}
    int available()const{
        std::error_code error;auto bytes=std::filesystem::file_size(root_/"frames.idx",error);
        return error?0:static_cast<int>(bytes/sizeof(uint64_t));
    }
    Object read(int frame,bool history=true){
        auto data=raw(frame);
        for(const auto& key:{"strikes","impact_points","image_impact_points","rally_graph"}){
            const std::string ref="_"+std::string(key)+"_ref";
            if(data.count(ref)){data[key]=block(std::stoi(data.at(ref)));data.erase(ref);}
        }
        if(data.count("intelligence")){
            auto intelligence=object(data.at("intelligence"));
            if(data.count("rally_graph")){intelligence["rally_graph"]=data.at("rally_graph");data.erase("rally_graph");}
            if(history){
                std::string trajectory="[";
                for(int i=std::max(1,frame-179);i<=frame;++i){auto old=raw(i);if(!old.count("intelligence"))continue;
                    const auto single=object(old.at("intelligence")).at("trajectory");
                    if(single.size()>2){if(trajectory.size()>1)trajectory+=',';trajectory+=single.substr(1,single.size()-2);}}
                intelligence["trajectory"]=trajectory+"]";
            }data["intelligence"]=json(intelligence);
        }return data;
    }
private:
    Object raw(int frame){
        if(frame<1||frame>available())throw std::runtime_error("Requested frame has not been analyzed");
        auto cached=frames_.find(frame);if(cached!=frames_.end())return cached->second;
        std::ifstream index(root_/"frames.idx",std::ios::binary),records(root_/"frames.jsonl",std::ios::binary);
        uint64_t offset=0;index.seekg((frame-1)*sizeof(offset));index.read(reinterpret_cast<char*>(&offset),sizeof(offset));
        if(!index)throw std::runtime_error("Incomplete analysis index");records.seekg(offset);
        std::string line;if(!std::getline(records,line))throw std::runtime_error("Incomplete analysis record");
        auto data=object(line);if(std::stoi(data.at("frame_id"))!=frame)throw std::runtime_error("Corrupt analysis index");
        if(frames_.size()>=256)frames_.clear();frames_[frame]=data;return data;
    }
    std::string block(int id){
        if(id<1)throw std::runtime_error("Invalid analysis block reference");
        auto found=blocks_.find(id);if(found!=blocks_.end())return found->second;
        auto value=readText(root_/"blocks"/(std::to_string(id)+".json"));if(blocks_.size()>=64)blocks_.clear();blocks_[id]=value;return value;
    }
    std::filesystem::path root_;std::map<int,Object> frames_;std::map<int,std::string> blocks_;
};
} // namespace recording
