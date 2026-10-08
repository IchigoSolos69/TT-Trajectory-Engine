#include "trajectory_intelligence.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv)try{
    if(argc<4 || argc==6 || argc>7)throw std::runtime_error("Usage: predictions.csv video.mp4 output.jsonl [normal|inject-misses|sweep] [acceleration_sigma measurement_sigma_scale]");
    std::ifstream input(argv[1]);std::ofstream output(argv[3]);if(!input||!output)throw std::runtime_error("Cannot open evaluation files");
    cv::VideoCapture cap(argv[2],cv::CAP_FFMPEG);if(!cap.isOpened())throw std::runtime_error("Cannot decode evaluation footage");
    const double fps=cap.get(cv::CAP_PROP_FPS);if(fps<=0)throw std::runtime_error("Invalid source FPS");
    const bool inject=argc>4&&std::string(argv[4])=="inject-misses",sweep=argc>4&&std::string(argv[4])=="sweep";
    if(argc>4&&!inject&&!sweep&&std::string(argv[4])!="normal")throw std::runtime_error("Unknown evaluation mode");
    std::vector<std::pair<float,float>> parameters{{argc==7?std::stof(argv[5]):4000.f,argc==7?std::stof(argv[6]):1.f}};
    if(sweep)parameters={{4000.f,1.f},{12000.f,1.f},{40000.f,1.f},{4000.f,2.f},{12000.f,2.f},{40000.f,2.f},{4000.f,4.f},{12000.f,4.f},{40000.f,4.f}};
    std::vector<intelligence::TrajectoryTracker> trackers;for(auto [acceleration,measurement]:parameters)trackers.emplace_back(acceleration,measurement);
    intelligence::TableCalibration table;std::string line;std::getline(input,line);cv::Mat frame;int current=0,rows=0,approximate=0;
    while(std::getline(input,line)){
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int fid,count;float x,y,score;if(!(row>>fid>>x>>y>>score>>count))throw std::runtime_error("Malformed prediction row");
        if(rows==0&&fid>1){if(!cap.set(cv::CAP_PROP_POS_FRAMES,fid-1))throw std::runtime_error("Cannot seek first evaluation frame");current=fid-1;}
        if(fid<=current)throw std::runtime_error("Prediction frame IDs must increase");
        while(current<fid){if(!cap.read(frame))throw std::runtime_error("Video shorter than predictions");++current;}
        float width=12,height=12;if(!(row>>width>>height)){width=height=12;++approximate;}
        std::vector<Detection> observations;const bool injected=inject && fid%97>=80 && fid%97<85;
        if(score>=.1f&&!injected)observations.push_back({0,score,(x-width/2)/frame.cols,(y-height/2)/frame.rows,width/frame.cols,height/frame.rows});
        for(size_t i=0;i<trackers.size();++i){
            auto s=trackers[i].update(fid,(fid-1)/fps,frame,observations,{},table);
            output<<std::setprecision(8);
            if(sweep)output<<"{\"acceleration_sigma\":"<<parameters[i].first<<",\"measurement_sigma_scale\":"<<parameters[i].second<<",\"sample\":";
            intelligence::sampleJson(output,s);if(sweep)output<<'}';output<<'\n';
        }++rows;
    }
    if(!output.good())throw std::runtime_error("Evaluation output write failed");
    std::cout<<"Evaluated "<<rows<<" native trajectory frames, "<<parameters.size()<<" configurations; "<<approximate<<" rows use approximate 12px crops; no table ground truth or new model inference\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
