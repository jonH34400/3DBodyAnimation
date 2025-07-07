// ── standard ──────────────────────────────────────────────────────────
#include <array>
#include <fstream>
#include <iostream>
#include <vector>

// ── 3rd-party ─────────────────────────────────────────────────────────
#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
#include <smplx/smplx.hpp>

// ── local ─────────────────────────────────────────────────────────────
#include "OptimizeSMPL.hpp"


// very small loader for MediaPipe JSON (33 landmarks)
static const int MAP[24] = { -1,23,24, 11,-1,0, 11,12,13,14,15,16,
                             25,26,27,28,29,30,31,32, 1,2,7,8 };

std::vector<PixelKP> load_mp(const std::string& p,int W,int H)
{
    std::ifstream f(p); nlohmann::json j; f>>j;
    std::vector<PixelKP> out;
    auto mid = [&](int a,int b){
        return std::array<double,3>{
            (j[a]["x"].get<double>()+j[b]["x"].get<double>())*0.5,
            (j[a]["y"].get<double>()+j[b]["y"].get<double>())*0.5,
            std::min(j[a]["visibility"].get<double>(),
                     j[b]["visibility"].get<double>()) };
    };
    const auto pelvis = mid(23,24), spine2 = mid(11,12);
    for(int sid=0; sid<24; ++sid){
        double x,y,vis;
        if(sid==0){ x=pelvis[0]; y=pelvis[1]; vis=pelvis[2]; }
        else if(sid==4){ x=spine2[0]; y=spine2[1]; vis=spine2[2]; }
        else{
            int mp=MAP[sid]; if(mp<0) continue;
            x=j[mp]["x"].get<double>();
            y=j[mp]["y"].get<double>();
            vis=j[mp]["visibility"].get<double>();
        }
        if(vis<0.5) continue;
        out.push_back({sid,x*W,y*H});
    }
    return out;
}

int main(int argc,char** argv)
{
    if(argc<3){
        std::cout<<"usage: <SMPL.npz> <mp.json>\n"; return 0;
    }

    smplx::Model<smplx::model_config::SMPL_v1> model(argv[1]);

    constexpr int W=1280,H=720;
    auto kps = load_mp(argv[2],W,H);
    std::cout<<"keypoints used: "<<kps.size()<<"\n";

    auto result = optimize_smpl(kps, model);

    std::cout << result.summary.BriefReport() << "\n";

    std::cout<<"Final trans : ["<<result.trans[0]<<", "<<result.trans[1]<<", "<<result.trans[2]<<"]\n";
    std::cout<<"Beta[0..4]  : { ";
    for(int i=0;i<5;++i) std::cout<<result.beta[i]<<(i<4?", ":" ");
    std::cout<<"}\n";
    std::cout<<"Theta[0..4] : { ";
    for(int i=0;i<5;++i) std::cout<<result.theta[i]<<(i<4?", ":" ");
    std::cout<<"}\n";

    smplx::Body<smplx::model_config::SMPL_v1> body(model);
    using S = smplx::Scalar;
    body.shape().head<10>() =
        Eigen::Map<const Eigen::Matrix<double,10,1>>(result.beta).template cast<S>();
    body.pose() =
        Eigen::Map<const Eigen::Matrix<double,72,1>>(result.theta).template cast<S>();
    body.trans() =
        Eigen::Map<const Eigen::Matrix<double,3 ,1>>(result.trans).template cast<S>();

    body.update();

}
