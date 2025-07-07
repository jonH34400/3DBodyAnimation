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
#include "ReprojCost.hpp"

struct PixelKP { int jid; double u,v; };

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

    // 1. load SMPL_v1 (10-shape) model
    smplx::Model<smplx::model_config::SMPL_v1> model(argv[1]);

    // 2. keypoints
    constexpr int W=1280,H=720;
    auto kps = load_mp(argv[2],W,H);
    std::cout<<"keypoints used: "<<kps.size()<<"\n";

    // 3. build Ceres problem
    ceres::Problem pb;
    double theta[72]={0}, beta[10]={0}, trans[3]={0,0,2};

    for(auto& kp : kps){
        auto* cost = new ceres::NumericDiffCostFunction<
                         ReprojCost,ceres::CENTRAL,2,72,10,3>(
                         new ReprojCost(kp.jid,kp.u,kp.v,model));
        pb.AddResidualBlock(cost,nullptr,theta,beta,trans);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 50;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary sum;
    ceres::Solve(opts,&pb,&sum);
    std::cout<<sum.BriefReport()<<"\n";

    // ** Debug: print optimized parameters **
    std::cout<<"Final trans : ["<<trans[0]<<", "<<trans[1]<<", "<<trans[2]<<"]\n";
    std::cout<<"Beta[0..4]  : { ";
    for(int i=0;i<5;++i) std::cout<<beta[i]<<(i<4?", ":" ");
    std::cout<<"}\n";
    std::cout<<"Theta[0..4] : { ";
    for(int i=0;i<5;++i) std::cout<<theta[i]<<(i<4?", ":" ");
    std::cout<<"}\n";

    // 4. instantiate Body and update
    smplx::Body<smplx::model_config::SMPL_v1> body(model);
    using S = smplx::Scalar;
    body.shape().head<10>() =
        Eigen::Map<const Eigen::Matrix<double,10,1>>(beta).template cast<S>();
    body.pose() =
        Eigen::Map<const Eigen::Matrix<double,72,1>>(theta).template cast<S>();
    body.trans() =
        Eigen::Map<const Eigen::Matrix<double,3 ,1>>(trans).template cast<S>();

    body.update();

    // ** Debug: print first two vertices **
    const auto& V = body.verts();
    if(V.rows()>=2){
        std::cout<<"Vertex[0] = "<<V.row(0)<<"\n";
        std::cout<<"Vertex[1] = "<<V.row(1)<<"\n";
    }

    // 5. export mesh
    std::ofstream out("fitted.obj");
    for(int i=0;i<V.rows();++i)
        out<<"v "<<V(i,0)<<' '<<V(i,1)<<' '<<V(i,2)<<"\n";
    for(int i=0;i<model.faces.rows();++i)
        out<<"f "<<model.faces(i,0)+1<<' '
                  <<model.faces(i,1)+1<<' '
                  <<model.faces(i,2)+1<<"\n";
    std::cout<<"wrote fitted.obj\n";



    smplx::Body<smplx::model_config::SMPL_v1> body_neutral(model);
    body_neutral.shape().head<10>()=Eigen::Map<const Eigen::Matrix<double,10,1>>(beta).template cast<S>();
    body_neutral.pose().setZero();
    body_neutral.trans()=Eigen::Map<const Eigen::Matrix<double,3,1>>(trans).template cast<S>();
    body_neutral.update();
    std::ofstream outN("neutral.obj");
    auto N=body_neutral.verts();
    for(int i=0;i<N.rows();++i)
        outN<<"v "<<N(i,0)<<" "<<N(i,1)<<" "<<N(i,2)<<"\n";
    for(int i=0;i<model.faces.rows();++i)
        outN<<"f "<<model.faces(i,0)+1<<" "
            <<model.faces(i,1)+1<<" "<<model.faces(i,2)+1<<"\n";
    std::cout<<"wrote neutral.obj\n";
}
