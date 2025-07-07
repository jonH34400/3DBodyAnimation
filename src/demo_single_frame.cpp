// smpl_mediapipe_fit.cpp
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <smpl/SMPL.h>

// ---------------------------- config ----------------------------------
constexpr int W = 1280;               // image width
constexpr int H = 720;                // image height
constexpr double FX = 1000, FY = 1000, CX = W / 2.0, CY = H / 2.0;
constexpr double VIS_THRESH = 0.5;    // MediaPipe visibility cut-off
// ----------------------------------------------------------------------

// ---------- MediaPipe → SMPL mapping (24 entries, -1 = unused) --------
/*
      SMPL order (Loper et al.):
      0 pelvis  1 Lhip 2 Rhip 3 spine1 4 spine2 5 spine3
      6 Lshould 7 Rshould 8 Lelbow 9 Relbow 10 Lwrist 11 Rwrist
      12 Lknee  13 Rknee 14 Lankle 15 Rankle 16 Lheel 17 Rheel
      18 Ltoe   19 Rtoe 20 Leye  21 Reye  22 Lear   23 Rear
*/
static const std::array<int,24> MP_ID = {
    -1,                 // 0 pelvis will be mid(23,24)
    23, 24,             // hips
    11,                 // spine1  (left_shldr)
    -1,                 // spine2  (mid shoulders) – calc mid(11,12)
    0,                  // spine3 / neck (nose)
    11, 12,             // shoulders
    13, 14,             // elbows
    15, 16,             // wrists
    25, 26,             // knees
    27, 28,             // ankles
    29, 30,             // heels
    31, 32,             // big-toe (foot index)
    1, 2,               // eyes
    7, 8                // ears
};
// ----------------------------------------------------------------------

struct PixelKP {
    int   smpl_id;          // 0-23
    double u, v;            // pixel coords
};

// ---------- helper: load & convert MediaPipe JSON ---------------------
std::vector<PixelKP> load_mp_keypoints(const std::string& path)
{
    std::ifstream f(path);
    nlohmann::json j;  f >> j;                         // j is a 33-elem array

    struct MP { double x,y,vis; };
    std::vector<MP> mp;
    mp.reserve(33);
    for (auto& l : j)
        mp.push_back({l["x"], l["y"], l["visibility"]});

    // Build SMPL subset
    std::vector<PixelKP> out;
    out.reserve(24);

    // pre-compute midpoints we’ll need
    auto midpoint = [&](int a, int b)->MP {
        return {(mp[a].x + mp[b].x)*0.5, (mp[a].y + mp[b].y)*0.5,
                std::min(mp[a].vis, mp[b].vis)};
    };
    MP pelvis   = midpoint(23,24);
    MP spine2   = midpoint(11,12);

    for (int sid = 0; sid < 24; ++sid)
    {
        MP lnd;
        if (sid == 0)          lnd = pelvis;
        else if (sid == 4)     lnd = spine2;
        else {
            int mpid = MP_ID[sid];
            if (mpid < 0) continue;               // unused joint
            lnd = mp[mpid];
        }
        if (lnd.vis < VIS_THRESH) continue;       // drop low-vis
        out.push_back(
            {sid, lnd.x * W, lnd.y * H}           // to pixels
        );
    }
    return out;   // 10-20 keypoints typically
}
// ----------------------------------------------------------------------

// --------- Ceres reprojection error functor (2 residuals) -------------
struct ReprojErr {
    ReprojErr(int smpl_id, double u, double v)
        : id(smpl_id), u_obs(u), v_obs(v) {}

    template<typename T>
    bool operator()(const T* const theta,   // 72
                    const T* const beta,    // 10
                    const T* const trans,   // 3
                    T* residual) const
    {
        // ---- get 3-D joint i from SMPL (you expose a wrapper) ----
        Eigen::Matrix<T, 3, 1> J =
            smpl_forward_joint<T>(id, theta, beta, trans);  // (X,Y,Z)

        const T& X = J[0];
        const T& Y = J[1];
        const T& Z = J[2];

        T u_proj = T(FX) * X / Z + T(CX);
        T v_proj = T(FY) * Y / Z + T(CY);

        residual[0] = u_proj - T(u_obs);
        residual[1] = v_proj - T(v_obs);
        return true;
    }
    int    id;
    double u_obs, v_obs;
};
// ----------------------------------------------------------------------

// -------------- main demo ---------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "usage: smpl_mediapipe_fit <smpl_json> <mp.json>\n";
        return 0;
    }

    // -- 1. build SMPL neutral mesh -----------------------------------
    smpl::SMPL smpl;
    smpl.setModelPath(argv[1]);    // e.g. SMPL_FEMALE.json
    smpl.init();

    torch::Tensor betas = torch::zeros({1,10},   torch::kFloat32);
    torch::Tensor theta = torch::zeros({1,24,3}, torch::kFloat32);
    torch::Tensor trans = torch::tensor({{0.0f,0.0f,2.0f}}); // simple +2 m

    // -- 2. load 2-D keypoints ----------------------------------------
    auto kps = load_mp_keypoints(argv[2]);
    std::cout << "using " << kps.size() << " keypoints\n";

    // -- 3. build Ceres problem ---------------------------------------
    ceres::Problem problem;
    double theta_d[72] = {0}, beta_d[10] = {0}, trans_d[3] = {0,0,2};

    for (auto& kp : kps)
    {
        typedef ceres::AutoDiffCostFunction<ReprojErr,2,72,10,3> Cost;
        problem.AddResidualBlock(
            new Cost(new ReprojErr(kp.smpl_id, kp.u, kp.v)),
            nullptr, theta_d, beta_d, trans_d);
    }
    // simple priors (lambda = 0.001)
    for (int i=0;i<72;++i) problem.SetParameterLowerBound(theta_d,i,-3.14),
                           problem.SetParameterUpperBound(theta_d,i, 3.14);

    ceres::Solver::Options opt;
    opt.linear_solver_type = ceres::DENSE_QR;
    opt.max_num_iterations = 50;
    ceres::Solver::Summary sum;
    ceres::Solve(opt,&problem,&sum);
    std::cout << sum.BriefReport() << '\n';

    // -- 4. run SMPL with the optimised params and write OBJ ----------
    torch::Tensor theta_t = torch::from_blob(theta_d, {1,24,3}).clone();
    torch::Tensor beta_t  = torch::from_blob(beta_d,  {1,10   }).clone();
    torch::Tensor trans_t = torch::from_blob(trans_d, {1,3    }).clone();

    smpl.launch(beta_t, theta_t);               // you may need ->launch(beta,theta,trans)
    smpl.setVertPath("fitted.obj");
    smpl.out(0);

    std::cout << "wrote fitted.obj\n";
    return 0;
}
