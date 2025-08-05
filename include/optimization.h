#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <algorithm>

// You already have this.
struct PixelKP { int jid; double u, v; };

namespace unified_ad {

// === 1) Templated, shape- and pose-aware joint computation (no skinning needed) ===
template <typename T>
inline void compute_joints_T(const ark::AvatarModel& model,
                             const T* w,                 // [nS]
                             const T* theta,             // [3*(nJ-1)] (AA for joints 1..nJ-1)
                             Eigen::Matrix<T,3,Eigen::Dynamic>& Jworld) // (3 x nJ)
{
    const int nJ = model.numJoints();
    const int nS = model.numShapeKeys();

    // --- (a) shaped cloud as vector (3*N)
    Eigen::Matrix<T, Eigen::Dynamic, 1> shapedCloudVec =
        model.keyClouds.cast<T>() * Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1>>(w, nS)
        + model.baseCloud.cast<T>();
    // Map to (3 x N)
    Eigen::Map<const Eigen::Matrix<T,3,Eigen::Dynamic>> shapedCloud(shapedCloudVec.data(), 3, model.numPoints());

    // --- (b) shape-aware base joints J0
    Eigen::Matrix<T,3,Eigen::Dynamic> J0(3, nJ);
    if (model.useJointShapeRegressor) {
        J0 = model.initialJointPos.cast<T>();
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,1>> J0vec(J0.data(), 3*nJ);
        J0vec.noalias() += model.jointShapeReg.cast<T>() *
                           Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1>>(w, nS);
    } else {
        J0.noalias() = shapedCloud * model.jointRegressor.cast<T>();
    }

    // --- (c) local rotations from theta (pelvis/root=identity)
    std::vector<Eigen::Matrix<T,3,3>> Rloc(nJ), Rglob(nJ);
    std::vector<Eigen::Matrix<T,3,1>> Tglob(nJ);
    for (int j = 0; j < nJ; ++j) {
        Rloc[j].setIdentity();
        Rglob[j].setIdentity();
        Tglob[j].setZero();
    }
    for (int j = 1; j < nJ; ++j) {
        const T* aa = &theta[3*(j-1)];
        T Rm[9];
        ceres::AngleAxisToRotationMatrix(aa, Rm);
        Rloc[j] << Rm[0],Rm[3],Rm[6],
                   Rm[1],Rm[4],Rm[7],
                   Rm[2],Rm[5],Rm[8];
    }

    // --- (d) forward kinematics with bone translations from shape-aware J0
    // root: no global translation here; global t will be a separate parameter in the BA
    for (int j = 0; j < nJ; ++j) {
        const int p = model.parent[j];
        if (p < 0) { // root
            Rglob[j] = Rloc[j];
            Tglob[j].setZero();
        } else {
            const Eigen::Matrix<T,3,1> tloc = (J0.col(j) - J0.col(p)).template cast<T>();
            Rglob[j].noalias() = Rglob[p] * Rloc[j];
            Tglob[j].noalias() = Rglob[p] * tloc + Tglob[p];
        }
    }

    // World joint positions are Tglob
    Jworld.resize(3, nJ);
    for (int j = 0; j < nJ; ++j) Jworld.col(j) = Tglob[j];
}

// === 2) AutoDiff reprojection residual (single kp) ===
struct ReprojCostAD {
    ReprojCostAD(int jid,
                 double u_obs, double v_obs,
                 double fx, double fy, double cx, double cy,
                 const ark::AvatarModel* model)
        : jid_(jid), u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          model_(model) {}

    // Parameter blocks:
    // 0: s      (1)
    // 1: R_aa   (3)
    // 2: t      (3)
    // 3: theta  (3*(nJ-1))
    // 4: w      (nS)
    template <typename T>
    bool operator()(T const* const* params, T* r) const {
        const T  s    = params[0][0];
        const T* Raa  = params[1];
        const T* t    = params[2];
        const T* th   = params[3];
        const T* w    = params[4];

        const int nJ = model_->numJoints();
        const int nS = model_->numShapeKeys();

        // Joints as function of (w, th)
        Eigen::Matrix<T,3,Eigen::Dynamic> Jw;
        compute_joints_T<T>(*model_, w, th, Jw);
        Eigen::Matrix<T,3,1> J = Jw.col(jid_);

        // Apply global rotation R_aa
        T Jv[3] = { J(0), J(1), J(2) };
        ceres::AngleAxisRotatePoint(Raa, Jv, Jv);

        // Sim3 + projection
        const T X = s * Jv[0] + t[0];
        const T Y = s * Jv[1] + t[1];
        const T Z = s * Jv[2] + t[2];

        const T u = T(fx_) * (X/Z) + T(cx_);
        const T v = T(fy_) * (Y/Z) + T(cy_);

        r[0] = u - T(u_obs_);
        r[1] = v - T(v_obs_);
        return true;
    }

    int jid_;
    double u_obs_, v_obs_;
    double fx_, fy_, cx_, cy_;
    const ark::AvatarModel* model_;
};

// === 3) Single-stage solver (no priors), AutoDiff ===
inline std::pair<bool, std::string>
OptimizeAllReprojection_AutoDiff(const ark::AvatarModel& model,
                                 ark::Avatar& avatar_io,               // for init + final writeback
                                 const std::vector<PixelKP>& kps,
                                 double fx, double fy, double cx, double cy,
                                 const std::vector<int>& valid_joint_ids,
                                 int max_iters,
                                 // outputs:
                                 double* s_out, double Raa_out[3], double t_out[3])
{
    const int nJ = model.numJoints();
    const int nS = model.numShapeKeys();
    const int nTheta = 3 * std::max(0, nJ - 1);

    // --- parameters (separate blocks so Ceres sees the structure)
    double s = 0.7;
    double Raa[3] = {0,0,0};
    double t[3]   = { avatar_io.p.x(), avatar_io.p.y(), avatar_io.p.z() };

    std::vector<double> theta(nTheta, 0.0);                 // AA for joints 1..nJ-1
    std::vector<double> w(std::max(0, nS), 0.0);
    if (avatar_io.w.size() == nS) {
        for (int i = 0; i < nS; ++i) w[i] = avatar_io.w[i];
    }

    ceres::Problem problem;
    auto* huber = new ceres::HuberLoss(3.0);

    for (const auto& kp : kps) {
        if (kp.jid < 0 || kp.jid >= nJ) continue;
        if (!valid_joint_ids.empty() &&
            std::find(valid_joint_ids.begin(), valid_joint_ids.end(), kp.jid) == valid_joint_ids.end()) {
            continue;
        }
        auto* fun = new ReprojCostAD(kp.jid, kp.u, kp.v, fx, fy, cx, cy, &model);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<ReprojCostAD>(fun);
        cost->AddParameterBlock(1);        // s
        cost->AddParameterBlock(3);        // Raa
        cost->AddParameterBlock(3);        // t
        cost->AddParameterBlock(nTheta);   // theta
        cost->AddParameterBlock(nS);       // w
        cost->SetNumResiduals(2);
        problem.AddResidualBlock(cost, huber, &s, Raa, t, theta.data(), w.data());
    }

    // keep scale sane
    problem.SetParameterLowerBound(&s, 0, 0.3);
    problem.SetParameterUpperBound(&s, 0, 3.0);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = std::max(1, max_iters);
    opts.num_threads = std::thread::hardware_concurrency();
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // --- unpack
    *s_out = s;
    Raa_out[0] = Raa[0]; Raa_out[1] = Raa[1]; Raa_out[2] = Raa[2];
    t_out[0]   = t[0];   t_out[1]   = t[1];   t_out[2]   = t[2];

    // write back pose/shape to avatar (so your overlay/render uses them)
    if (avatar_io.w.size() != nS) avatar_io.w.resize(nS);
    for (int i = 0; i < nS; ++i) avatar_io.w[i] = w[i];

    avatar_io.r.assign(nJ, Eigen::Matrix3d::Identity());
    for (int j = 1; j < nJ; ++j) {
        Eigen::Vector3d aa(theta[3*(j-1)+0], theta[3*(j-1)+1], theta[3*(j-1)+2]);
        const double th = aa.norm();
        if (th > 1e-12) avatar_io.r[j] = Eigen::AngleAxisd(th, aa/th).toRotationMatrix();
    }
    // avatar_io.p is NOT set from t (t is a camera/world translation). Keep p at 0 for model-space.
    avatar_io.update();

    return { summary.IsSolutionUsable(), summary.FullReport() };
}

} // namespace unified_ad
