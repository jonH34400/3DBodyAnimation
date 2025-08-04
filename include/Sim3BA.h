#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <vector>
#include <algorithm>

struct PixelKP { int jid; double u, v; };

struct Sim3Params {
    double data[7]; // [s, aa(3), t(3)]
    double& scale() { return data[0]; }
    double* aa_root() { return data + 1; }
    double* trans()   { return data + 4; }
    const double& scale() const { return data[0]; }
    const double* aa_root() const { return data + 1; }
    const double* trans()   const { return data + 4; }
};

struct ReprojCostSim3 {
    ReprojCostSim3(const Eigen::Vector3d& J,
                   double u_obs, double v_obs,
                   double fx, double fy, double cx, double cy)
        : J_(J), u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const sim3, T* residuals) const {
        const T& s = sim3[0];
        const T aa[3] = { sim3[1], sim3[2], sim3[3] };
        const T t[3]  = { sim3[4], sim3[5], sim3[6] };

        T Jd[3] = { T(J_(0)), T(J_(1)), T(J_(2)) };
        T Jrot[3];
        ceres::AngleAxisRotatePoint(aa, Jd, Jrot);

        T X[3] = { s * Jrot[0] + t[0],
                   s * Jrot[1] + t[1],
                   s * Jrot[2] + t[2] };

        T u = T(fx_) * X[0] / X[2] + T(cx_);
        T v = T(fy_) * X[1] / X[2] + T(cy_);

        residuals[0] = u - T(u_obs_);
        residuals[1] = v - T(v_obs_);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& J,
                                       double u_obs, double v_obs,
                                       double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojCostSim3, 2, 7>(
            new ReprojCostSim3(J, u_obs, v_obs, fx, fy, cx, cy));
    }

    const Eigen::Vector3d J_;
    const double u_obs_, v_obs_;
    const double fx_, fy_, cx_, cy_;
};

inline std::pair<Sim3Params, std::string>
OptimizeSim3Reprojection(const Eigen::Matrix<double,3,Eigen::Dynamic>& avatarJoints,
                         const std::vector<PixelKP>& kps,
                         double fx, double fy, double cx, double cy,
                         const std::vector<int>& valid_joint_ids = {},
                         const Sim3Params* init = nullptr,
                         int max_iters = 100)
{
    Sim3Params params{};
    params.scale() = 1.0;
    params.aa_root()[0] = params.aa_root()[1] = params.aa_root()[2] = 0.0;
    params.trans()[0] = params.trans()[1] = 0.0;
    params.trans()[2] = 3.0;
    if (init) params = *init;

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(3.0);

    for (const auto& kp : kps) {
        if (kp.jid < 0 || kp.jid >= avatarJoints.cols()) continue;
        if (!valid_joint_ids.empty() &&
            std::find(valid_joint_ids.begin(), valid_joint_ids.end(), kp.jid) == valid_joint_ids.end())
            continue;

        Eigen::Vector3d J = avatarJoints.col(kp.jid);
        ceres::CostFunction* cost = ReprojCostSim3::Create(J, kp.u, kp.v, fx, fy, cx, cy);
        problem.AddResidualBlock(cost, loss, params.data);
    }

    problem.SetParameterLowerBound(params.data, 0, 0.3);
    problem.SetParameterUpperBound(params.data, 0, 3.0);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.num_threads = 4;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    return { params, summary.FullReport() };
}

inline std::pair<Sim3Params, std::string>
OptimizeSim3Reprojection_OnePoint(const Eigen::Matrix<double,3,Eigen::Dynamic>& avatarJoints,
                                  const std::vector<PixelKP>& kps,
                                  double fx, double fy, double cx, double cy,
                                  int use_jid,                // only this joint id
                                  const Sim3Params* init = nullptr,
                                  bool tx_ty_only = true,     // lock others if true
                                  double fixed_tz = 3.0,      // used if tx_ty_only
                                  int max_iters = 50)
{
    Sim3Params params{};
    params.scale() = 1.0;
    params.aa_root()[0] = params.aa_root()[1] = params.aa_root()[2] = 0.0;
    params.trans()[0] = params.trans()[1] = 0.0;
    params.trans()[2] = fixed_tz;
    if (init) params = *init;

    ceres::Problem problem;
    ceres::LossFunction* loss = nullptr; // single point: keep it simple

    // Add only the first residual for the requested joint id.
    bool added = false;
    for (const auto& kp : kps) {
        if (kp.jid != use_jid) continue;
        if (kp.jid < 0 || kp.jid >= avatarJoints.cols()) continue;

        const Eigen::Vector3d J = avatarJoints.col(kp.jid);
        ceres::CostFunction* cost =
            ReprojCostSim3::Create(J, kp.u, kp.v, fx, fy, cx, cy);
        problem.AddResidualBlock(cost, loss, params.data);
        added = true;
        break; // use only one point
    }

    if (!added) {
        return { params, "No matching keypoint found for the requested joint id." };
    }

    if (tx_ty_only) {
        // Keep only tx (idx 4) and ty (idx 5) free; lock s (0), aa(1..3), tz(6).
        std::vector<int> constant_idx = {0, 1, 2, 3, 6};
        problem.SetParameterization(
            params.data,
            new ceres::SubsetParameterization(7, constant_idx));
        // Ensure tz equals the requested fixed value.
        params.trans()[2] = fixed_tz;
    } else {
        // When everything is free, at least bound the scale like in the multi-point case.
        problem.SetParameterLowerBound(params.data, 0, 0.3);
        problem.SetParameterUpperBound(params.data, 0, 3.0);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.num_threads = 1;                 // one residual: QR on a tiny system is fine
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    return { params, summary.FullReport() };
}
