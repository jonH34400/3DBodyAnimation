#pragma once
#include "Sim3BA.h"          
#include <ceres/ceres.h>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Per-frame pose container                                           */
/* ------------------------------------------------------------------ */
struct FramePoseParams {
    double  scale;                 // Sim3 scale
    double  rootAA[3];             // root angle-axis
    double  rootT[3];              // root translation
    std::vector<std::array<double,3>> jointAA;   // size == nJ
};

/* ------------------------------------------------------------------ */
/*  First-order temporal smoothness on a 3-vector                      */
/*  r = weight * (a − b)                                               */
/* ------------------------------------------------------------------ */
struct Vec3DiffCost {
    explicit Vec3DiffCost(double weight) : weight_(weight) {}
    template <typename T>
    bool operator()(const T* a, const T* b, T* r) const {
        for (int i = 0; i < 3; ++i) r[i] = (a[i] - b[i]) * T(weight_);
        return true;
    }
    const double weight_;
};

/* ------------------------------------------------------------------ */
/*  Multi-frame BA with shared shape, per-frame pose, temporal prior   */
/* ------------------------------------------------------------------ */
inline std::pair<bool,std::string>
OptimizeMultiFrame(const ark::AvatarModel&                  model,
                   const std::vector<ark::Avatar*>&         avatars,
                   const std::vector<std::vector<PixelKP>>& kps_vec,
                   double fx,double fy,double cx,double cy,
                   const std::vector<int>&                  valid_ids,
                   std::vector<FramePoseParams>&            poses,
                   double betaPose,
                   double betaShape,
                   double lambdaTemp,
                   int    max_iters = 100)
{
    const int F  = static_cast<int>(avatars.size());
    const int nJ = model.numJoints();
    const int nS = model.numShapeKeys();

    /* ------------ skeleton topology & base offsets ------------- */
    std::vector<int> parent(nJ);
    for (int j = 0; j < nJ; ++j) parent[j] = model.parent[j];

    ark::Avatar base(model); base.w.setZero(); base.update();
    Eigen::Matrix<double,3,Eigen::Dynamic> J0 = base.jointPos;
    const Eigen::Vector3d rootOff = J0.col(0);
    for (int j = 0; j < nJ; ++j) J0.col(j) -= rootOff;

    std::vector<Eigen::Vector3d> offset(nJ, Eigen::Vector3d::Zero());
    for (int j = 1; j < nJ; ++j)
        offset[j] = (J0.col(j) - J0.col(parent[j])).eval();

    /* ------------ Ceres problem ------------------------------- */
    ceres::Problem problem;
    auto* huber = new ceres::HuberLoss(3.0);

    /* ---- shared shape block ---- */
    double* w_block = avatars.front()->w.data();
    if (nS > 0) problem.AddParameterBlock(w_block, nS);

    /* ---- per-frame blocks & residuals ---- */
    for (int f = 0; f < F; ++f) {
        FramePoseParams& P = poses[f];

        problem.AddParameterBlock(&P.scale,   1);
        problem.AddParameterBlock(P.rootAA,   3);
        problem.AddParameterBlock(P.rootT,    3);
        for (int j = 1; j < nJ; ++j)
            problem.AddParameterBlock(P.jointAA[j].data(), 3);

        for (const auto& kp : kps_vec[f]) {
            if (!valid_ids.empty() &&
                std::find(valid_ids.begin(), valid_ids.end(), kp.jid) == valid_ids.end())
                continue;

            auto* fun = new ReprojCostShape(
                kp.jid, kp.u, kp.v, fx, fy, cx, cy,
                parent, offset, avatars[f]->r[0],
                nJ, (betaShape > 0.0 ? &model.jointShapeReg : nullptr));

            auto* cost = new ceres::DynamicAutoDiffCostFunction<ReprojCostShape>(fun);
            cost->AddParameterBlock(1);                  // scale
            cost->AddParameterBlock(3);                  // rootAA
            cost->AddParameterBlock(3);                  // rootT
            for (int j = 1; j < nJ; ++j) cost->AddParameterBlock(3);
            if (nS > 0) cost->AddParameterBlock(nS);     // shape
            cost->SetNumResiduals(2);

            std::vector<double*> blocks = { &P.scale, P.rootAA, P.rootT };
            for (int j = 1; j < nJ; ++j) blocks.push_back(P.jointAA[j].data());
            if (nS > 0) blocks.push_back(w_block);

            problem.AddResidualBlock(cost, huber, blocks);
        }

        /* pose prior */
        if (betaPose > 0.0 && nJ > 1) {
            std::vector<double*> poseBlocks;
            for (int j = 1; j < nJ; ++j) poseBlocks.push_back(P.jointAA[j].data());
            auto* prior = new PosePriorAAAnalytic(nJ - 1, betaPose, nullptr);
            problem.AddResidualBlock(prior, nullptr, poseBlocks);
        }
    }

    /* ---- shape prior (or lock) ---- */
    if (nS > 0 && betaShape > 0.0) {
        auto* shp = new ShapePriorL2Analytic(nS, betaShape);
        problem.AddResidualBlock(shp, nullptr, w_block);
    }

    /* ---- temporal smoothness ---- */
    if (lambdaTemp > 0.0 && F > 1) {
        for (int f = 0; f < F - 1; ++f) {
            FramePoseParams& A = poses[f];
            FramePoseParams& B = poses[f + 1];

            auto* diffRT = new ceres::AutoDiffCostFunction<Vec3DiffCost,3,3,3>(
                               new Vec3DiffCost(lambdaTemp));
            problem.AddResidualBlock(diffRT, nullptr, A.rootT, B.rootT);

            auto* diffRA = new ceres::AutoDiffCostFunction<Vec3DiffCost,3,3,3>(
                               new Vec3DiffCost(lambdaTemp));
            problem.AddResidualBlock(diffRA, nullptr, A.rootAA, B.rootAA);

            for (int j = 1; j < nJ; ++j) {
                auto* diffJ = new ceres::AutoDiffCostFunction<Vec3DiffCost,3,3,3>(
                                  new Vec3DiffCost(lambdaTemp));
                problem.AddResidualBlock(diffJ, nullptr,
                                         A.jointAA[j].data(),
                                         B.jointAA[j].data());
            }
        }
    }

    /* ---- solve ---- */
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.num_threads        = 8;
    opts.minimizer_progress_to_stdout = true;       
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    /* ---- write-back to Avatars ---- */
    for (int f = 0; f < F; ++f) {
        FramePoseParams& P = poses[f];

        Eigen::Vector3d aa(P.rootAA[0], P.rootAA[1], P.rootAA[2]);
        double theta = aa.norm();
        Eigen::Matrix3d Rroot = Eigen::Matrix3d::Identity();
        if (theta > 1e-12)
            Rroot = Eigen::AngleAxisd(theta, aa / theta).toRotationMatrix();

        avatars[f]->r[0] = Rroot * avatars[f]->r[0];
        avatars[f]->p    = { P.rootT[0], P.rootT[1], P.rootT[2] };

        for (int j = 1; j < nJ; ++j) {
            Eigen::Vector3d a(P.jointAA[j][0], P.jointAA[j][1], P.jointAA[j][2]);
            double t = a.norm();
            avatars[f]->r[j] = (t > 1e-12)
                               ? Eigen::AngleAxisd(t, a / t).toRotationMatrix()
                               : Eigen::Matrix3d::Identity();
        }
        avatars[f]->update();
    }

    return { summary.IsSolutionUsable(), summary.BriefReport() };
}
