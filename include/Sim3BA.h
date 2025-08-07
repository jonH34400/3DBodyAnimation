#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <vector>
#include <algorithm>

struct PixelKP { int jid; double u, v; };

struct Sim3Params {
    double data[7]; // [s, aa(3), t(3)]
    double& scale()             { return data[0]; }
    double* aa_root()           { return data + 1; }
    double* trans()             { return data + 4; }
    const double& scale() const { return data[0]; }
    const double* aa_root() const { return data + 1; }
    const double* trans()   const { return data + 4; }
};

struct ReprojCostFull {
    ReprojCostFull(int jid,
                   double u_obs, double v_obs,
                   double fx, double fy, double cx, double cy,
                   const std::vector<int>& parent,
                   const std::vector<Eigen::Vector3d>& offset,
                   const Eigen::Matrix3d& R0_init,
                   int nJ,
                   const Eigen::MatrixXd* jointShapeReg = nullptr)
        : jid_(jid), u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          parent_(parent), offset_(offset), R0_(R0_init), nJ_(nJ)
    {
        if (jointShapeReg && jointShapeReg->size() > 0) {
            use_joint_shape_reg_ = true;
            jointShapeRegPtr_ = jointShapeReg;
            nS_ = static_cast<int>(jointShapeReg->cols());
        } else {
            use_joint_shape_reg_ = false;
            jointShapeRegPtr_ = nullptr;
            nS_ = 0;
        }
    }

    template <typename T>
    bool operator()(T const* const* params, T* residuals) const {
        // Parameter blocks layout:
        // params[0]: scale (1)
        // params[1]: rootAA (3)
        // params[2]: rootTrans (3)
        // params[3..(3 + nJ_-2)]: jointAA for joints 1..(nJ_-1) (3 each)
        // params[last] (if use_joint_shape_reg_): shape coefficients (nS_)

        const T* scale   = params[0];
        const T* rootAA  = params[1];
        const T* rootT   = params[2];
        // Helper to get pointer to joint j's angle-axis (for j >= 1)
        auto jointAA = [&](int j)->const T* { return params[3 + (j - 1)]; };

        // Start from this joint's base offset in model space
        Eigen::Matrix<T,3,1> pos_T = offset_[jid_].template cast<T>();

        // If shape is being optimized, add shape-induced offset (delta_j - delta_parent)
        if (use_joint_shape_reg_ && nS_ > 0) {
            const T* shape_params = params[3 + (nJ_ - 1)];  // shape param block (last)
            Eigen::Matrix<T,3,1> delta_j = Eigen::Matrix<T,3,1>::Zero();
            Eigen::Matrix<T,3,1> delta_parent = Eigen::Matrix<T,3,1>::Zero();
            // Compute shape offset for joint j and its parent
            for (int c = 0; c < nS_; ++c) {
                // joint j shape effect
                double jx = (*jointShapeRegPtr_)(3 * jid_ + 0, c);
                double jy = (*jointShapeRegPtr_)(3 * jid_ + 1, c);
                double jz = (*jointShapeRegPtr_)(3 * jid_ + 2, c);
                delta_j(0) += T(jx) * shape_params[c];
                delta_j(1) += T(jy) * shape_params[c];
                delta_j(2) += T(jz) * shape_params[c];
                // parent joint shape effect (if parent exists)
                if (parent_[jid_] >= 0) {
                    int pj = parent_[jid_];
                    double px = (*jointShapeRegPtr_)(3 * pj + 0, c);
                    double py = (*jointShapeRegPtr_)(3 * pj + 1, c);
                    double pz = (*jointShapeRegPtr_)(3 * pj + 2, c);
                    delta_parent(0) += T(px) * shape_params[c];
                    delta_parent(1) += T(py) * shape_params[c];
                    delta_parent(2) += T(pz) * shape_params[c];
                }
            }
            pos_T += (delta_j - delta_parent);
        }

        // Traverse up the kinematic chain (apply rotations and offsets up to, but not including, root)
        int cur = jid_;
        while (parent_[cur] != -1 && parent_[cur] != 0) {
            int p = parent_[cur];
            // Rotate current position by parent p's local rotation
            ceres::AngleAxisRotatePoint(jointAA(p), pos_T.data(), pos_T.data());
            // Translate by parent p's base offset (and shape offset if applicable)
            if (use_joint_shape_reg_ && nS_ > 0) {
                const T* shape_params = params[3 + (nJ_ - 1)];
                Eigen::Matrix<T,3,1> delta_p = Eigen::Matrix<T,3,1>::Zero();
                Eigen::Matrix<T,3,1> delta_parent2 = Eigen::Matrix<T,3,1>::Zero();
                for (int c = 0; c < nS_; ++c) {
                    // shape effect for joint p
                    double px = (*jointShapeRegPtr_)(3 * p + 0, c);
                    double py = (*jointShapeRegPtr_)(3 * p + 1, c);
                    double pz = (*jointShapeRegPtr_)(3 * p + 2, c);
                    delta_p(0) += T(px) * shape_params[c];
                    delta_p(1) += T(py) * shape_params[c];
                    delta_p(2) += T(pz) * shape_params[c];
                    // shape effect for p's parent (ancestor one level up)
                    int pp = parent_[p];
                    if (pp >= 0) {  // (will be 0 or another joint; if 0, we still compute and subtract)
                        double qx = (*jointShapeRegPtr_)(3 * pp + 0, c);
                        double qy = (*jointShapeRegPtr_)(3 * pp + 1, c);
                        double qz = (*jointShapeRegPtr_)(3 * pp + 2, c);
                        delta_parent2(0) += T(qx) * shape_params[c];
                        delta_parent2(1) += T(qy) * shape_params[c];
                        delta_parent2(2) += T(qz) * shape_params[c];
                    }
                }
                pos_T += offset_[p].template cast<T>() + (delta_p - delta_parent2);
            } else {
                pos_T += offset_[p].template cast<T>();
            }
            cur = p;
        }

        // Apply the fixed initial root orientation (R0_)
        Eigen::Matrix<T,3,1> pos_cam;
        pos_cam(0) = T(R0_(0,0)) * pos_T(0) + T(R0_(0,1)) * pos_T(1) + T(R0_(0,2)) * pos_T(2);
        pos_cam(1) = T(R0_(1,0)) * pos_T(0) + T(R0_(1,1)) * pos_T(1) + T(R0_(1,2)) * pos_T(2);
        pos_cam(2) = T(R0_(2,0)) * pos_T(0) + T(R0_(2,1)) * pos_T(1) + T(R0_(2,2)) * pos_T(2);

        // Apply optimized root rotation (rootAA), then scale and translation
        ceres::AngleAxisRotatePoint(rootAA, pos_cam.data(), pos_cam.data());
        pos_cam(0) = (*scale) * pos_cam(0) + rootT[0];
        pos_cam(1) = (*scale) * pos_cam(1) + rootT[1];
        pos_cam(2) = (*scale) * pos_cam(2) + rootT[2];

        // Perspective projection onto image
        T u_pred = T(fx_) * pos_cam(0) / pos_cam(2) + T(cx_);
        T v_pred = T(fy_) * pos_cam(1) / pos_cam(2) + T(cy_);
        residuals[0] = u_pred - T(u_obs_);
        residuals[1] = v_pred - T(v_obs_);
        return true;
    }

    // Data members
    const int jid_;
    const double u_obs_, v_obs_;
    const double fx_, fy_, cx_, cy_;
    const std::vector<int>& parent_;
    const std::vector<Eigen::Vector3d>& offset_;
    const Eigen::Matrix3d R0_;
    const int nJ_;
    const Eigen::MatrixXd* jointShapeRegPtr_;  // pointer to model.jointShapeReg (if available)
    bool use_joint_shape_reg_;
    int nS_;
};

// ===== Pose prior over non-root joints (angle-axis, 3 per joint) =====
// If you have a GaussianMixture prior, pass it in; otherwise pass nullptr and
// we use an L2 fallback (betaPose * aa).
struct PosePriorAAAnalytic : ceres::CostFunction {
    // GaussianMixture must expose:
    //  - VectorXd residual(const VectorXd&, int* compIdx)   // Mahalanobis residual stacked (3*N + 1)
    //  - std::vector<MatrixXd> prec_cho;                    // L s.t. Precision = L * L^T
    struct GaussianMixture {
        // Minimal interface placeholder; replace with your real type if available.
        Eigen::VectorXd residual(const Eigen::VectorXd&, int* compIdx) const { *compIdx = 0; return Eigen::VectorXd(); }
        std::vector<Eigen::MatrixXd> prec_cho;
        bool valid = false;
    };

    PosePriorAAAnalytic(int nNonRootJoints, double beta_pose,
                        const GaussianMixture* gmm_prior = nullptr)
        : nJ_(nNonRootJoints),
          betaPose_(beta_pose),
          gmm_(gmm_prior) 
    {
        // Residuals: 3 per joint (+1 if using GMM for the mixture constant)
        const int nRes = gmm_ && gmm_->valid ? (nJ_ * 3 + 1) : (nJ_ * 3);
        set_num_residuals(nRes);
        auto* sizes = mutable_parameter_block_sizes();
        for (int i = 0; i < nJ_; ++i) sizes->push_back(3); // angle-axis per non-root joint
    }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const final
    {
        const bool use_gmm = (gmm_ && gmm_->valid);
        const int nRes = use_gmm ? (nJ_ * 3 + 1) : (nJ_ * 3);

        // Stack AA into a single vector x \in R^{3*nJ}
        Eigen::VectorXd x(3 * nJ_);
        for (int j = 0; j < nJ_; ++j) {
            x.segment<3>(3*j) = Eigen::Map<const Eigen::Vector3d>(parameters[j]);
        }

        int compIdx = 0;
        Eigen::Map<Eigen::VectorXd> r(residuals, nRes);

        if (use_gmm) {
            // GMM Mahalanobis residual (already whitened by component precision)
            r.noalias() = gmm_->residual(x, &compIdx) * betaPose_;
        } else {
            // Simple L2 prior: r = beta * x  (no +1 row)
            r.head(3*nJ_).noalias() = x * betaPose_;
        }

        if (jacobians) {
            if (use_gmm) {
                const Eigen::MatrixXd& L = gmm_->prec_cho[compIdx];
                // Precision = L * L^T ; residual ~ L^T * (x - mu) with scaling baked into residual()
                for (int j = 0; j < nJ_; ++j) {
                    if (jacobians[j]) {
                        // Each block is (nRes x 3); fill top (3*nJ_) rows; last row (mixture const) is zero.
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> J(jacobians[j], nRes, 3);
                        J.setZero();
                        // Take the 3 rows of L corresponding to this joint, transpose them
                        // (implementation detail depends on how residual() is formed; this matches the
                        // typical "whitened" residual stacking by joints).
                        J.topRows(3*nJ_).noalias() =
                            L.middleRows(3*j, 3).transpose() * betaPose_;
                    }
                }
            } else {
                // d(beta * x)/d x_j = beta * I_3
                for (int j = 0; j < nJ_; ++j) {
                    if (jacobians[j]) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> J(jacobians[j], nRes, 3);
                        J.setZero();
                        J.topRows(3*nJ_).block(3*j, 0, 3, 3).setIdentity();
                        J.topRows(3*nJ_) *= betaPose_;
                    }
                }
            }
        }
        return true;
    }

    const int nJ_;
    const double betaPose_;
    const GaussianMixture* gmm_; // optional
};

// ===== Shape prior: L2 on shape coefficients w =====
struct ShapePriorL2Analytic : ceres::CostFunction {
    ShapePriorL2Analytic(int num_shape_keys, double beta_shape)
        : nS_(num_shape_keys), betaShape_(beta_shape)
    {
        set_num_residuals(nS_);
        mutable_parameter_block_sizes()->push_back(nS_);
    }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const final
    {
        Eigen::Map<Eigen::VectorXd> r(residuals, nS_);
        Eigen::Map<const Eigen::VectorXd> w(parameters[0], nS_);
        r.noalias() = w * betaShape_;
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::MatrixXd> J(jacobians[0], nS_, nS_);
            J.setZero();
            J.diagonal().setConstant(betaShape_);
        }
        return true;
    }
    const int nS_;
    const double betaShape_;
};

inline std::pair<bool, std::string>
OptimizePoseReprojectionWithShape(const ark::AvatarModel& model,
                                  ark::Avatar& avatar,
                                  const std::vector<PixelKP>& kps,
                                  double fx, double fy, double cx, double cy,
                                  const std::vector<int>& valid_joint_ids,
                                  Sim3Params& initSim3,
                                  int max_iters = 100,
                                  double betaPose = 0.0,
                                  double betaShape = 0.0,
                                  const PosePriorAAAnalytic::GaussianMixture* gmmPosePrior = nullptr)
{
    int nJ = model.numJoints();
    int nS = model.numShapeKeys();
    // Prepare parent index list
    std::vector<int> parent(nJ);
    for (int j = 0; j < nJ; ++j) parent[j] = model.parent[j];

    // Compute base (zero-pose, zero-shape) joint positions
    ark::Avatar base_avatar(model);
    base_avatar.w.setZero();                            // zero shape
    base_avatar.p = Eigen::Vector3d::Zero();
    base_avatar.r.assign(nJ, Eigen::Matrix3d::Identity());
    base_avatar.update(ark::UpdateMode::JointsOnly);
    Eigen::Matrix<double,3,Eigen::Dynamic> baseJoints = base_avatar.jointPos;
    // Translate so that root joint is at origin (anchor the skeleton at root)
    Eigen::Vector3d root_offset = baseJoints.col(0);
    for (int j = 0; j < nJ; ++j) {
        baseJoints.col(j) -= root_offset;
    }
    // Compute offset vectors from each joint’s parent in the base pose
    std::vector<Eigen::Vector3d> offset(nJ);
    offset[0] = Eigen::Vector3d::Zero();
    for (int j = 1; j < nJ; ++j) {
        int pj = parent[j];
        for (int j = 1; j < nJ; ++j) {
            int pj = parent[j];
            if (pj >= 0) {
                offset[j] = (baseJoints.col(j) - baseJoints.col(pj)).eval();
            } else {
                offset[j] = baseJoints.col(j);
            }
        }

    }

    // Fixed initial root orientation (combination of yaw 180° + flipY from avatar.r[0])
    Eigen::Matrix3d R0_init = avatar.r[0];

    // Initialize optimization variables (scale, root rotation AA, root translation, joint AAs, shape)
    double scale = initSim3.scale();
    double rootAA[3]  = { initSim3.aa_root()[0], initSim3.aa_root()[1], initSim3.aa_root()[2] };
    double rootTrans[3] = { initSim3.trans()[0], initSim3.trans()[1], initSim3.trans()[2] };
    std::vector<std::array<double,3>> jointAA(nJ);
    for (int j = 0; j < nJ; ++j) {
        jointAA[j] = {0.0, 0.0, 0.0};  // start from zero-angle for all joints
    }

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(3.0);

    // Add residual blocks for each observed keypoint
    for (const PixelKP& kp : kps) {
        if (!valid_joint_ids.empty() &&
            std::find(valid_joint_ids.begin(), valid_joint_ids.end(), kp.jid) == valid_joint_ids.end()) {
            continue;  // skip keypoints not in the allowed list
        }
        int jid = kp.jid;
        // Prepare cost functor (pass jointShapeReg if shape is being optimized)
        const Eigen::MatrixXd* jsRegPtr = (betaShape > 0.0 && model.useJointShapeRegressor ? &model.jointShapeReg : nullptr);
        auto* reprojFunctor = new ReprojCostFull(jid, kp.u, kp.v, fx, fy, cx, cy,
                                                parent, offset, R0_init, nJ, jsRegPtr);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<ReprojCostFull>(reprojFunctor);
        // Define parameter block sizes in the same order as in operator():
        cost->AddParameterBlock(1);          // scale
        cost->AddParameterBlock(3);          // root angle-axis
        cost->AddParameterBlock(3);          // root translation
        for (int j = 1; j < nJ; ++j) {
            cost->AddParameterBlock(3);      // joint j angle-axis
        }
        if (betaShape > 0.0 && model.useJointShapeRegressor && nS > 0) {
            cost->AddParameterBlock(nS);     // shape coefficients (w)
        }
        cost->SetNumResiduals(2);

        // Set up parameter pointers for this residual
        std::vector<double*> param_blocks;
        param_blocks.reserve(3 + (nJ - 1) + (betaShape > 0.0 ? 1 : 0));
        param_blocks.push_back(&scale);
        param_blocks.push_back(rootAA);
        param_blocks.push_back(rootTrans);
        for (int j = 1; j < nJ; ++j) {
            param_blocks.push_back(jointAA[j].data());
        }
        if (betaShape > 0.0 && model.useJointShapeRegressor && nS > 0) {
            param_blocks.push_back(avatar.w.data());  // shape parameter block
        }

        problem.AddResidualBlock(cost, loss, param_blocks);
    }

    // Constrain scale to a reasonable range
    problem.SetParameterLowerBound(&scale, 0, 0.3);
    problem.SetParameterUpperBound(&scale, 0, 3.0);

    // Add pose prior on joint angle-axis if betaPose > 0
    if (betaPose > 0.0) {
        int nAngles = std::max(0, nJ - 1);
        auto* posePrior = new PosePriorAAAnalytic(nAngles, betaPose, gmmPosePrior);
        std::vector<double*> pose_params;
        pose_params.reserve(nAngles);
        for (int j = 1; j < nJ; ++j) {
            pose_params.push_back(jointAA[j].data());
        }
        problem.AddResidualBlock(posePrior, nullptr, pose_params);
    }

    // Add shape prior (L2 regularizer on shape coefficients) if betaShape > 0
    if (betaShape > 0.0 && nS > 0) {
        problem.AddParameterBlock(avatar.w.data(), nS);  // declare shape param block
        auto* shapePrior = new ShapePriorL2Analytic(nS, betaShape);
        problem.AddResidualBlock(shapePrior, nullptr, avatar.w.data());
    }

    // Solve the Ceres optimization
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.num_threads = 4;
    opts.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // Update the Avatar with optimized parameters
    // Root orientation: convert optimized rootAA to rotation matrix and apply to initial R0
    Eigen::Vector3d aa_root_vec(rootAA[0], rootAA[1], rootAA[2]);
    double theta = aa_root_vec.norm();
    Eigen::Matrix3d R_root = Eigen::Matrix3d::Identity();
    if (theta > 1e-12) {
        R_root = Eigen::AngleAxisd(theta, aa_root_vec / theta).toRotationMatrix();
    }
    avatar.r[0] = R_root * avatar.r[0];
    // Set each joint rotation from optimized angle-axis
    for (int j = 1; j < nJ; ++j) {
        Eigen::Vector3d aa_j(jointAA[j][0], jointAA[j][1], jointAA[j][2]);
        double th = aa_j.norm();
        avatar.r[j] = (th > 1e-12 ? Eigen::AngleAxisd(th, aa_j / th).toRotationMatrix()
                                  : Eigen::Matrix3d::Identity());
    }
    // Update root translation
    avatar.p = Eigen::Vector3d(rootTrans[0], rootTrans[1], rootTrans[2]);

    // Save optimized Sim3 parameters back to initSim3 (for logging or subsequent use)
    initSim3.scale()      = scale;
    initSim3.aa_root()[0] = rootAA[0];
    initSim3.aa_root()[1] = rootAA[1];
    initSim3.aa_root()[2] = rootAA[2];
    initSim3.trans()[0]   = rootTrans[0];
    initSim3.trans()[1]   = rootTrans[1];
    initSim3.trans()[2]   = rootTrans[2];

    bool success = summary.IsSolutionUsable();
    return { success, summary.FullReport() };
}