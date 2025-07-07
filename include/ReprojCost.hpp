#pragma once
#include <Eigen/Core>
#include <smplx/smplx.hpp>

// ── pin-hole intrinsics ------------------------------------------------
constexpr double FX = 1000, FY = 1000, CX = 640, CY = 360;

// ── Numeric-diff reprojection cost ------------------------------------
struct ReprojCost
{
    // Note: we now explicitly use the SMPL_v1 config (10 shape dims)
    ReprojCost(int jid, double u_px, double v_px,
               smplx::Model<smplx::model_config::SMPL_v1>& model)
        : jid_(jid), u_(u_px), v_(v_px), model_(model) {}

    /*  theta72 : pose (72)   */
    /*  beta10  : shape (10)  */
    /*  trans3  : trans (3)   */
    bool operator()(const double* const theta72,
                    const double* const beta10,
                    const double* const trans3,
                    double* residual) const
    {
        using S   = smplx::Scalar;           // typically float
        using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

        // Build a temporary body with the SMPL_v1 config
        smplx::Body<smplx::model_config::SMPL_v1> body(model_);
        body.set_zero();                      // clear all parameters

        // Shape (300 dims in SMPL-X but only first 10 used here)
        Eigen::Map<const Eigen::Matrix<double,10,1>> beta10_map(beta10);
        body.shape().head<10>() = beta10_map.template cast<S>();

        // Pose
        body.pose() = Eigen::Map<const Eigen::Matrix<double,72,1>>(theta72)
                          .template cast<S>();

        // Translation
        body.trans() = Eigen::Map<const Eigen::Matrix<double,3,1>>(trans3)
                           .template cast<S>();

        body.update();  // compute verts & joints via LBS

        // Project the requested joint
        const auto& J = body.joints();        // (24×3 in S)
        const S X = J(jid_,0), Y = J(jid_,1), Z = J(jid_,2);

        const double u_proj = FX * double(X) / double(Z) + CX;
        const double v_proj = FY * double(Y) / double(Z) + CY;

        residual[0] = u_proj - u_;
        residual[1] = v_proj - v_;
        return true;
    }

private:
    int   jid_;
    double u_, v_;
    smplx::Model<smplx::model_config::SMPL_v1>& model_;
};
