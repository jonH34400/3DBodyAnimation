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
    bool operator()(const double* theta72,
                    const double* beta10,
                    const double* trans3,
                    const double* scale1,
                    double* residual) const
    {
        using S = smplx::Scalar;

        // ---------- build body (unchanged) ----------
        smplx::Body<smplx::model_config::SMPL_v1> body(model_);
        body.set_zero();

        body.shape().head<10>() =
            Eigen::Map<const Eigen::Matrix<double,10,1>>(beta10).template cast<S>();
        body.pose()  =
            Eigen::Map<const Eigen::Matrix<double,72,1>>(theta72).template cast<S>();
        body.trans() =
            Eigen::Map<const Eigen::Matrix<double,3,1>>(trans3 ).template cast<S>();

        body.update();

        // ---------- joint position ----------
        Eigen::Vector3d P = body.joints().row(jid_).cast<double>();   // (X,Y,Z) in metres

        const double s = scale1[0];           // global scale  (>0)

        // apply scale in world space, then translation
        const double Xc = s * P.x();
        const double Yc = s * P.y();
        const double Zc = s * P.z();

        // ---------- pin-hole projection (single Y-flip) ----------
        const double u_proj =  FX *  Xc / Zc + CX;
        const double v_proj =  FY * -Yc / Zc + CY;

        // ---------- residual ----------
        residual[0] = u_proj - u_;
        residual[1] = v_proj - v_;
        return true;
    }


private:
    int   jid_;
    double u_, v_;
    smplx::Model<smplx::model_config::SMPL_v1>& model_;
};
