#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <smplx/smplx.hpp>

constexpr double FX = 1000, FY = 1000, CX = 640, CY = 360;

// General: si T es double, simplemente devuelve T
template <typename T>
double JetExtract(const T& val) {
    return static_cast<double>(val);  // Para tipos escalares normales
}

template <typename T, int N>
double JetExtract(const ceres::Jet<T, N>& val) {
    return val.a;
}

struct ReprojCost {
    ReprojCost(int jid, double u_px, double v_px,
               smplx::Model<smplx::model_config::SMPL_v1>& model)
        : jid_(jid), u_(u_px), v_(v_px), model_(model) {}

    template <typename T>
    bool operator()(const T* const theta,
                    const T* const beta,
                    const T* const trans,
                    const T* const scale,
                    T* residual) const
    {
        // ------------------------------
        // 1. Convert Jet<T,N> to double
        // ------------------------------
        Eigen::Matrix<double, 72, 1> theta_d;
        Eigen::Matrix<double, 10, 1> beta_d;
        Eigen::Matrix<double, 3, 1> trans_d;

        for (int i = 0; i < 72; ++i)
            theta_d[i] = JetExtract(theta[i]);
        for (int i = 0; i < 10; ++i)
            beta_d[i] = JetExtract(beta[i]);
        for (int i = 0; i < 3; ++i)
            trans_d[i] = JetExtract(trans[i]);

        // ------------------------------
        // 2. Forward model (in double)
        // ------------------------------
        smplx::Body<smplx::model_config::SMPL_v1> body(model_);
        body.set_zero();
        body.shape().head<10>() = beta_d;
        body.pose() = theta_d;
        body.trans() = trans_d;
        body.update();

        Eigen::Vector3d P = body.joints().row(jid_);

        // ------------------------------
        // 3. Projection (AutoDiff active)
        // ------------------------------
        T s = scale[0];
        const T Xc = s * T(P.x());
        const T Yc = s * T(P.y());
        const T Zc = s * T(P.z());

        const T u_proj = FX * Xc / Zc + CX;
        const T v_proj = FY * -Yc / Zc + CY;

        residual[0] = u_proj - T(u_);
        residual[1] = v_proj - T(v_);
        return true;
    }

private:
    int jid_;
    double u_, v_;
    smplx::Model<smplx::model_config::SMPL_v1>& model_;
};