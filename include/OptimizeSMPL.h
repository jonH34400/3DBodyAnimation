#pragma once

#include <ceres/ceres.h>
#include <vector>

struct PixelKP {
    int jid;
    double u, v;
};

struct SMPLFitResult {
    double theta[72];
    double beta[10];
    double trans[3];
    double scale;
    ceres::Solver::Summary summary;
};

struct ScalePrior {
    double mu, inv_sigma2;
    ScalePrior(double m, double sigma)
        : mu(m), inv_sigma2(1.0/(sigma*sigma)) {}
    template<typename T>
    bool operator()(const T* s, T* r) const {
        r[0] = sqrt(inv_sigma2) * (s[0] - T(mu));
        return true;
    }
};

struct CentrePrior {
    template<typename T>
    bool operator()(const T* trans3, T* r) const {
        r[0] = T(0.5) * trans3[0];   // λ = 0.1 → σ ≈ 3 px after projection
        r[1] = T(0.5) * trans3[1];
        return true;
    }
};

struct BetaPrior {
        explicit BetaPrior(double w) : w_(w) {}
        template<typename T>
        bool operator()(const T* b, T* r) const {
            for (int i = 0; i < 10; ++i) r[i] = T(w_) * b[i];
            return true;
        }
        double w_;
    };
