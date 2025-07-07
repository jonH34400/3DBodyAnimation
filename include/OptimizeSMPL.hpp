#pragma once

#include <ceres/ceres.h>
#include <vector>
#include <smplx/smplx.hpp>
#include "ReprojCost.hpp"

struct PixelKP {
    int jid;
    double u, v;
};

struct SMPLFitResult {
    double theta[72];
    double beta[10];
    double trans[3];
    ceres::Solver::Summary summary;
};

SMPLFitResult optimize_smpl(
    const std::vector<PixelKP>& kps,
    smplx::Model<smplx::model_config::SMPL_v1>& model,
    int max_iters = 50);
