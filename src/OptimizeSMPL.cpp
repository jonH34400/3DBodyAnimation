#include "OptimizeSMPL.hpp"
#include "ReprojCost.hpp"


SMPLFitResult optimize_smpl(
    const std::vector<PixelKP>& kps,
    smplx::Model<smplx::model_config::SMPL_v1>& model,
    int max_iters)
{
    SMPLFitResult result;
    std::fill_n(result.theta, 72, 0.0);
    std::fill_n(result.beta, 10, 0.0);
    result.trans[0] = 0;
    result.trans[1] = 0;
    result.trans[2] = 2;

    ceres::Problem pb;

    for(const auto& kp : kps){
        auto* cost = new ceres::NumericDiffCostFunction<
                         ReprojCost, ceres::CENTRAL, 2, 72, 10, 3>(
                         new ReprojCost(kp.jid, kp.u, kp.v, model));
        pb.AddResidualBlock(cost, nullptr, result.theta, result.beta, result.trans);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary sum;
    ceres::Solve(opts, &pb, &sum);
    result.summary = sum;
    return result;
}
