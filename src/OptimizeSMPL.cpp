#include "OptimizeSMPL.hpp"
#include "ReprojCost.hpp"
#include <random>


SMPLFitResult optimize_smpl(const std::vector<PixelKP>&              kps,
                            smplx::Model<smplx::model_config::SMPL_v1>& model,
                            int max_iters)
{
    SMPLFitResult result{};
    {
        std::mt19937 rng(
            static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::normal_distribution<double> N01(0.0, 0.5);
        for (double& b : result.beta) b = N01(rng);
        std::normal_distribution<double> yaw(0.0, 5.0 * M_PI / 180.0);
        result.theta[0] = yaw(rng);
    }
    
    std::fill_n(result.beta , 10, 0.0);
    std::fill_n(result.theta, 72, 0.0);
    result.trans[0] = 0.0;
    result.trans[1] = 0.0;
    result.trans[2] = 2.0;
    double s_log = 0.0;     
    result.scale = 1.0; 
    /* ---------- 1. common numeric-diff options ---------- */
    ceres::NumericDiffOptions nd_opt;
    nd_opt.relative_step_size                     = 1e-2;   // ← big enough for float
    nd_opt.ridders_relative_initial_step_size     = 1e-2;   // idem

    /* ---------- 2. problem set-up ---------- */
    ceres::Problem pb;
    for (const auto& kp : kps)
    {
        auto* cost = new ceres::NumericDiffCostFunction<
            ReprojCost, ceres::CENTRAL,
            2,              // residual dimension
            72, 10, 3, 1    // θ, β, t
        >(
            new ReprojCost(kp.jid, kp.u, kp.v, model),
            ceres::TAKE_OWNERSHIP,
            2,
            nd_opt);

        /* robustify each key-point */
        ceres::LossFunction* loss = new ceres::HuberLoss(5.0);   // δ in pixels

        pb.AddResidualBlock(cost, loss,
                            result.theta,
                            result.beta,
                            result.trans,
                            &result.scale);
        /* optional: keep global scale in a reasonable range */
        result.scale = std::exp(s_log);
        
    }

    pb.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ScalePrior,1,1>(
            new ScalePrior(/*mu=*/1.0, /*sigma=*/0.2)),   // tighter: σ = 0.2
        nullptr,
        &result.scale);
    
    pb.SetParameterLowerBound(result.trans, 2,  2.0);  // Z ≥ 0.5 m
    pb.SetParameterLowerBound(result.trans, 0,  -0.5);  // -1 m ≤ tx ≤ +1 m
    pb.SetParameterUpperBound(result.trans, 0,   0.5);
    pb.SetParameterLowerBound(result.trans, 1,  -0.5);
    pb.SetParameterUpperBound(result.trans, 1,   0.5);

    pb.AddResidualBlock(
    new ceres::AutoDiffCostFunction<CentrePrior, 2, 3>(
        new CentrePrior),
    nullptr,
    result.trans);

    
    pb.AddResidualBlock(
        new ceres::AutoDiffCostFunction<BetaPrior, 10, 10>(
            new BetaPrior(1)),
        nullptr,
        result.beta);

    /* ---------- 3. solver options ---------- */
    ceres::Solver::Options opts;
    opts.linear_solver_type              = ceres::DENSE_QR;
    opts.max_num_iterations              = max_iters;
    opts.minimizer_progress_to_stdout    = true;

    ceres::Solver::Summary sum;
    ceres::Solve(opts, &pb, &sum);

    result.summary = sum;
    return result;
}


// const double scale_prior = 1.0;
        // pb.AddResidualBlock(
        //     new ceres::AutoDiffCostFunction<ScalePrior,1,1>(
        //         new ScalePrior(scale_prior, 0.05)),   // λ = 1/σ²
        //     nullptr, &result.scale);
        
        // pb.SetParameterLowerBound(result.trans, 2,  2.5);  // Z ≥ 0.5 m
        // pb.SetParameterLowerBound(result.trans, 0,  -0.5);  // -1 m ≤ tx ≤ +1 m
        // pb.SetParameterUpperBound(result.trans, 0,   0.5);
        // pb.SetParameterLowerBound(result.trans, 1,  -0.5);
        // pb.SetParameterUpperBound(result.trans, 1,   0.5);

        // pb.AddResidualBlock(
        //     new ceres::AutoDiffCostFunction<CentrePrior,2,3>(new CentrePrior),
        //     nullptr, result.trans);
