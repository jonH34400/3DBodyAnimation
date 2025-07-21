#include "OptimizeSMPL.hpp"
#include "ReprojCost.hpp"
#include <random>
#include <fstream>  // << Esta línea es necesaria


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* TRYS FOR SOLVING THE CERES AUTODIFF******************************************************************/
/////////////////////////////////////////////////////////////////////////////////////////////////////////

struct LossLogger : public ceres::IterationCallback {
    std::vector<double>& losses;

    explicit LossLogger(std::vector<double>& loss_storage)
        : losses(loss_storage) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        losses.push_back(summary.cost);
        return ceres::SOLVER_CONTINUE;
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

SMPLFitResult optimize_smpl(const std::vector<PixelKP>&              kps,
                            smplx::Model<smplx::model_config::SMPL_v1>& model,
                            int max_iters)
{
    SMPLFitResult result{};
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<double> loss_curve;                      // << AÑADIR
    LossLogger* logger = new LossLogger(loss_curve);     // << AÑADIR
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
        std::mt19937 rng(
            static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::normal_distribution<double> N01(0.0, 0.5);
        for (double& b : result.beta) b = N01(rng);
        std::normal_distribution<double> yaw(0.0, 5.0 * M_PI / 180.0);
        result.theta[0] = yaw(rng);
    }
    
    //std::fill_n(result.beta, 10, 0.0);  // ← This erases your random initialization!
    //std::fill_n(result.theta, 72, 0.0); // ← Same problem
    result.trans[0] = 0.0;
    result.trans[1] = 0.0;
    result.trans[2] = 2.0;
    result.scale = 1.0; 
    double s_log = std::log(result.scale);     
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* ---------- 1. common numeric-diff options ---------- */
    //ceres::NumericDiffOptions nd_opt;
    //nd_opt.relative_step_size                     = 1e-6;   // ← big enough for float
    //nd_opt.ridders_relative_initial_step_size     = 1e-6;   // idem
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* ---------- 2. problem set-up ---------- */
    ceres::Problem pb;
    for (const auto& kp : kps)
    {
        auto* cost = new ceres::AutoDiffCostFunction<ReprojCost, 2, 72, 10, 3, 1>(
            new ReprojCost(kp.jid, kp.u, kp.v, model));

        ceres::LossFunction* loss = new ceres::HuberLoss(5.0);  // δ in pixels
        pb.AddResidualBlock(cost, loss, result.theta, result.beta, result.trans, &result.scale);
        
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
    opts.max_num_iterations              = 100;
    opts.minimizer_progress_to_stdout    = true;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    opts.update_state_every_iteration = true;            // << NECESARIO para que los callbacks funcionen
    opts.callbacks.push_back(logger);                    // << AÑADIR CALLBACK
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    ceres::Solver::Summary sum;
    ceres::Solve(opts, &pb, &sum);

    result.summary = sum;
    std::cout << sum.FullReport() << "\n";

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Guardar la curva de pérdida para graficar
    std::ofstream loss_file("loss_curve.txt");
    loss_file << "iteration,loss\n";
    for (size_t i = 0; i < loss_curve.size(); ++i) {
        loss_file << i << "," << loss_curve[i] << "\n";
    }
    loss_file.close();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

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
