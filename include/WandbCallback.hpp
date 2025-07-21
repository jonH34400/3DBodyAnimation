#pragma once
#include "wandbcpp.hpp"
#include <ceres/ceres.h>

class WandbCallback : public ceres::IterationCallback {
 public:
  WandbCallback() {
    wandbcpp::init(
        {.project = "smpl_fitting",
         .name    = "run_" + std::to_string(std::time(nullptr)),
         .tags    = {"ceres", "smpl"}});
  }

  // Called by Ceres after every successfully completed iteration.
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& s) override {
    wandbcpp::log({{"iter",          s.iteration},
                   {"cost",          s.cost},
                   {"cost_change",   s.cost_change},
                   {"grad_norm",     s.gradient_max_norm},
                   {"step_norm",     s.step_norm},
                   {"tr_radius",     s.trust_region_radius},
                   {"iter_time_s",   s.iteration_time_in_seconds}});
    return ceres::SOLVER_CONTINUE;
  }

  ~WandbCallback() override { wandbcpp::finish(); }
};
