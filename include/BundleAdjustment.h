//
// Created by chuchu on 8/5/26.
//
#pragma once
#include <string>
#include "BundleAdjustment.h"
#include "General.h"
#include  "BaHelper.h"
#include "DataLoader.h"
#include "Optimizer.h"
#include "SIFT.h"


class BundleAdjustment {
public:
  BundleAdjustment(std::string config_path);
  ~BundleAdjustment() =default;
  Status Run();

private:
  Config config_;
  DataLoader  data_loader_;
  BundleAdjustmentOptimizer  optimizer_;

};