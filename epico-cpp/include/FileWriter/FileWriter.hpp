#pragma once

#include <vector>
#include <memory>

#include "RandomDatasetGenerator.hpp"

class FileWriter {
public:
    virtual void saveTestDatasetAndPredictions(size_t simulation_num,
                                                    const std::vector<std::vector<double>> &test_dataset,
                                                    const std::vector<double> &test_probs,
                                                    const std::vector<double> &preds) = 0;

    virtual void saveWholeDataset(size_t simulation_num,
                                  const std::unique_ptr<RandomDatasetGenerator> &rdGenerator) = 0;

    virtual void saveMetricsResults(const std::vector<std::vector<double>> &result,
                                    const std::vector<double> &thresholds,
                                    const std::string &metric_name) = 0;

    virtual ~FileWriter() = default;
};