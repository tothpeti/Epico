#pragma once
#include "FileWriter.hpp"

#include <string>
#include <fstream>

class CsvFileWriter : public FileWriter {

public:
    explicit CsvFileWriter() = default;

    void saveTestDatasetAndPredictions(size_t simulation_num,
                                               const std::vector<std::vector<double>> &test_dataset,
                                               const std::vector<double> &test_probs,
                                               const std::vector<double> &preds) override;

    void saveWholeDataset(size_t simulation_num,
                          const std::unique_ptr<RandomDatasetGenerator> &rdGenerator) override;



    void saveMetricsResults(const std::vector<std::vector<double>> &result,
                                    const std::vector<double> &thresholds,
                                    const std::string &metric_name) override;

};