#include "../include/FileWriter/CsvFileWriter.hpp"

void CsvFileWriter::saveTestDatasetAndPredictions(size_t simulation_num,
                                                  const std::vector<std::vector<double>> &test_dataset,
                                                  const std::vector<double> &test_probs,
                                                  const std::vector<double> &preds)
{

    // Create file name
    std::string file_name = std::to_string(simulation_num);
    file_name.append(".csv");

    // Create file
    std::ofstream my_file;
    my_file.open(file_name);

    // Create HEADER row
    size_t features_num = test_dataset[0].size()-1;
    for (size_t i = 0; i < features_num; i++)
    {
        std::string col_name = "x";
        col_name.append(std::to_string(i+1));
        my_file << col_name << ",";
    }
    my_file << "y,y_pred,p\n";

    // Filling up the rows with values
    for (size_t i = 0; i < test_dataset.size(); i++)
    {
        for (size_t j = 0; j < test_dataset[0].size(); j++)
        {
            my_file << std::fixed << std::setprecision(3);
            my_file << test_dataset[i][j] << ",";
        }
        my_file << preds[i] <<"," << test_probs[i] << "\n";
    }

    my_file.close();
}

void CsvFileWriter::saveWholeDataset(size_t simulation_num,
                                     const std::unique_ptr<RandomDatasetGenerator> &rdGenerator)
{
    // Create file name
    std::string file_name = std::to_string(simulation_num);
    file_name.append(".csv");

    // Create file
    std::ofstream my_file;
    my_file.open(file_name);

    const auto features = rdGenerator->getFeatures();
    const auto features_accessor = features.accessor<double, 2>();
    const auto target = rdGenerator->getTarget();
    const auto target_accessor = target.accessor<double, 2>();

    // Create HEADER row
    size_t features_num = features_accessor.size(1);
    for (size_t i = 0; i < features_num; i++)
    {
        std::string col_name = "x";
        col_name.append(std::to_string(i+1));
        my_file << col_name << ",";
    }
    my_file << "y\n";

    // Filling up the rows with values
    for (size_t i = 0; i < features_accessor.size(0); i++)
    {
        for (size_t j = 0; j < features_accessor.size(1); j++)
        {
            my_file << std::fixed << std::setprecision(3);
            my_file << features_accessor[i][j] << ",";
        }
        my_file << target_accessor[i][0] << "\n";
    }

    my_file.close();
}

void CsvFileWriter::saveMetricsResults(const std::vector<std::vector<double>> &result,
                                       const std::vector<double> &thresholds,
                                       const std::string &metric_name)
{
    // Create file name
    std::string file_name = metric_name;
    file_name.append(".csv");

    // Create file
    std::ofstream my_file;
    my_file.open(file_name);


    // Creating the HEADER part of the csv file
    for (size_t i = 0; i < thresholds.size(); i++)
    {
        my_file << thresholds[i] << "_" << metric_name;

        // If the current index is not the last one, then we append a comma
        if (i != thresholds.size()-1)
            my_file << ",";
    }

    my_file << "\n";

    // Filling up all the rows with the generated results
    for (const auto &vec : result)
    {
        for (size_t i = 0; i < vec.size(); i++)
        {
            my_file << std::fixed << std::setprecision(3);
            my_file << vec[i];

            // If the current index is not the last one, then we append a comma
            if (i != vec.size()-1)
                my_file << ",";
        }
        my_file << "\n";
    }

    my_file.close();
}
