#include "RandomDatasetGenerator.hpp"
#include "RandomDataset.hpp"
#include "LogisticRegression.hpp"

/*
 * Helper functions
 */

void write_csv(const std::vector<std::vector<double>> &result, const std::vector<double> &thresholds,
               const std::string &metric_name);
std::vector<double> generate_threshold_values(double from, double to, double stride);


int main() {
    /*
     * Parameters
     */
    const int number_of_rows        = 1000;  // number of rows which the RandomDataset will have
    const int number_of_features    = 10;    // how many columns will be created manually, used for Model
    const int batch_size            = 50;
    const size_t number_of_epochs   = 5;     // how many rounds used for training the model
    const size_t simulation_rounds  = 50;
    const double learning_rate      = 0.001; // used for Model

    // Generating threshold values
    auto thresholds = generate_threshold_values(0.5, 0.95, 0.05);

    std::vector<std::vector<double>> all_sensitivity_results;
    std::vector<std::vector<double>> all_specificity_results;
    std::vector<std::vector<double>> all_precision_results;
    std::vector<std::vector<double>> all_f1score_results;
    std::vector<std::vector<double>> all_accuracy_results;

    all_sensitivity_results.reserve(thresholds.size());
    all_specificity_results.reserve(thresholds.size());
    all_precision_results.reserve(thresholds.size());
    all_f1score_results.reserve(thresholds.size());
    all_accuracy_results.reserve(thresholds.size());

    /*
     *  SIMULATION
     */
    for (const auto &threshold : thresholds) {

        std::vector<double> sensitivity_results;
        std::vector<double> specificity_results;
        std::vector<double> precision_results;
        std::vector<double> f1score_results;
        std::vector<double> accuracy_results;

        // Reserve memories
        sensitivity_results.reserve(simulation_rounds);
        specificity_results.reserve(simulation_rounds);
        precision_results.reserve(simulation_rounds);
        f1score_results.reserve(simulation_rounds);
        accuracy_results.reserve(simulation_rounds);

        /*
         * SIMULATION START HERE
         */
        for (size_t i = 0; i < simulation_rounds; i++) {

            //Creating columns for RandomDataset
            RandomDatasetGenerator::ColumnDataType bern{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                    {{"prob", 0.5}, {"weight", 0.5}}        //parameters
            };
            RandomDatasetGenerator::ColumnDataType bern2{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 0.75}}
            };
            RandomDatasetGenerator::ColumnDataType bern3{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 1}}
            };
            RandomDatasetGenerator::ColumnDataType bern4{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 1.25}}
            };
            RandomDatasetGenerator::ColumnDataType bern5{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 1.5}}
            };

            RandomDatasetGenerator::ColumnDataType bern6{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 1.75}}
            };
            RandomDatasetGenerator::ColumnDataType bern7{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 2}}
            };

            RandomDatasetGenerator::ColumnDataType bern8{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 1.5}}
            };

            RandomDatasetGenerator::ColumnDataType bern9{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 2}}
            };

            RandomDatasetGenerator::ColumnDataType bern10{
                    RandomDatasetGenerator::DistributionTypes::Bernoulli,
                    {{"prob", 0.5}, {"weight", 3}}
            };

            std::vector<RandomDatasetGenerator::ColumnDataType> cols{
                    bern, bern2, bern3, bern4, bern5, bern6, bern7, bern, bern9, bern10
            };


            /*
             * Creating Training and Test Data Loaders
             */

            // Create RandomDataLoader from previously created Columns
            auto rdGenerator = std::make_unique<RandomDatasetGenerator>(number_of_rows, cols, true);

            // Create training RandomDataset from the RandomDatasetGenerator
            auto rdTrain = std::make_unique<RandomDataset>(rdGenerator->getFeatures(), rdGenerator->getTarget(),
                                                           RandomDataset::Mode::kTrain, 0.6);
            auto numberOfTrainSamples = rdTrain->size().value();

            auto trainingDataLoader = torch::data::make_data_loader(
                    std::move(rdTrain->map(torch::data::transforms::Stack<>())),
                    torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
            );

            // Create test RandomDataset from the RandomDatasetGenerator
            auto rdTest = std::make_unique<RandomDataset>(rdGenerator->getFeatures(), rdGenerator->getTarget(),
                                                          RandomDataset::Mode::kTest, 0.4);
            auto numberOfTestSamples = rdTest->size().value();

            auto testingDataLoader = torch::data::make_data_loader(
                    std::move(rdTest->map(torch::data::transforms::Stack<>())),
                    torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
            );


            // Logistic regression model
            LogisticRegression model(number_of_features);

            // Loss and optimizer
            torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

            // Set floating point output precision - USE IF WANT TO PRINT RESULTS!!
            //std::cout << std::fixed << std::setprecision(4);

            // Train the model
            model->train();
            for (size_t epoch = 0; epoch != number_of_epochs; epoch++) {

                // Initializing running metrics
                double runningLoss = 0.0;

                for (auto &batch: *trainingDataLoader) {
                    optimizer.zero_grad();

                    auto data = batch.data.to(torch::kFloat32);
                    auto target = batch.target.to(torch::kFloat32);

                    // Forward pass
                    auto output = model->forward(data);

                    // Calculate loss
                    auto loss = torch::nn::functional::binary_cross_entropy(output, target);

                    // Update running loss
                    runningLoss += loss.item<double>();

                    // Backward pass and optimize
                    loss.backward();
                    optimizer.step();
                }
                //auto sampleMeanLoss = runningLoss / numberOfTrainSamples;
            }

            // Test the model
            model->eval();
            torch::NoGradGuard no_grad;

            double running_loss = 0.0;
            double numberOfCorrect = 0.0;
            double sensitivities = 0.0;
            double specificities = 0.0;
            double precisions = 0.0;
            double f1scores = 0.0;

            for (const auto &batch : *testingDataLoader) {
                auto data = batch.data.to(torch::kFloat32);
                auto target = batch.target.to(torch::kFloat32);

                // Forward pass
                auto output = model->forward(data);

                // Calculate loss
                auto loss = torch::nn::functional::binary_cross_entropy(output, target);
                running_loss += loss.item<double>();

                // Rounding output's values via custom_threshold into 0 or 1
                auto custom_threshold = torch::tensor({threshold});
                auto rounded_output = torch::where(output > custom_threshold, torch::tensor({1}), torch::tensor({0}));

                // Calculate other metrics
                auto tp = (target * rounded_output).sum().to(torch::kFloat64);
                auto tn = ((1 - target) * (1 - rounded_output)).sum().to(torch::kFloat64);
                auto fp = ((1 - target) * rounded_output).sum().to(torch::kFloat64);
                auto fn = (target * (1 - rounded_output)).sum().to(torch::kFloat64);

                auto precision = 0.0;
                auto recall = 0.0;

                // Calculate precision and recall
                if (tp.item<double>() > 0) {
                    precision = (tp / (tp + fp)).to(torch::kFloat64).item<double>();
                    recall = (tp / (tp + fn)).to(torch::kFloat64).item<double>(); // alias sensitivity
                } else if (fp.item<double>() > 0) {
                    precision = (tp / (tp + fp)).to(torch::kFloat64).item<double>();
                } else if (fn.item<double>() > 0) {
                    recall = (tp / (tp + fn)).to(torch::kFloat64).item<double>(); // alias sensitivity
                }

                auto f1_score = 0.0;
                if (precision > 0 || recall > 0)
                    f1_score = (2 * (precision * recall) / (precision + recall));

                auto specificity = 0.0;
                if (tn.item<double>() > 0 || fp.item<double>() > 0)
                    specificity = (tn / (tn + fp)).to(torch::kFloat64).item<double>();

                // For calculating Accuracy
                numberOfCorrect += rounded_output.view({-1, 1}).eq(target).sum().item<int>();

                sensitivities += recall;
                specificities += specificity;
                precisions += precision;
                f1scores += f1_score;
            }

            auto accuracy = (100 * numberOfCorrect) / numberOfTestSamples;
            auto test_sample_mean_loss = running_loss / numberOfTestSamples;

            specificity_results.emplace_back(specificities);
            sensitivity_results.emplace_back(sensitivities);
            precision_results.emplace_back(precisions);
            f1score_results.emplace_back(f1scores);
            accuracy_results.emplace_back(accuracy);

            /*
            std::cout << "-------------FINAL RESULTS-----------\n";
            for (const auto &elem: sensitivity_results) std::cout << elem << " ";
            std::cout << "\n";

            for (const auto &elem: specificity_results) std::cout << elem << " ";
            std::cout << "\n";

            for (const auto &elem: precision_results) std::cout << elem << " ";
            std::cout << "\n";

            for (const auto &elem: f1score_results) std::cout << elem << " ";
            std::cout << "\n";

            for (const auto &elem: accuracy_results) std::cout << elem << " ";
            std::cout << "\n\n";
            */
        }

        // Storing results of the given threshold
        all_specificity_results.emplace_back(specificity_results);
        all_sensitivity_results.emplace_back(sensitivity_results);
        all_precision_results.emplace_back(precision_results);
        all_f1score_results.emplace_back(f1score_results);
        all_accuracy_results.emplace_back(accuracy_results);
    }

    // Writing results into csv files .. for analyzing and later visualization
    write_csv(all_specificity_results, thresholds, "specificity");
    write_csv(all_sensitivity_results, thresholds, "sensitivity");
    write_csv(all_precision_results, thresholds, "precision");
    write_csv(all_f1score_results, thresholds, "f1score");
    write_csv(all_accuracy_results, thresholds, "accuracy");

    std::cout << "Finished simulation. Files are ready to analyze!\n";
    return 0;
}


std::vector<double> generate_threshold_values(double from, double to, double stride)
{
    std::vector<double> tmp_results;
    for (double i = from; i < to + stride; i += stride)
    {
        tmp_results.push_back(i);
    }
    return tmp_results;
}

void write_csv(const std::vector<std::vector<double>> &result, const std::vector<double> &thresholds,
               const std::string &metric_name)
{
    // Create file name
    std::string file_name = metric_name;
    file_name.append(".csv");

    // Create file
    std::ofstream my_file;
    my_file.open(file_name);


    // Creating the HEADER part of the csv file
    for (const auto &threshold : thresholds) {
        my_file << threshold << "_" << metric_name << ",";
    }
    my_file << "\n";

    // Filling up all the rows with the generated results
    for (size_t i = 0; i < result[0].size(); i++)
    {
        for (const auto &res: result)
        {
            my_file << res[i] << ",";
        }
        my_file << "\n";
    }

    my_file.close();
}

/*
void writeCsv(const std::vector<double> &sensi, const std::vector<double> &speci,
              const std::vector<double> &preci, const std::vector<double> &f1sc,
              const std::vector<double> &accur) {
    std::ofstream myfile;
    myfile.open("example.csv");

    myfile << "sensitivity,specificity,precision,f1,accuracy\n";
    for (int i = 0; i < sensi.size(); i++) {
        myfile << sensi[i] << "," << speci[i] << "," << preci[i] << ","
               << f1sc[i] << "," << accur[i] << "\n";
    }

    myfile.close();
}
*/