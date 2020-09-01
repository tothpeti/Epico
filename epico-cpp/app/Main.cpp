#include "RandomDatasetGenerator.hpp"
#include "FileWriter/CsvFileWriter.hpp"
#include "RandomDataset.hpp"
#include "LogisticRegression.hpp"

/*
 * Helper functions
 */

std::vector<double> generate_threshold_values(double from, double to, double stride);

void print_stored_metrics(const std::vector<double> &sens, const std::vector<double> &spec,
                          const std::vector<double> &prec, const std::vector<double> &f1score,
                          const std::vector<double> &acc);

int main() {
    const size_t number_of_rows = 1000;
    const size_t number_of_simulations = 50;

    CsvFileWriter writer;

    for (size_t i = 0; i < number_of_simulations; i++) {

        //Creating columns for RandomDataset
        RandomDatasetGenerator::ColumnDataType bern{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(0.25)}}        //parameters
        };
        RandomDatasetGenerator::ColumnDataType bern2{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(0.5)}}
        };
        RandomDatasetGenerator::ColumnDataType bern3{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(0.75)}}
        };
        RandomDatasetGenerator::ColumnDataType bern4{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(1)}}
        };
        RandomDatasetGenerator::ColumnDataType bern5{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(1.25)}}
        };

        RandomDatasetGenerator::ColumnDataType bern6{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(1.5)}}
        };
        RandomDatasetGenerator::ColumnDataType bern7{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(1.75)}}
        };

        RandomDatasetGenerator::ColumnDataType bern8{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(2)}}
        };

        RandomDatasetGenerator::ColumnDataType bern9{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(2.25)}}
        };

        RandomDatasetGenerator::ColumnDataType bern10{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,
                {{"prob", 0.5}, {"weight", std::log(2.5)}}
        };

        std::vector<RandomDatasetGenerator::ColumnDataType> cols{
                bern, bern2, bern3, bern4, bern5, bern6, bern7, bern8, bern9, bern10
        };


        // Create RandomDataLoader from previously created Columns
        auto rdGenerator = std::make_unique<RandomDatasetGenerator>(number_of_rows, cols, true);
        writer.saveWholeDataset(i+1, rdGenerator);
    }
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


void print_stored_metrics(const std::vector<double> &sens, const std::vector<double> &spec,
                          const std::vector<double> &prec, const std::vector<double> &f1score,
                          const std::vector<double> &acc)
{

    std::cout << "-------------FINAL RESULTS-----------\n";
    for (const auto &elem: sens) std::cout << elem << " ";
    std::cout << "\n";

    for (const auto &elem: spec) std::cout << elem << " ";
    std::cout << "\n";

    for (const auto &elem: prec) std::cout << elem << " ";
    std::cout << "\n";

    for (const auto &elem: f1score) std::cout << elem << " ";
    std::cout << "\n";

    for (const auto &elem: acc) std::cout << elem << " ";
    std::cout << "\n\n";
}

/*

int main() {
    //Parameters
    const int number_of_rows        = 1000;  // number of rows which the RandomDataset will have
    const int number_of_features    = 10;    // how many columns will be created manually, used for Model
    const int batch_size            = 64;
    const size_t number_of_epochs   = 500;     // how many rounds used for training the model
    const size_t simulation_rounds  = 50;
    const double learning_rate      = 0.01; // used for Model

    // Generating threshold values
    auto thresholds = generate_threshold_values(0.5, 0.95, 0.05);

    std::vector<std::vector<double>> all_sensitivity_results;
    std::vector<std::vector<double>> all_specificity_results;
    std::vector<std::vector<double>> all_precision_results;
    std::vector<std::vector<double>> all_f1score_results;
    std::vector<std::vector<double>> all_accuracy_results;

    all_sensitivity_results.reserve(simulation_rounds);
    all_specificity_results.reserve(simulation_rounds);
    all_precision_results.reserve(simulation_rounds);
    all_f1score_results.reserve(simulation_rounds);
    all_accuracy_results.reserve(simulation_rounds);

    //SIMULATION
    for (size_t i = 0; i < simulation_rounds; i++) {

    std::vector<double> sensitivity_results;
    std::vector<double> specificity_results;
    std::vector<double> precision_results;
    std::vector<double> f1score_results;
    std::vector<double> accuracy_results;

    // Reserve memories
    sensitivity_results.reserve(thresholds.size());
    specificity_results.reserve(thresholds.size());
    precision_results.reserve(thresholds.size());
    f1score_results.reserve(thresholds.size());
    accuracy_results.reserve(thresholds.size());

    //Creating columns for RandomDataset
    RandomDatasetGenerator::ColumnDataType bern{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
            {{"prob", 0.5}, {"weight", std::log(0.25)}}        //parameters
    };
    RandomDatasetGenerator::ColumnDataType bern2{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(0.5)}}
    };
    RandomDatasetGenerator::ColumnDataType bern3{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(0.75)}}
    };
    RandomDatasetGenerator::ColumnDataType bern4{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(1)}}
    };
    RandomDatasetGenerator::ColumnDataType bern5{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(1.25)}}
    };

    RandomDatasetGenerator::ColumnDataType bern6{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(1.5)}}
    };
    RandomDatasetGenerator::ColumnDataType bern7{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(1.75)}}
    };

    RandomDatasetGenerator::ColumnDataType bern8{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(2)}}
    };

    RandomDatasetGenerator::ColumnDataType bern9{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(2.25)}}
    };

    RandomDatasetGenerator::ColumnDataType bern10{
            RandomDatasetGenerator::DistributionTypes::Bernoulli,
            {{"prob", 0.5}, {"weight", std::log(2.5)}}
    };

    std::vector<RandomDatasetGenerator::ColumnDataType> cols{
            bern, bern2, bern3, bern4, bern5, bern6, bern7, bern8, bern9, bern10
    };


    //Creating Training and Test Data Loaders

    // Create RandomDataLoader from previously created Columns
    auto rdGenerator = std::make_unique<RandomDatasetGenerator>(number_of_rows, cols, true);

    // Create training RandomDataset from the RandomDatasetGenerator
    auto rdTrain = std::make_unique<RandomDataset>(rdGenerator->getFeatures(), rdGenerator->getTarget(),
                                                   RandomDataset::Mode::kTrain, 0.7);
    //auto numberOfTrainSamples = rdTrain->size().value();

    auto trainingDataLoader = torch::data::make_data_loader(
            std::move(rdTrain->map(torch::data::transforms::Stack<>())),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
    );



    // Create test RandomDataset from the RandomDatasetGenerator
    auto rdTest = std::make_unique<RandomDataset>(rdGenerator->getFeatures(), rdGenerator->getTarget(),
                                                  RandomDataset::Mode::kTest, 0.3);
    auto numberOfTestSamples = rdTest->size().value();

    // SAVING DATA FOR LATER USAGE WHEN WANT TO WRITE IT TO FILE
    auto test_vec = rdTest->convert_dataset_to_vector();
    auto test_prob_vec = rdGenerator->getProbabilityOutput("test", 0.3);

    int counter = 0;
    for(const auto &elem : test_prob_vec)
    {
        if (elem >= 0.5)
        {
            counter++;
        }
    }
    std::cout << "Number of 1s in test dataset: "<< counter << "\n\n";

    auto testingDataLoader = torch::data::make_data_loader(
            std::move(rdTest->map(torch::data::transforms::Stack<>())),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
    );


    // Logistic regression model
    //LogisticRegression model(number_of_features);

    // Loss and optimizer
    //torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    //torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

    // Set floating point output precision - USE IF WANT TO PRINT RESULTS!!
    //std::cout << std::fixed << std::setprecision(4);

    //Training the model
    model->train();
    for (size_t epoch = 0; epoch < number_of_epochs; epoch++) {

        // Initializing running metrics
        double runningLoss = 0.0;

        for (auto &batch: *trainingDataLoader) {
            // Clear the previous gradients
            optimizer.zero_grad();

            auto data = batch.data.to(torch::kFloat32);
            auto target = batch.target.to(torch::kFloat32);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss

            auto loss = torch::binary_cross_entropy(output, target);
            //auto loss = criterion(output, target);

            // Update running loss
            runningLoss += loss.item<double>();
            //std::cout << loss.item<double>() << "\n" << "-------- \n";

            // Compute gradients
            loss.backward();

            // Adjust weights
            optimizer.step();
        }
        //if (epoch % 50 == 0)
        //    std::cout << epoch + 1 << " - loss: "<< runningLoss << "\n";
        //auto sampleMeanLoss = runningLoss / numberOfTrainSamples;
    }

    //Testing the trained model
    torch::NoGradGuard no_grad;
    model->eval();

    bool store_once = true;
    std::vector<double> preds;
    preds.reserve(numberOfTestSamples);

    for (const auto &threshold : thresholds) {

        double running_loss = 0.0;
        double numberOfCorrect = 0.0;
        double sensitivity = 0.0;
        double specificity = 0.0;
        double precision = 0.0;
        double f1score = 0.0;
        double accuracy = 0.0;
        double tp = 0.0;
        double tn = 0.0;
        double fp = 0.0;
        double fn = 0.0;
        for (const auto &batch : *testingDataLoader) {
            auto data = batch.data.to(torch::kFloat32);
            auto target = batch.target.to(torch::kFloat32);

            // Forward pass
            auto output = model->forward(data);

            // Convert and store output into vector
            if (store_once)
            {
                const auto output_accessor = output.accessor<float, 2>();
                for (size_t col_idx = 0; col_idx < output_accessor.size(0); col_idx++)
                {
                    for (size_t row_idx = 0; row_idx < output_accessor.size(1); row_idx++)
                        preds.emplace_back(output_accessor[col_idx][row_idx]);
                }
            }

            // Calculate loss
            //auto loss = torch::nn::functional::binary_cross_entropy(output, target);
            //running_loss += loss.item<double>();

            // Rounding output's values via custom_threshold into 0 or 1
            auto custom_threshold = torch::tensor({threshold});
            auto rounded_output = torch::where(output >= custom_threshold, torch::tensor({1}),
                                               torch::tensor({0}));


            // Calculate true positive, true negative, false positive, false positive (per batch)
            // rounded_output = predicted
            // target = true labels
            auto tp_batch = torch::logical_and(target == 1, rounded_output == 1).sum().to(torch::kFloat64);
            auto tn_batch = torch::logical_and(target == 0, rounded_output == 0).sum().to(torch::kFloat64);
            auto fp_batch = torch::logical_and(target == 0, rounded_output == 1).sum().to(torch::kFloat64);
            auto fn_batch = torch::logical_and(target == 1, rounded_output == 0).sum().to(torch::kFloat64);

            tp += tp_batch.item<double>();
            tn += tn_batch.item<double>();
            fp += fp_batch.item<double>();
            fn += fn_batch.item<double>();

            // For calculating Accuracy
            numberOfCorrect += rounded_output.view({-1, 1}).eq(target).sum().item<int>();
        }

        // Used for storing predictions only once!!
        store_once = false;


        //Calculating other metrics
        if (tp > 0) {
            precision = (tp / (tp + fp));
            sensitivity = (tp / (tp + fn));
        } else if (fp > 0) {
            precision = (tp / (tp + fp));
        } else if (fn > 0) {
            sensitivity = (tp / (tp + fn));
        }


        if (precision > 0 || sensitivity > 0)
            f1score = (2 * (precision * sensitivity) / (precision + sensitivity));

        if (tn > 0 || fp > 0)
            specificity = (tn / (tn + fp));

        accuracy = (100 * numberOfCorrect) / numberOfTestSamples;


        specificity_results.emplace_back(specificity);
        sensitivity_results.emplace_back(sensitivity);
        precision_results.emplace_back(precision);
        f1score_results.emplace_back(f1score);
        accuracy_results.emplace_back(accuracy);
    }
    // Saving test dataset into a csv file
    //std::thread t1(write_dataset_and_prediction_into_csv, i+1 , test_vec, test_prob_vec, preds);
    //t1.join();


    // Storing results of the given threshold
    all_specificity_results.emplace_back(specificity_results);
    all_sensitivity_results.emplace_back(sensitivity_results);
    all_precision_results.emplace_back(precision_results);
    all_f1score_results.emplace_back(f1score_results);
    all_accuracy_results.emplace_back(accuracy_results);
    }

    // Writing results into csv files .. for analyzing and later visualization

    std::thread threads[5];
    threads[0] = std::thread(write_result_metrics_into_csv, all_specificity_results, thresholds, "specificity");
    threads[1] = std::thread(write_result_metrics_into_csv, all_sensitivity_results, thresholds, "sensitivity");
    threads[2] = std::thread(write_result_metrics_into_csv, all_precision_results, thresholds, "precision");
    threads[3] = std::thread(write_result_metrics_into_csv, all_f1score_results, thresholds, "f1score");
    threads[4] = std::thread(write_result_metrics_into_csv, all_accuracy_results, thresholds, "accuracy");

    for (auto &thread : threads)
    {
        thread.join();
    }
    std::cout << "Finished simulation. Files are ready to analyze!\n";
    return 0;
    }
*/