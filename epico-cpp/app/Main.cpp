#include "RandomDatasetGenerator.hpp"
#include "FileWriter/CsvFileWriter.hpp"
#include "RandomDataset.hpp"

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
        RandomDatasetGenerator::ColumnDataType bern1{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(0.25)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern2{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(0.5)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern3{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(0.75)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern4{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(1.0)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern5{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(2.0)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern6{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(3.0)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern7{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(4.0)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern8{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(5.0)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern9{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(6.0)}}        //parameters
        };

        RandomDatasetGenerator::ColumnDataType bern10{
                RandomDatasetGenerator::DistributionTypes::Bernoulli,                //type
                {{"prob", 0.5}, {"weight", std::log(7.0)}}        //parameters
        };
        std::vector<RandomDatasetGenerator::ColumnDataType> cols{
               bern1, bern2, bern3, bern4, bern5, bern6, bern7, bern8, bern9, bern10
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
