#pragma once

#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "torch/torch.h"

/*
  This class is used for creating a random m_features via different kinds of distributions.
*/
class RandomDatasetGenerator {
private:
    const size_t m_rows;
    std::vector<std::string> m_labels;
    std::vector<double> m_outcome_probabilities;
    std::vector<double> m_weights;
    torch::Tensor m_features;
    torch::Tensor m_target;
    std::mt19937 m_generator;


    /*
      This template helper function is used for generating random values by a 
      given distribution.

      @param: dist -> arbitrary type of distribution
      @param: weight -> tells how likely the column will affect the outcome

      @return: torch::Tensor type column with the generated random values
    */
    template<typename T>
    torch::Tensor generateRandomValuesHelper(T &dist, const bool is_whole_number = true) {
        // Creating X type of distributed random numbers, and storing them in distValues
        std::vector<double> distValues(m_rows);

        if (is_whole_number)
        {
            for (auto &elem : distValues) {
                elem = (int)(dist(m_generator));
            }
        }
        else
        {
            for (auto &elem : distValues) {
                elem = (dist(m_generator));
            }
        }
        // Converting the distValues vector into Tensor and returning it
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);

        return torch::from_blob(distValues.data(), {static_cast<int>(m_rows), 1}, opts);
    };

    /*
      This function lets you concatenate a column to the m_features
      
      @param: col -> torch::Tensor type column
      @return: -
    */
    template<typename T>
    void appendToFeatures(const T &col) {
        // Checking if m_features is empty
        if (m_features.numel() == 0) {
            // If it is, then the "col" parameter will be the first column
            m_features = col.clone().detach();
        } else {
            // If it is NOT, then append "col" to the existing m_features
            m_features = torch::cat({m_features, col}, 1);
        }
    }

    /*
      This function appends column "header" to the data table headers

      @param: base -> colum name
      @return: -
    */
    void appendLabel(std::string &&base);


public:

    /*
      This enum class is used for identifying the different distribution types 
      when creating a column and also when parsing it.
    */
    enum class DistributionTypes {
        Binomial,
        Bernoulli,
        Normal,
        UniformDiscrete,
        UniformReal,
        Gamma
    };

    /*
      This struct is for storing important information about the
      column which would like to append to the dataset
    */
    struct ColumnDataType {
        const DistributionTypes type;
        const std::unordered_map<std::string, double> parameters;
    };

    RandomDatasetGenerator() = delete;

    explicit RandomDatasetGenerator(const size_t &r, const std::vector<RandomDatasetGenerator::ColumnDataType> &vec,
                                    bool binary_target);

    ~RandomDatasetGenerator() = default;

    void testPrint() const;

    torch::Tensor getFeatures();

    torch::Tensor getTarget();

    /*
      Produces random non-negative integer values into a column, distributed 
      according to discrete probability function.

      @param: num_trials -> values between 0 and num_trials
      @param: prob      -> probability of success of each trial
      @param: weight -> tells how likely the column will affect the outcome
      @return: -
    */
    void generateBinomialColumn(const size_t &num_trials, const double &prob, const double &weight = 1.0);

    /*
      Produces random boolean values into a column, 
      according to the discrete probability function.

      @param: prob  -> probability of true
      @param: weight -> tells how likely the column will affect the outcome
      @return: -
    */
    void generateBernoulliColumn(const double &prob, const double &weight = 1.0);

    /*
      Generates random numbers according to the Normal
      (or Gaussian) random number distribution... into a column

      @param: mean    -> distribution mean
      @param: stddev  -> standard deviation
      @param: weight -> tells how likely the column will affect the outcome
      @return: -
    */
    void generateNormalColumn(const double &mean, const double &stddev, const double &weight = 1.0);


    /*
      Produces random integer values into a column, uniformly distributed 
      on the closed interval [a, b], that is, distributed according 
      to the discrete probability function.

      @param: from  -> range FROM 
      @param: to  -> range TO
      @param: weight -> tells how likely the column will affect the outcome
      @return: -
    */
    void generateUniformDiscreteColumn(const int &from, const int &to, const double &weight = 1.0);

    /*
      Produces random floating-point values into a column, 
      uniformly distributed on the interval [a, b), that is, distributed 
      according to the probability density function.

      @param: from  -> range FROM (inclusive)
      @param: to  -> range TO (exclusive)
      @param: weight -> tells how likely the column will affect the outcome
      @return: -
    */
    void generateUniformRealColumn(const double &from, const double &to, const double &weight = 1.0);


    /*
      Produces random positive floating-point values into a column, distributed 
      according to probability density function.

      @param: alpha -> alpha is known as the shape parameter
      @param: beta  -> beta is known as the scale parameter
      @param: weight -> tells how likely the column will affect the outcome
      @return: - 
    */
    void generateGammaColumn(const double &alpha, const double &beta, const double &weight = 1.0);

    /*
      With this function, can create the TARGET column, which contains values between 0 and 1
    */
    void generateBinaryTargetColumn();

    /*
     * Return the calculated probability outcome column for later usage (saving into file)
     */
    std::vector<double> getProbabilityOutput(const std::string &train_or_test = "test", double split_size = 0.6);

    /*
      This function helps parsing the input vector parameter of the RandomDatasetGenerator constructor.
      
      @param: vec -> it contains various types of columns
      @return: -
    */
    void parseInputColumnData(const std::vector<ColumnDataType> &vec);

    /*
      This function lets you write the content of the m_features
      into a CSV file.

      @param: -
      @return: -
    */
    void writeCSV();

    void prettyPrint() const;


    friend std::ostream &operator<<(std::ostream &os, const RandomDatasetGenerator &rd);

};

