#if !defined(RANDOMDATASETGENERATOR_H_INCLUDED)
#define RANDOMDATASETGENERATOR_H_INCLUDED

#include "torch/torch.h"
#include <random>
#include <vector>
#include <string>

/*
  This class is used for creating a random features via different kinds of distributions.
*/
class RandomDataset : public torch::data::Dataset<RandomDataset>{
  private:
    const size_t rows;
    std::vector<std::string> labels;
    torch::Tensor features; 
    torch::Tensor target;
    std::mt19937 generator;


    /*
      This template helper function is used for generating random values by a 
      given distribution.

      @param: dist -> arbitrary type of distribution
      @param: weight -> tells how likely the column will affect the outcome

      @return: torch::Tensor type column with the generated random values
    */
    template<typename T>
    torch::Tensor generateRandomValuesHelper(T &dist, const double &weight=1.0) {
    	// Initializing the random generator
	    //std::random_device rd; 
	    //std::mt19937 gen(rd());

	    // Creating X type of distributed random numbers, and storing them in distValues 
	    std::vector<double> distValues(this->rows);

	    auto generateElems = [this, &dist, &weight, i = 0]() mutable { ++i; return (dist(this->generator) * weight); };
	    std::generate(begin(distValues), end(distValues), generateElems);

	    // Converting the distValues vector into Tensor and returning it	
      auto opts = torch::TensorOptions().dtype(torch::kFloat64);
      
      return torch::from_blob(distValues.data(), {static_cast<int>(this->rows), 1}, opts);
    };

    /*
      This function lets you concatenate a column to the features
      
      @param: col -> torch::Tensor type column
      @return: -
    */
    template<typename T>
    void appendToFeatures(const T &col) {
      // Checking if features is empty
      if(this->features.numel() == 0) 
      {
        // If it is, then the "col" parameter will be the first column 
        this->features = col.clone().detach();
      }
      else 
      {
        // If it is NOT, then append "col" to the existing features
        this->features = torch::cat({this->features, col}, 1);
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
      This enum used for identifying the different distribution types 
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
      This struct is for storing important informations about the 
      column which would like to append to the dataset
    */
    struct ColumnDataType {
      const DistributionTypes type;
      const std::unordered_map<std::string, double> parameters;
    };
    
    RandomDataset() = delete;    
    explicit RandomDataset(const size_t& r, const std::vector<RandomDataset::ColumnDataType> &vec, 
                           bool binaryTarget = true);
    ~RandomDataset() = default;
   
    void testPrint() const;

    /*
        Returns the 'Example' at the given 'index'.
    */
    torch::data::Example<> get(size_t index) override;

    /*
        Returns the size of the dataset.
    */
    torch::optional<size_t> size() const override;

    /*
      Return all features stacked into a single tensor
    */
    const torch::Tensor& getFeatures() const;

    /*
      Return all target stacked into a single tensor
    */
    const torch::Tensor& getTarget() const;

    /*
      This struct is used for storing training and test datasets and 
      then returning via trainTestSplit function.
    */
    struct TrainTestDataset{
      torch::Tensor X_train;
      torch::Tensor y_train;
      torch::Tensor X_test;
      torch::Tensor y_test;
    };

    /*
      This function produces the training and test datasets which will be used for 
      training the ML model.

      @param: trainSplit -> indicates how many % of the features will be used as TRAINING featurek
      @param: testSplit -> indicates how many % of the features will be used as TEST featurek
      @return: X_train, y_train, X_test, y_test bundled in TrainTestDataset struct
    */
    TrainTestDataset trainTestSplit(const double &trainSplit = 0.6, const double &testSplit = 0.4);


    /*
      Produces random non-negative integer values into a column, distributed 
      according to discrete probability function.

      @param: numTrials -> values between 0 and numTrials
      @param: prob      -> probability of success of each trial
      @param: weight -> tells how likely the column will affect the outcome
      @return: -
    */
    void generateBinomialColumn(const size_t &numTrials, const double &prob, const double &weight = 1.0);

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
      This function helps parsing the input vector parameter of the RandomDataset constructor.
      
      @param: vec -> it contains various types of columns
      @return: -
    */
    void parseInputColumnData(const std::vector<ColumnDataType> &vec);

    /*
      This function lets you write the content of the features
      into a CSV file.

      @param: -
      @return: -
    */
    void writeCSV();
    
    void prettyPrint() const;


    friend std::ostream& operator<<(std::ostream& os, const RandomDataset &rd);

};


#endif // RANDOMDATASETGENERATOR_H_INCLUDED