#if !defined(RANDOMDATASETGENERATOR_H_INCLUDED)
#define RANDOMDATASETGENERATOR_H_INCLUDED

#include "torch/torch.h"
#include <ATen/ATen.h>
#include <random>
/*
  This class is used for creating a random dataset via different kinds of distributions.
*/
class RandomDataset {
  private:
    const size_t rows;
    torch::Tensor dataset = torch::Tensor(); 

    /*
      This template helper function is used for generating random values by a 
      given distribution.

      @param: dist -> arbitrary type of distribution
      @return: torch::Tensor type column with the generated random values
    */
    template<typename T>
    torch::Tensor generateRandomValuesHelper(T &dist, std::string &ty) {
    	// Initializing the random generator
	    std::random_device rd; 
	    std::mt19937 gen(rd());
      //std::default_random_engine gen;

	    // Creating X type of distributed random numbers, and storing them in distValues 
	    std::vector<float> distValues(this->rows);
      std::cout << ty << "\n";
	    auto generateElems = [&gen, &dist, i = 0]() mutable { ++i; return dist(gen); };
	    std::generate(begin(distValues), end(distValues), generateElems);

      std::for_each(begin(distValues), end(distValues),
        [](const float &val) {std::cout << val << " ";}
      );
      std::cout << "\n";
	    // Converting the distValues vector into Tensor and returning it	
      // std::data
	    return torch::from_blob(distValues.data(), {(int)distValues.size(), 1});
    };

  public:
    RandomDataset() = delete;
    explicit RandomDataset(size_t r);
    RandomDataset(const RandomDataset &rd);
    RandomDataset(const RandomDataset &&rd);
    ~RandomDataset();

    /*
      Produces random non-negative integer values, distributed 
      according to discrete probability function... into a column.

      @param: numTrials -> values between 0 and numTrials
      @param: prob      -> probability of success of each trial
      @return: torch::Tensor type column
    */
    torch::Tensor generateBinomialColumn(const size_t &numTrials, const float &prob);

    /*
      Produces random boolean values, 
      according to the discrete probability function... into a column.

      @param: prob  -> probability of true
      @return: torch::Tensor type column
    */
    torch::Tensor generateBernoulliColumn(const float &prob);

    /*
      Generates random numbers according to the Normal
      (or Gaussian) random number distribution... into a column

      @param: mean    -> distribution mean
      @param: stddev  -> standard deviation
      @return: torch::Tensor type column
    */
    torch::Tensor generateNormalColumn(const float &mean, const float &stddev);


    /*
      Produces random integer values into a column, uniformly distributed 
      on the closed interval [a, b], that is, distributed according 
      to the discrete probability function.

      @param: a  -> range FROM 
      @param: b  -> range TO
      @return: torch::Tensor type column
    */
    torch::Tensor generateUniformDiscreteColumn(const int &a, const int &b);

    /*
      Produces random floating-point values into a column, uniformly distributed 
      on the interval [a, b), that is, distributed 
      according to the probability density function.

      @param: a  -> range FROM (inclusive)
      @param: b  -> range TO (exclusive)
      @return: torch::Tensor type column
    */
    torch::Tensor generateUniformRealColumn(const float &a, const float &b);


    /*
      Produces random positive floating-point values into a column, distributed 
      according to probability density function.

      @param: alpha -> alpha is known as the shape parameter
      @param: beta  -> beta is known as the scale parameter
      @return: torch::Tensor type column
    */
    torch::Tensor generateGammaColumn(const float &alpha, const float &beta);

    void prettyPrint() const;

    /*
      This function lets you concatenate columns in place.
      @param: first -> torch::Tensor type container
      @param: args  -> unknown numbers of torch::Tensor type containers

      Example:
        RandomDataset rd = RandomDataset(rows=7);
    	  rd.concatenateColumns(
		      rd.generateBinomialColumn(15, 0.4),
		      rd.generateNormalColumn(2.5, 4.5),
		      rd.generateBernoulliColumn(0.6)
	      );
    */
    template<typename T, typename... Args>
    void concatenateColumns(T &&first, Args&&... args){
	    // Checking if the dataset is empty 
	    if (this->dataset.numel() == 0)
	    {
		    // If it is, then the resultTensor will be the first column of the dataset
		    this->dataset = first.detach().clone();
		    this->dataset = torch::cat({this->dataset, args...}, 1);
	    }
	    else
	    {
		    // Else, append the newly generated column to the dataset
		    this->dataset = torch::cat({this->dataset, first, args...}, 1);
	    }
    };
};


#endif // RANDOMDATASETGENERATOR_H_INCLUDED