#if !defined(RANDOMDATASETGENERATOR_H_INCLUDED)
#define RANDOMDATASETGENERATOR_H_INCLUDED

#include "torch/torch.h"
#include <ATen/ATen.h>

/*
  This class is used for creating a random dataset via different kinds of distributions.
*/
class RandomDataset {
  private:
    const size_t rows;
    torch::Tensor dataset = torch::Tensor(); 
  public:
    RandomDataset() = delete;
    explicit RandomDataset(size_t r);
    RandomDataset(const RandomDataset &rd);
    RandomDataset(const RandomDataset &&rd);
    ~RandomDataset();

    /*
      This function generates a Binomial distributed column.
      @param: numTrials -> values between 0 and numTrials
      @param: prob      -> probability of success of each trial
      @return: torch::Tensor type container
    */
    torch::Tensor generateBinomialColumn(const size_t &numTrials, const float &prob);

    /*
      This function generates a Bernoulli distributed column
      @param: prob  -> probability of true
      @return: torch::Tensor type container
    */
    torch::Tensor generateBernoulliColumn(const float &prob);

    /*
      This function generates a Normal distributed column
      @param: mean    -> distribution mean
      @param: stddev  -> standard deviation
      @return: torch::Tensor type container
    */
    torch::Tensor generateNormalColumn(const float &mean, const float &stddev);

    void prettyPrint() const;

    /*
      This function lets you concatenate columns in place.
      @param: first -> torch::Tensor type container
      @param: args  -> unknown numbers of torch::Tenor type containers
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
		    this->dataset = torch::cat({this->dataset, std::forward<Args>(args)...}, 1);
	    }
	    else
	    {
		    // Else, append the newly generated column to the dataset
		    this->dataset = torch::cat({this->dataset, first, std::forward<Args>(args)...}, 1);
	    }
    };

};



#endif // RANDOMDATASETGENERATOR_H_INCLUDED