#if !defined(RANDOMDATASETGENERATOR_H_INCLUDED)
#define RANDOMDATASETGENERATOR_H_INCLUDED

#include "torch/torch.h"
#include <ATen/ATen.h>

/*
  This class is used for creating a random dataset via different distributions.
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
    */
    void generateBinomialColumn(const size_t &numTrials, const float &prob);

    /*
      This function generates a Bernoulli distributed column
      @param: prob  -> probability of true
    */
    void generateBernoulliColumn(const float &prob);
    void prettyPrint() const;
};



#endif // RANDOMDATASETGENERATOR_H_INCLUDED