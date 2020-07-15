#pragma once

#include "torch/torch.h"

class RandomDataset : public torch::data::Dataset<RandomDataset> {
  private:
    torch::Tensor features;
    torch::Tensor target;
  public:    
    
    /*
      This enum class is used for identifying which part of the dataset should we return.
      (kTrain -> Training part, kTest -> Test part)
    */
    enum class Mode {kTrain, kTest};


    explicit RandomDataset(const torch::Tensor &fs, const torch::Tensor &t,
                          RandomDataset::Mode mode, const float splitSize);

    ~RandomDataset() = default;


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
};