#pragma once

#include "torch/torch.h"

class RandomDataset : public torch::data::Dataset<RandomDataset> {
  private:
    torch::Tensor m_features;
    torch::Tensor m_target;
  public:    
    
    /*
      This enum class is used for identifying which part of the dataset should we return.
      (kTrain -> Training part, kTest -> Test part)
    */
    enum class Mode {kTrain, kTest};


    explicit RandomDataset(torch::Tensor features, torch::Tensor target,
                          RandomDataset::Mode mode, float split_size);

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
   Return all m_features stacked into a single tensor
  */
  const torch::Tensor& getFeatures() const;

  /*
    Return all m_target stacked into a single tensor
  */
  const torch::Tensor& getTarget() const;

  std::vector<std::vector<double>> convert_dataset_to_vector();
};