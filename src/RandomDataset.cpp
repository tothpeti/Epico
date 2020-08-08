#include "RandomDataset.hpp"

#include <utility>

RandomDataset::RandomDataset(torch::Tensor features, torch::Tensor target,
    RandomDataset::Mode mode, float split_size)
  : m_features(std::move(features)), m_target(std::move(target))
{
	if (mode == RandomDataset::Mode::kTrain) {
		auto cut_till = std::round(m_features.size(0) * split_size);

		// Get all the values from 0 to the cut_till row number
		m_features = m_features.index({torch::indexing::Slice(0, cut_till) }).clone();
		m_target = m_target.index({torch::indexing::Slice(0, cut_till) }).clone();

	}
	else 
	{
		auto current_dataset_length = m_features.size(0);
		auto cut_from = std::round(m_features.size(0) * (1 - split_size));

		// Get all the values from the cut_from row number
		m_features = m_features.index({torch::indexing::Slice(cut_from, current_dataset_length) }).clone();
		m_target = m_target.index({torch::indexing::Slice(cut_from, current_dataset_length) }).clone();
	}
}

const torch::Tensor& RandomDataset::getFeatures() const {
	return this->m_features;
}


const torch::Tensor& RandomDataset::getTarget() const {
	return this->m_target;
}


torch::data::Example<> RandomDataset::get(size_t index)
{
	return {m_features[index], m_target[index]};
}


torch::optional<size_t> RandomDataset::size() const
{
	return m_features.size(0);
}