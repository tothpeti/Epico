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
	return m_features;
}


const torch::Tensor& RandomDataset::getTarget() const {
	return m_target;
}


torch::data::Example<> RandomDataset::get(size_t index)
{
	return {m_features[index], m_target[index]};
}


torch::optional<size_t> RandomDataset::size() const
{
	return m_features.size(0);
}

std::vector<std::vector<double>> RandomDataset::convert_dataset_to_vector() {

    std::vector<std::vector<double>> res;

    const auto features_accessor = m_features.accessor<double, 2>();
    const auto target_accessor = m_target.accessor<double, 2>();

    for (size_t i = 0; i < features_accessor.size(0); i++)
    {
        std::vector<double> row;
        row.reserve(features_accessor.size(1) + 1);

        for (size_t j = 0; j < features_accessor.size(1); j++)
        {
            row.emplace_back(features_accessor[i][j]);
        }
        row.emplace_back(target_accessor[i][0]);
        res.emplace_back(row);
    }

    return res;
}
