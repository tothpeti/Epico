#include "RandomDataset.hpp"

RandomDataset::RandomDataset(const torch::Tensor &fs, const torch::Tensor &t, 
    RandomDataset::Mode mode, const float splitSize)
  : features(fs), target(t)
{
	if (mode == RandomDataset::Mode::kTrain) {
		auto cutTill = round(this->features.size(0) * splitSize);

		this->features = this->features.index({ torch::indexing::Slice(0, cutTill) }).clone();
		this->target = this->target.index({ torch::indexing::Slice(0, cutTill) }).clone();

	}
	else 
	{
		auto currDatasetLength = this->features.size(0);
		auto cutFrom = round(this->features.size(0) * (1 - splitSize));

		this->features = this->features.index({ torch::indexing::Slice(cutFrom, currDatasetLength) }).clone();
		this->target = this->target.index({ torch::indexing::Slice(cutFrom, currDatasetLength) }).clone();
	}
}

const torch::Tensor& RandomDataset::getFeatures() const {
	return this->features;
}


const torch::Tensor& RandomDataset::getTarget() const {
	return this->target;
}


torch::data::Example<> RandomDataset::get(size_t index)
{
	return { this->features[index], this->target[index]};
}


torch::optional<size_t> RandomDataset::size() const
{
	return this->features.size(0);
}