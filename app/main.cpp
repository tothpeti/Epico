#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"


int main() {

	RandomDataset::ColumnDataType bern{
		"bernoulli", 											//name
		{{"prob", 0.5}, {"weight", 0.75}} //parameters
	};	
	RandomDataset::ColumnDataType bern2{
		"bernoulli",
		{{"prob", 0.5}, {"weight", 1.25}}
	};
	std::vector<RandomDataset::ColumnDataType> cols;
	cols.push_back(bern);
	cols.push_back(bern2);

	auto rd = RandomDataset(10, cols, true).map(torch::data::transforms::Stack<>());
	
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(rd)
	);

	std::cout << "pepe\n";
	for(auto& batch: *data_loader){
		std::cout << "am i here? \n";
		std::cout << batch.data << "\n";
		std::cout << "or here? \n";
	}
	
	return 0;
}