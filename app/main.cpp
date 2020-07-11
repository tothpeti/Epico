#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"

class TestData : public torch::data::datasets::Dataset<TestData>{
	private:
		int row;
		int col;
		torch::Tensor res;
		torch::Tensor tar;
	public:
		explicit TestData(int row, int col): 
			row(row), col(col) {
				res = torch::ones({row,col}).clone().detach();
				tar = torch::ones({row,1});
			}

	torch::data::Example<> get(size_t index) override
	{
		return { this->res[index], this->tar[index]};
	}

	torch::optional<size_t> size() const override
	{
		return this->res.size(0);
	}
		

};

int main() {
	
	auto rad = RandomDataset(10);
	rad.generateBernoulliColumn(0.5, 0.5);
	rad.generateBernoulliColumn(0.5, 0.75);	
	rad.generateBinaryTargetColumn();	
	auto rd = RandomDataset(rad).map(torch::data::transforms::Stack<>());
	//auto rd = TestData(2,3).map(torch::data::transforms::Stack<>());
	//rd.prettyPrint();
	//rd.testPrint();

	std::cout << rd.size().value()<<"\n";
	std::cout << "pepeD\n";
	
	//rd.prettyPrint();	
	auto data_loader = torch::data::make_data_loader(
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