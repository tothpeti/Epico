#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"
#include "ATen/ATen.h"

int main() {
	std::cout << "Am i here?" << "\n";
	auto rd = RandomDataset(7);
	std::cout << "or here?" << "\n";
	rd.generateBinomialColumn(15, 0.4);
	std::cout << "here?" << "\n";
	rd.generateBinomialColumn(5, 0.7);
	std::cout << "or?" << "\n";
	rd.generateBernoulliColumn(0.6);
	std::cout << "hoh?" << "\n";
	rd.generateNormalColumn(3.0, 4.0);
	rd.prettyPrint();
	return 0;
}