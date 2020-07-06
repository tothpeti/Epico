#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"
#include "ATen/ATen.h"

int main() {
	std::cout << "Am i here?" << "\n";
	auto rd = RandomDataset(7);

	rd.concatenateColumns(
		rd.generateBinomialColumn(15, 0.4),
		rd.generateNormalColumn(2.5, 4.5),
		rd.generateBernoulliColumn(0.6)
	);
	rd.prettyPrint();
	return 0;
}