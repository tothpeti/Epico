#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"
#include "ATen/ATen.h"

int main() {
	std::cout << "Am i here?" << "\n";
	auto rd = RandomDataset(7);

	rd.concatenateColumns(
		rd.generateUniformDiscreteColumn(1, 6),
		rd.generateUniformRealColumn(1.0, 6.5)
	);
	rd.prettyPrint();
	return 0;
}