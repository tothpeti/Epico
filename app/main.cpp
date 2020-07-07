#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"

int main() {
	auto rd = RandomDataset(7);

	rd.concatenateColumns(
		rd.generateBinomialColumn(4, 0.6),
		rd.generateBernoulliColumn(0.35),
		rd.generateNormalColumn(2.5, 1.35),
		rd.generateUniformDiscreteColumn(1, 6),
		rd.generateUniformRealColumn(1.0, 6.5),
		rd.generateGammaColumn(1.0, 2.0)
	);

	std::cout << rd << "\n";

	return 0;
}