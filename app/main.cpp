#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"

int main() {
	auto rd = RandomDataset(10);
	rd.generateBernoulliColumn(0.5, 0.5);
	rd.generateBernoulliColumn(0.5, 0.75);	
	rd.generateBernoulliColumn(0.5, 1);	
	rd.generateBernoulliColumn(0.5, 1.25);	
	rd.generateBernoulliColumn(0.5, 1.5);	
	rd.generateBernoulliColumn(0.5, 1.75);

	rd.generateBinaryTargetColumn();
/*	
	rd.generateBinomialColumn(4, 0.6);
	rd.generateNormalColumn(2.5, 1.35);
	rd.generateUniformDiscreteColumn(1, 6);
	rd.generateUniformRealColumn(1.0, 6.5);
	rd.generateGammaColumn(1.0, 2.0);
*/
	rd.prettyPrint();
	//std::cout << rd << "\n";

	return 0;
}