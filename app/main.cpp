#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"

int main() {

	for(size_t i = 0; i<10; i++) {

		auto rd = RandomDataset(10);
		rd.generateBernoulliColumn(0.5, 0.5);
		rd.generateBernoulliColumn(0.5, 0.75);	
		rd.generateBernoulliColumn(0.5, 1);	
		rd.generateBernoulliColumn(0.5, 1.25);	
		rd.generateBernoulliColumn(0.5, 1.5);	
		rd.generateBernoulliColumn(0.5, 1.75);
		rd.generateBinaryTargetColumn();

		rd.prettyPrint();
	}

	return 0;
}