#include <iostream>
#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"

void generator::sayHello(const torch::Tensor &t) {
	std::cout << t << std::endl;
}