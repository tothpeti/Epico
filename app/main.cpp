#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"

int main() {
	torch::Tensor t = torch::rand({2, 3});
	generator::sayHello(t);
	return 0;
}