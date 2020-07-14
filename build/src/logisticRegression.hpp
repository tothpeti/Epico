#pragma once

#include <torch/torch.h>

class LogisticRegression : torch::nn::Module {
public:	
	torch::Tensor output;
	torch::nn::Linear linear;

	LogisticRegression(int64_t inputSize);

	torch::Tensor forward(torch::Tensor input);

};
