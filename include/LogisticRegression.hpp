#pragma once

#include <torch/torch.h>

struct LogisticRegressionImpl : torch::nn::Module {
	torch::Tensor output;
	torch::nn::Linear linear;

	LogisticRegressionImpl(int64_t number_of_input_features)
		: linear(register_module("linear", torch::nn::Linear(number_of_input_features, 1)))
	{
	}

	torch::Tensor forward(const torch::Tensor &input) {
		auto x = linear(input);
		auto a = torch::sigmoid(x);
		return a;
		//return linear(input);
	}
};

TORCH_MODULE(LogisticRegression);