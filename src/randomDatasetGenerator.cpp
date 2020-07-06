#include <iostream>
#include <random>
#include <iomanip>
#include <cstdio>
#include "randomDatasetGenerator.hpp"

#include "torch/torch.h"
#include "ATen/ATen.h"


RandomDataset::RandomDataset(size_t r)
	: rows(r) 
{
}

RandomDataset::RandomDataset(const RandomDataset &rd)
	: RandomDataset{rd.rows}
{
}

RandomDataset::RandomDataset(const RandomDataset &&rd)
	: rows(rd.rows)
{
}
	
RandomDataset::~RandomDataset(){
}

torch::Tensor RandomDataset::generateBinomialColumn(const size_t &numTrials, const float &prob){
	std::binomial_distribution<> d(numTrials, prob);
	return RandomDataset::generateRandomValuesHelper(d, std::string("bin"));
}

torch::Tensor RandomDataset::generateBernoulliColumn(const float &prob) {
	std::bernoulli_distribution d(prob);
	return RandomDataset::generateRandomValuesHelper(d, std::string("bern"));
}

torch::Tensor RandomDataset::generateNormalColumn(const float &mean, const float &stddev){
	std::normal_distribution<float> d(mean, stddev);
	auto tens = RandomDataset::generateRandomValuesHelper(d, std::string("norm"));
	std::cout << "in cpp: norm \n";
	std::cout << tens << "\n";
	return tens;
}

torch::Tensor RandomDataset::generateUniformDiscreteColumn(const int &a, const int &b) {
	std::uniform_int_distribution<> d(a, b);
	return RandomDataset::generateRandomValuesHelper(d, std::string("disc-uni"));
}

torch::Tensor RandomDataset::generateUniformRealColumn(const float &a, const float &b) {
	std::uniform_real_distribution<float> d(a, b);
	return RandomDataset::generateRandomValuesHelper(d, std::string("real-uni"));
}

torch::Tensor RandomDataset::generateGammaColumn(const float &alpha, const float &beta) {
	std::gamma_distribution<float> d(alpha, beta);
	auto tens = RandomDataset::generateRandomValuesHelper(d, std::string("gamma"));
	//std::cout << "in cpp: gamma \n";
	//std::cout << tens << "\n";
	return tens;
}

void RandomDataset::prettyPrint() const {
	auto dataset_accessor = this->dataset.accessor<float, 2>();

	for(size_t i = 0; i < dataset_accessor.size(0); i++) {
		for(size_t j = 0; j < dataset_accessor.size(1); j++) {
			//std::cout << std::fixed << std::setprecision(5) << dataset_accessor[i][j] << "  |  ";
			std::printf("%.4f  ", dataset_accessor[i][j]);
		}
		std:: cout << "\n";
	}
	
}
