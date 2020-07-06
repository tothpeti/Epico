#include <iostream>
#include <random>
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
	// Initializing the random generator
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::binomial_distribution<> d(numTrials, prob);

	// Creating binomial distributed random numbers, and storing them in binom_values
	std::vector<float> binomValues(this->rows);

	auto generateElems = [&gen, &d, i = 0]() mutable { ++i; return d(gen); };
	std::generate(begin(binomValues), end(binomValues), generateElems);


	// Converting the binom_values vector into Tensor and returning it
	return torch::from_blob(std::data(binomValues), {(int)binomValues.size(), 1});
}

torch::Tensor RandomDataset::generateBernoulliColumn(const float &prob) {
	// Initializing the random generator
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::bernoulli_distribution d(prob);

	// Creating bernoulli distributed random numbers, and storing them in bern_values
	std::vector<float> bernValues(this->rows);

	auto generateElems = [&gen, &d, i = 0]() mutable { ++i; return d(gen); };
	std::generate(begin(bernValues), end(bernValues), generateElems);

	// Converting the binom_values vector into Tensor and returning it
	return torch::from_blob(std::data(bernValues), {(int)bernValues.size(), 1});
}

torch::Tensor RandomDataset::generateNormalColumn(const float &mean, const float &stddev){
	// Initializing the random generator
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::normal_distribution<float> d(mean, stddev);

	// Creating normal distributed random numbers, and storing them in normal_values
	std::vector<float> normalValues(this->rows);

	auto generateElems = [&gen, &d, i = 0]() mutable { ++i; return d(gen); };
	std::generate(begin(normalValues), end(normalValues), generateElems);

	// Converting the binom_values vector into Tensor and returning it	
	return torch::from_blob(std::data(normalValues), {(int)normalValues.size(), 1});
}

torch::Tensor RandomDataset::generateUniformDiscreteColumn(const int &a, const int &b) {
	// Initializing the random generator
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> d(a, b);

	// Creating uniformly distributed random numbers, and storing them in uniformIntValues 
	std::vector<float> uniformDiscreteValues(this->rows);

	auto generateElems = [&gen, &d, i = 0]() mutable { ++i; return d(gen); };
	std::generate(begin(uniformDiscreteValues), end(uniformDiscreteValues), generateElems);

	// Converting the binom_values vector into Tensor and returning it	
	return torch::from_blob(std::data(uniformDiscreteValues), {(int)uniformDiscreteValues.size(), 1});
}

torch::Tensor RandomDataset::generateUniformRealColumn(const int &a, const int &b) {
	// Initializing the random generator
	std::random_device rd; 
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> d(a, b);

	// Creating uniformly distributed random numbers, and storing them in uniformRealValues 
	std::vector<float> uniformRealValues(this->rows);

	auto generateElems = [&gen, &d, i = 0]() mutable { ++i; return d(gen); };
	std::generate(begin(uniformRealValues), end(uniformRealValues), generateElems);

	// Converting the binom_values vector into Tensor and returning it	
	return torch::from_blob(std::data(uniformRealValues), {(int)uniformRealValues.size(), 1});
}

void RandomDataset::prettyPrint() const {
	std::cout << this->dataset << "\n";
}
