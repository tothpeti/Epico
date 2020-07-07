#include <iostream>
#include <iomanip>
#include <fstream>

#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"


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
	return RandomDataset::generateRandomValuesHelper(d);
}

torch::Tensor RandomDataset::generateBernoulliColumn(const float &prob) {
	std::bernoulli_distribution d(prob);
	return RandomDataset::generateRandomValuesHelper(d);
}

torch::Tensor RandomDataset::generateNormalColumn(const float &mean, const float &stddev){
	std::normal_distribution<double> d(mean, stddev);
	return RandomDataset::generateRandomValuesHelper(d);
}

torch::Tensor RandomDataset::generateUniformDiscreteColumn(const int &a, const int &b) {
	std::uniform_int_distribution<> d(a, b);
	return RandomDataset::generateRandomValuesHelper(d);
}

torch::Tensor RandomDataset::generateUniformRealColumn(const float &a, const float &b) {
	std::uniform_real_distribution<double> d(a, b);
	return RandomDataset::generateRandomValuesHelper(d);
}

torch::Tensor RandomDataset::generateGammaColumn(const float &alpha, const float &beta) {
	std::gamma_distribution<double> d(alpha, beta);
	return RandomDataset::generateRandomValuesHelper(d);
	//appendToDataset(tens);
}

void RandomDataset::prettyPrint() const {
	std::cout << std::fixed << std::setprecision(4);
	std::cout << this->dataset << std::endl;
}

std::ostream& operator<<(std::ostream &os, const RandomDataset &rd) {
	os.precision(3);
	os.fixed;
	os << rd.dataset;
	return os;
}


void RandomDataset::writeCSV() {
	std::ofstream myfile;
	myfile.open("example.csv");
	
	auto dataset_accessor = this->dataset.accessor<float, 2>();

	for(int i = 0; i < dataset_accessor.size(0); i++) {
		for(int j = 0; j < dataset_accessor.size(1); j++) {
			myfile << dataset_accessor[i][j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();
}
