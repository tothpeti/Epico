#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"


RandomDataset::RandomDataset(size_t r)
	: rows(r), generator((std::random_device())())  // it's like generator(rd()) syntax but it's inplace
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

void RandomDataset::generateBinomialColumn(const size_t &numTrials, const double &prob, const double &weight){
	std::binomial_distribution<> d(numTrials, prob);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToDataset(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}

void  RandomDataset::generateBernoulliColumn(const double &prob, const double &weight) {
	std::bernoulli_distribution d(prob);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToDataset(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}

void RandomDataset::generateNormalColumn(const double &mean, const double &stddev, const double &weight){
	std::normal_distribution<double> d(mean, stddev);
	
	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToDataset(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}

void RandomDataset::generateUniformDiscreteColumn(const int &a, const int &b, const double &weight) {
	std::uniform_int_distribution<> d(a, b);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToDataset(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}

void RandomDataset::generateUniformRealColumn(const double &a, const double &b, const double &weight) {
	std::uniform_real_distribution<double> d(a, b);
	
	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToDataset(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}

void RandomDataset::generateGammaColumn(const double &alpha, const double &beta, const double &weight) {
	std::gamma_distribution<double> d(alpha, beta);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToDataset(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}

void RandomDataset::generateBinaryTargetColumn() {
	auto inverseLogit = [](const double &p){
		return (std::exp(p) / (1 + std::exp(p)));
	};

	std::vector<double> probOutcome;
	probOutcome.reserve(this->rows);

	// Get iterators for the dataset 
	auto dataset_accessor = this->dataset.accessor<double, 2>();

	// Calculating the row-by-row outcome's probability with inverseLogit
	for(int i = 0; i < dataset_accessor.size(0); i++) {
		double probSum = 0.0;
		for(int j = 0; j < dataset_accessor.size(1); j++) {
			probSum = probSum + dataset_accessor[i][j];
		}
		probOutcome.emplace_back( inverseLogit(probSum) );
	}

	//std::random_device rd;
	//std::mt19937 gen(rd());

	// Calculating the binary outcomes
	std::vector<double> binaryOutcome;
	binaryOutcome.reserve(probOutcome.size());

	for(const auto &val: probOutcome) {
		std::binomial_distribution<> dist(1, val);
		binaryOutcome.emplace_back( dist(this->generator) );
	}

	// Converting the distValues vector into Tensor and returning it	
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  auto targetTens = torch::from_blob(binaryOutcome.data(), {(int)this->rows, 1}, opts);

	appendToDataset(targetTens);

	// Creating label for the column
	appendLabel( std::string(1, 'y'));

}


void RandomDataset::appendLabel(std::string &base) {
	// If the input string is TARGET --> y  
	if(base.compare("y") == 0) {

		// Then won't concatenate anything to it
		this->labels.emplace_back(base);
	} else {

		// Else, append the column number to the input string --> x
		base.append( std::to_string(this->dataset.sizes()[1]));
		this->labels.emplace_back(base);
	}
}


void RandomDataset::prettyPrint() const {
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n";
	for(const auto &label: this->labels) {
		std::cout << "   " <<label << std::setfill(' ') << std::setw(6);
	}	
	std::cout << "\n";
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
	
	auto dataset_accessor = this->dataset.accessor<double, 2>();

	for(int i = 0; i < dataset_accessor.size(0); i++) {
		for(int j = 0; j < dataset_accessor.size(1); j++) {
			myfile << dataset_accessor[i][j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();
}
