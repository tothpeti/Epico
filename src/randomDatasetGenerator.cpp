#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"


RandomDataset::RandomDataset(const size_t& r)
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


void RandomDataset::testPrint() const {
	std::cout << "TEST \n";
	std::cout << std::fixed << std::setprecision(4);
	std::cout << this->features[0].reshape({1, this->features.size(1)}) << "\n";
	std::cout << this->target[1] << "\n";
	std::cout << "SIZE TARGET " << this->target.size(0) << "\n";
}


const torch::Tensor& RandomDataset::getFeatures() const {
	return this->features;
}


const torch::Tensor& RandomDataset::getTarget() const {
	return this->target;
}


torch::data::Example<> RandomDataset::get(size_t index)
{
	return { this->features[index], this->target[index]};
}


torch::optional<size_t> RandomDataset::size() const
{
	return this->features.size(0);
}


void RandomDataset::generateBinomialColumn(const size_t &numTrials, const double &prob, const double &weight){
	std::binomial_distribution<> d(numTrials, prob);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToFeatures(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}


void  RandomDataset::generateBernoulliColumn(const double &prob, const double &weight) {
	std::bernoulli_distribution d(prob);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToFeatures(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}


void RandomDataset::generateNormalColumn(const double &mean, const double &stddev, const double &weight){
	std::normal_distribution<double> d(mean, stddev);
	
	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToFeatures(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}


void RandomDataset::generateUniformDiscreteColumn(const int &a, const int &b, const double &weight) {
	std::uniform_int_distribution<> d(a, b);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToFeatures(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}


void RandomDataset::generateUniformRealColumn(const double &a, const double &b, const double &weight) {
	std::uniform_real_distribution<double> d(a, b);
	
	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToFeatures(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}


void RandomDataset::generateGammaColumn(const double &alpha, const double &beta, const double &weight) {
	std::gamma_distribution<double> d(alpha, beta);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDataset::generateRandomValuesHelper(d, weight);
	appendToFeatures(tens);

	// Creating label for the column
	appendLabel(std::string(1, 'x'));
}


void RandomDataset::generateBinaryTargetColumn() {
	auto inverseLogit = [](const double &p){
		return (std::exp(p) / (1 + std::exp(p)));
	};

	std::vector<double> probOutcome;
	probOutcome.reserve(this->rows);

	// Get iterators for the features 
	const auto features_accessor = this->features.accessor<double, 2>();

	// Calculating the row-by-row outcome's probability with inverseLogit
	for(int i = 0; i < features_accessor.size(0); i++) {
		double probSum = 0.0;
		for(int j = 0; j < features_accessor.size(1); j++) {
			probSum = probSum + features_accessor[i][j];
		}
		probOutcome.emplace_back( inverseLogit(probSum) );
	}

	// Calculating the binary outcomes
	std::vector<double> binaryOutcome;
	binaryOutcome.reserve(probOutcome.size());

	for(const auto &val: probOutcome) {
		std::binomial_distribution<> dist(1, val);
		binaryOutcome.emplace_back( dist(this->generator) );
	}

	// Converting the distValues vector into Tensor and returning it	
	const auto opts = torch::TensorOptions().dtype(torch::kFloat64);
	const auto targetTens = torch::from_blob(binaryOutcome.data(), {static_cast<int>(this->rows), 1}, opts);

	this->target = targetTens.clone().detach();

	// Creating label for the column
	appendLabel( std::string(1, 'y'));
}


RandomDataset::TrainTestDataset RandomDataset::trainTestSplit(const double &trainSplit, const double &testSplit) {
	// Calculating the new datasets sizes
	const size_t trainSize = this->rows * trainSplit;
	const size_t testSize = this->rows * testSplit;

	return {};
}


void RandomDataset::appendLabel(std::string &&base) {
	// If the input string is TARGET --> y  
	if(base.compare("y") == 0) {

		// Then won't concatenate anything to it
		this->labels.emplace_back(base);
	} else {

		// Else, append the column number to the input string --> x
		base.append( std::to_string(this->features.sizes()[1]));
		this->labels.emplace_back(base);
	}
}


void RandomDataset::prettyPrint() const {
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n";

	// Concatenate together the features and target columns
	const auto dataset = torch::cat({this->features, this->target}, 1);

	// Printing out the labels
	for(const auto &label: this->labels) {
		std::cout << "   " <<label << std::setfill(' ') << std::setw(6);
	}	
	std::cout << "\n";

	// Printing out the whole dataset
	std::cout << dataset;	
	std::cout << "\n";
}


std::ostream& operator<<(std::ostream &os, const RandomDataset &rd) {
	os.precision(3);
	os.fixed;
	os << rd.features;
	return os;
}


void RandomDataset::writeCSV() {
	std::ofstream myfile;
	myfile.open("example.csv");
	
	const auto features_accessor = this->features.accessor<double, 2>();

	for(int i = 0; i < features_accessor.size(0); i++) {
		for(int j = 0; j < features_accessor.size(1); j++) {
			myfile << features_accessor[i][j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();
}
