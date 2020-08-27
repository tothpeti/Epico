#include <fstream>
#include <sstream>

#include "RandomDatasetGenerator.hpp"

RandomDatasetGenerator::RandomDatasetGenerator(
		const size_t& r, 
		const std::vector<RandomDatasetGenerator::ColumnDataType> &vec, 
		bool binary_target)
	: m_rows(r), m_generator((std::random_device())())  // it's like m_generator(rd()) syntax but it's inplace
{
	parseInputColumnData(vec);

	if(binary_target)
	{
		generateBinaryTargetColumn();
	}
}


void RandomDatasetGenerator::testPrint() const {
	std::cout << "TEST \n";
	std::cout << std::fixed << std::setprecision(4);
	std::cout << this->m_features[0].reshape({1, m_features.size(1)}) << "\n";
	std::cout << this->m_target[1] << "\n";
	std::cout << "SIZE TARGET " << this->m_target.size(0) << "\n";
}


torch::Tensor RandomDatasetGenerator::getFeatures() {
	return m_features;
}


torch::Tensor RandomDatasetGenerator::getTarget() {
	return m_target;
}


void RandomDatasetGenerator::generateBinomialColumn(const size_t &num_trials, const double &prob, const double &weight){
	std::binomial_distribution<> d(num_trials, prob);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDatasetGenerator::generateRandomValuesHelper(d);
	appendToFeatures(tens);

	// Saving weights for calculating outcome
	m_weights.push_back(weight);

	// Creating label for the column
	//appendLabel(std::string(1, 'x'));
}


void  RandomDatasetGenerator::generateBernoulliColumn(const double &prob, const double &weight) {
	std::bernoulli_distribution d(prob);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDatasetGenerator::generateRandomValuesHelper(d);
	appendToFeatures(tens);

    // Saving weights for calculating outcome
    m_weights.push_back(weight);

	// Creating label for the column
	//appendLabel(std::string(1, 'x'));
}


void RandomDatasetGenerator::generateNormalColumn(const double &mean, const double &stddev, const double &weight){
	std::normal_distribution<double> d(mean, stddev);
	
	// Creating Tensor column filled with distributed values
	auto tens = RandomDatasetGenerator::generateRandomValuesHelper(d);
	appendToFeatures(tens);

    // Saving weights for calculating outcome
    m_weights.push_back(weight);

	// Creating label for the column
	//appendLabel(std::string(1, 'x'));
}


void RandomDatasetGenerator::generateUniformDiscreteColumn(const int &from, const int &to, const double &weight) {
	std::uniform_int_distribution<> d(from, to);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDatasetGenerator::generateRandomValuesHelper(d);
	appendToFeatures(tens);

    // Saving weights for calculating outcome
    m_weights.push_back(weight);

	// Creating label for the column
	//appendLabel(std::string(1, 'x'));
}


void RandomDatasetGenerator::generateUniformRealColumn(const double &from, const double &to, const double &weight) {
	std::uniform_real_distribution<double> d(from, to);
	
	// Creating Tensor column filled with distributed values
	auto tens = RandomDatasetGenerator::generateRandomValuesHelper(d);
	appendToFeatures(tens);

    // Saving weights for calculating outcome
    m_weights.push_back(weight);

	// Creating label for the column
	//appendLabel(std::string(1, 'x'));
}


void RandomDatasetGenerator::generateGammaColumn(const double &alpha, const double &beta, const double &weight) {
	std::gamma_distribution<double> d(alpha, beta);

	// Creating Tensor column filled with distributed values
	auto tens = RandomDatasetGenerator::generateRandomValuesHelper(d);
	appendToFeatures(tens);

    // Saving weights for calculating outcome
    m_weights.emplace_back(weight);

	// Creating label for the column
	//appendLabel(std::string(1, 'x'));
}


void RandomDatasetGenerator::generateBinaryTargetColumn() {
	auto inverse_logit = [](double &p){
		return (std::exp(p) / (1 + std::exp(p)));
	};

	const double intercept = -1.5;

    m_outcome_probabilities.reserve(m_rows);

	// Get iterators for the m_features
	const auto features_accessor = m_features.accessor<double, 2>();

	// Calculating the row-by-row outcome's probability with inverse_logit
	for(int i = 0; i < features_accessor.size(0); i++) {
		double probSum = 0.0;
		for(int j = 0; j < features_accessor.size(1); j++) {
			probSum = probSum + (features_accessor[i][j] * m_weights[j]);
		}
		auto logit = probSum + intercept;
		auto p = inverse_logit(logit);
		m_outcome_probabilities.push_back(p);
	}

	// Calculating the binary outcomes
	std::vector<double> binaryOutcome;
    binaryOutcome.reserve(m_outcome_probabilities.size());

	// Generating 0 or 1 value
	int counter = 0;
	for(const auto &val: m_outcome_probabilities) {
	    if (val < 0.5)
        {
            binaryOutcome.push_back(0.0);
        }
	    else if (val >= 0.5)
        {
	        counter++;
            binaryOutcome.push_back(1.0);
        }
	}
	//std::cout << "This many 1s : " << counter << "\n\n";

	// Converting the binaryOutcome vector into Tensor
	const auto opts = torch::TensorOptions().dtype(torch::kFloat64);
	const auto targetTens = torch::from_blob(binaryOutcome.data(), {static_cast<int32_t>(m_rows), 1}, opts);

	// Move the generated binary outcome tensor column to m_target
	m_target = targetTens.clone().detach();

	// Creating label for the column
	//appendLabel( std::string(1, 'y'));
}


void RandomDatasetGenerator::appendLabel(std::string &&base) {
	// If the input string is TARGET --> y  
	if(base == "y") {

		// Then won't concatenate anything to it
		this->m_labels.emplace_back(base);
	} else {

		// Else, append the column number to the input string --> x
		base.append( std::to_string(m_features.sizes()[1]));
		this->m_labels.emplace_back(base);
	}
}

void RandomDatasetGenerator::prettyPrint() const {
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n";

	// Concatenate together the m_features and m_target columns
	const auto dataset = torch::cat({m_features, m_target}, 1);

	// Printing out the m_labels
	for(const auto &label: m_labels) {
		std::cout << "   " <<label << std::setfill(' ') << std::setw(6);
	}	
	std::cout << "\n";

	// Printing out the whole dataset
	std::cout << dataset;	
	std::cout << "\n";
}


std::ostream& operator<<(std::ostream &os, const RandomDatasetGenerator &rd) {
	os.precision(3);
	os.fixed;
	os << rd.m_features;
	return os;
}


void RandomDatasetGenerator::writeCSV() {
	std::ofstream myfile;
	myfile.open("example.csv");
	
	const auto features_accessor = m_features.accessor<double, 2>();

	for(int i = 0; i < features_accessor.size(0); i++) {
		for(int j = 0; j < features_accessor.size(1); j++) {
			myfile << features_accessor[i][j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();
}


void RandomDatasetGenerator::parseInputColumnData(const std::vector<ColumnDataType> &vec) {
	for(const auto &col: vec) {
		switch (col.type)
		{
			case RandomDatasetGenerator::DistributionTypes::Binomial: 
				generateBinomialColumn(col.parameters.at("numtrials"),
														 col.parameters.at("prob"),
														 col.parameters.at("weight"));			
				break;

			case RandomDatasetGenerator::DistributionTypes::Bernoulli:
				generateBernoulliColumn(col.parameters.at("prob"),
															col.parameters.at("weight"));
				break;

			case RandomDatasetGenerator::DistributionTypes::Normal:
				generateNormalColumn(col.parameters.at("mean"),
													 col.parameters.at("stddev"),
													 col.parameters.at("weight"));
				break;
			
			case RandomDatasetGenerator::DistributionTypes::UniformDiscrete:
				generateUniformDiscreteColumn(col.parameters.at("from"),
																		col.parameters.at("to"),
																		col.parameters.at("weight"));	
				break;

			case RandomDatasetGenerator::DistributionTypes::UniformReal:
				generateUniformRealColumn(col.parameters.at("from"),
                                            col.parameters.at("to"),
                                            col.parameters.at("weight"));
				break;
		
			case RandomDatasetGenerator::DistributionTypes::Gamma:
				generateGammaColumn(col.parameters.at("alpha"),
													col.parameters.at("beta"),
													col.parameters.at("weight"));
				break;
		}
	}
}

std::vector<double> RandomDatasetGenerator::getProbabilityOutput(const std::string &train_or_test, double split_size) {
    if (train_or_test == "test")
    {
        auto current_dataset_length = m_features.size(0);
        auto cut_from = std::round(m_features.size(0) * (1 - split_size));
        //std::cout << "CUT_FROM: " << cut_from <<"\n";
        return {m_outcome_probabilities.begin() + cut_from, m_outcome_probabilities.end() };
    }
    else
    {
        auto cut_till = std::round(m_features.size(0) * split_size);

        return {m_outcome_probabilities.begin(), m_outcome_probabilities.begin()+cut_till};
    }
}

