#include "randomDatasetGenerator.hpp"
#include "torch/torch.h"


int main() {
	/*
		Hyperparameters
	*/
	const int inputSize = 10; // number of rows
	const int batchSize = 5;
	const int numberOfClasses = 2; // binary
	const size_t numberOfEpochs = 5;
	const double learningRate = 0.001;

	/*
		Creating columns for RandomDataset
	*/
	RandomDataset::ColumnDataType bern{
		RandomDataset::DistributionTypes::Bernoulli,		//type
		{{"prob", 0.5}, {"weight", 0.75}} 							//parameters
	};	
	RandomDataset::ColumnDataType bern2{
		RandomDataset::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1.25}}
	};
	std::vector<RandomDataset::ColumnDataType> cols;
	cols.push_back(bern);
	cols.push_back(bern2);

	auto rd = RandomDataset(inputSize, cols, true).map(torch::data::transforms::Stack<>());

	auto numberOfTrainSamples = rd.size().value();

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(rd), batchSize
	);

	// Logistic regression model

	torch::nn::Linear model(inputSize, numberOfClasses);

	// Loss and optimizer
	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learningRate));

	// Set floating point output precision
	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";

	// Train the model

	for(size_t epoch = 0; epoch != numberOfEpochs; epoch++){

		// Initializing running metrics
		double runningLoss = 0.0;
		size_t numberOfCorrect = 0;
	
		for(auto& batch: *data_loader){
		 	auto data = batch.data.view({batchSize, -1});
		 	auto target = batch.target;

			std::cout << "after splitting values\n" << target << "\n";
		 	// Forward pass
		 	auto output = model->forward(data);

			std::cout << "after forward pass - here?\n";

		 	// Calculate loss
		 	auto loss = torch::nn::functional::binary_cross_entropy(output, target);

			std::cout << "after calc loss-here?\n";
			
		 	// Update running loss
		 	runningLoss += loss.item<double>() * data.size(0);

			std::cout << "after update running lo - here?\n";

		 	// Calculate prediction
		 	auto prediction = output.argmax(1);

		 	// Update number of correctly classified samples
	 	 	numberOfCorrect += prediction.eq(target).sum().item<int>();

			std::cout << "here?\n";

			// Backward pass and optimize
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			std::cout << "end-here?\n";
		}

		auto sampleMeanLoss = runningLoss / numberOfTrainSamples;
		auto accuracy = static_cast<double>(numberOfCorrect) / numberOfTrainSamples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << numberOfEpochs << "], Trainset - Loss: "
            << sampleMeanLoss << ", Accuracy: " << accuracy << '\n';	
	}
	
  std::cout << "Training finished!\n\n";

	/*
 	std::cout << "Testing...\n";
	
	// Test the model
  model->eval();
  torch::NoGradGuard no_grad;

  double running_loss = 0.0;
  size_t num_correct = 0;

  for (const auto& batch : *test_loader) {
  	auto data = batch.data.view({batch_size, -1}).to(device);
  	auto target = batch.target.to(device);

    auto output = model->forward(data);

    auto loss = torch::nn::functional::cross_entropy(output, target);

    running_loss += loss.item<double>() * data.size(0);

    auto prediction = output.argmax(1);

    num_correct += prediction.eq(target).sum().item<int64_t>();
  }

  std::cout << "Testing finished!\n";

  auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
  auto test_sample_mean_loss = running_loss / num_test_samples;

  std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
	*/
	return 0;
}