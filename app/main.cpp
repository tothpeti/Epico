#include "RandomDatasetGenerator.hpp"
#include "RandomDataset.hpp"
#include "LogisticRegression.hpp"
#include "torch/torch.h"


int main() {
	/*
		Hyperparameters
	*/
	const int numberOfRows = 15;
	const int numberOfFeatures = 2; // number of feature columns 
	const int batchSize = 5;
	const int numberOfClasses = 1; // binary
	const size_t numberOfEpochs = 5;
	const double learningRate = 0.001;

	/*
		Creating columns for RandomDataset
	*/
	RandomDatasetGenerator::ColumnDataType bern{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,		//type
		{{"prob", 0.5}, {"weight", 0.75}} 							//parameters
	};	
	RandomDatasetGenerator::ColumnDataType bern2{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1.25}}
	};
	std::vector<RandomDatasetGenerator::ColumnDataType> cols;
	cols.push_back(bern);
	cols.push_back(bern2);

	auto rdGenerator = RandomDatasetGenerator(
		numberOfRows,
		cols, 
		true
	);
	
	auto rd = RandomDataset(
		rdGenerator.getFeatures(), 
		rdGenerator.getTarget(), 
		RandomDataset::Mode::kTrain, 
		0.6
	).map(torch::data::transforms::Stack<>());

	auto numberOfTrainSamples = rd.size().value();

	auto trainingDataLoader = torch::data::make_data_loader(
		std::move(rd), torch::data::DataLoaderOptions().batch_size(batchSize).workers(2)
	);


	for (torch::data::Example<>& batch : *trainingDataLoader) {
  	std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
  	for (int64_t i = 0; i < batch.data.size(0); ++i) {
    	std::cout << batch.target[i].item<int64_t>() << " ";
  	}
  	std::cout << std::endl;
	}
	std::cout << "OKI \n";

	// Logistic regression model
	std::cout << "Initializing model\n";
	//torch::nn::Linear model(inputSize, numberOfClasses);
	LogisticRegression model(numberOfFeatures);
	// Loss and optimizer
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));

	// Set floating point output precision
	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";

	//for(const auto &pair: model->named_parameters()) {
	//	std::cout << pair.key() << ": " << pair.value() << "\n";
	//}

	// Train the model
	model->train();
	for(size_t epoch = 0; epoch != numberOfEpochs; epoch++){

		// Initializing running metrics
		double runningLoss = 0.0;
		size_t numberOfCorrect = 0;

		for(auto& batch: *trainingDataLoader){
			optimizer.zero_grad();

		 	//auto data = batch.data.view({-1, 1});
		 	auto data = batch.data.to(torch::kFloat32);
		 	auto target = batch.target.to(torch::kFloat32);
			

		 	// Forward pass
		 	auto output = model->forward(data);
			

		 	// Calculate loss
		 	auto loss = torch::nn::functional::binary_cross_entropy(output, target);

		 	// Update running loss
		 	runningLoss += loss.item<double>() * data.size(0);

		 	// Calculate prediction
		 	auto prediction = output.argmax(1);

		 	// Update number of correctly classified samples
	 	 	numberOfCorrect += prediction.eq(target).sum().item<int64_t>();

			// Backward pass and optimize
			loss.backward();
			optimizer.step();
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