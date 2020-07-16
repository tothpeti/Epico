#include "RandomDatasetGenerator.hpp"
#include "RandomDataset.hpp"
#include "LogisticRegression.hpp"
#include "torch/torch.h"


int main() {
	/*
		Hyperparameters
	*/
	const int numberOfRows = 1000;
	const int numberOfFeatures = 10; // number of feature columns 
	const int batchSize = 50;
	const int numberOfClasses = 1; // binary
	const size_t numberOfEpochs = 5;
	const double learningRate = 0.001;

	/*
		Creating columns for RandomDataset
	*/
	RandomDatasetGenerator::ColumnDataType bern{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,		//type
		{{"prob", 0.5}, {"weight", 0.5}} 							//parameters
	};	
	RandomDatasetGenerator::ColumnDataType bern2{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 0.75}}
	};	
	RandomDatasetGenerator::ColumnDataType bern3{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1}}
	};
	RandomDatasetGenerator::ColumnDataType bern4{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1.25}}
	};
	RandomDatasetGenerator::ColumnDataType bern5{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1.5}}
	};

	RandomDatasetGenerator::ColumnDataType bern6{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1.75}}
	};
	RandomDatasetGenerator::ColumnDataType bern7{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 2}}
	};

	RandomDatasetGenerator::ColumnDataType bern8{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 1.5}}
	};

	RandomDatasetGenerator::ColumnDataType bern9{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 2}}
	};

	RandomDatasetGenerator::ColumnDataType bern10{
		RandomDatasetGenerator::DistributionTypes::Bernoulli,
		{{"prob", 0.5}, {"weight", 3}}
	};

	
	std::vector<RandomDatasetGenerator::ColumnDataType> cols;
	cols.push_back(bern);
	cols.push_back(bern2);
	cols.push_back(bern3);
	cols.push_back(bern4);
	cols.push_back(bern5);
	cols.push_back(bern6);
	cols.push_back(bern7);
	cols.push_back(bern8);
	cols.push_back(bern9);
	cols.push_back(bern10);

	auto rdGenerator = std::make_unique< RandomDatasetGenerator>(numberOfRows,cols, true);

	//auto rdTrain = RandomDataset(rdGenerator->getFeatures(), rdGenerator->getTarget(), RandomDataset::Mode::kTrain, 0.6
	//).map(torch::data::transforms::Stack<>());

	auto rdTrain = std::make_unique<RandomDataset>(rdGenerator->getFeatures(), rdGenerator->getTarget(), RandomDataset::Mode::kTrain, 0.6);
	auto numberOfTrainSamples = rdTrain->size().value();

	auto trainingDataLoader = torch::data::make_data_loader(
		std::move(rdTrain->map(torch::data::transforms::Stack<>())), torch::data::DataLoaderOptions().batch_size(batchSize).workers(2)
	);

	//auto rdTest = RandomDataset(rdGenerator->getFeatures(),rdGenerator->getTarget(), RandomDataset::Mode::kTest, 0.4).map(torch::data::transforms::Stack<>());
	auto rdTest = std::make_unique<RandomDataset>(rdGenerator->getFeatures(),rdGenerator->getTarget(), RandomDataset::Mode::kTest, 0.4);
	auto numberOfTestSamples = rdTest->size().value();

	auto testingDataLoader = torch::data::make_data_loader(
		std::move(rdTest->map(torch::data::transforms::Stack<>())), torch::data::DataLoaderOptions().batch_size(batchSize).workers(2)
	);

/*
	for (torch::data::Example<>& batch : *trainingDataLoader) {
  	std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
  	for (int64_t i = 0; i < batch.data.size(0); ++i) {
    	std::cout << batch.target[i].item<int64_t>() << " ";
  	}
  	std::cout << std::endl;
	}
*/


	// Logistic regression model
	std::cout << "Initializing model\n";
	//torch::nn::Linear model(inputSize, numberOfClasses);
	LogisticRegression model(numberOfFeatures);
	//auto model = std::make_shared<LogisticRegressionImpl>(numberOfFeatures);

	// Loss and optimizer
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));

	// Set floating point output precision
	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";


	// Containers for storing evaluation metrics
	std::vector<double> sensitivityVec;
	std::vector<double> specificityVec;
	std::vector<double> precisionVec;
	std::vector<double> f1scoreVec;
	std::vector<double> accuracyVec;

	// Train the model
	model->train();
	for(size_t epoch = 0; epoch != numberOfEpochs; epoch++){
		// Initializing running metrics
		double runningLoss = 0.0;
		int numberOfCorrect = 0;
		double sensitivities = 0.0;
		double specificities = 0.0;
		double precisions = 0.0;
		double f1scores = 0.0;
		double accuracies = 0.0;

		for(auto& batch: *trainingDataLoader){
			optimizer.zero_grad();

		 	//auto data = batch.data.view({-1, 1});
		 	auto data = batch.data.to(torch::kFloat32);
		 	auto target = batch.target.to(torch::kFloat32);
			//std::cout << "here it comes!!\n";	
			//std::cout << "target: \n" << target << "\n";
		 	// Forward pass
		 	auto output = model->forward(data);

		 	// Calculate loss
		 	auto loss = torch::nn::functional::binary_cross_entropy(output, target);

		 	// Update running loss
		 	//runningLoss += loss.item<double>() * data.size(0);
			runningLoss += loss.item<double>();


			auto tp = (target * round(output)).sum().to(torch::kFloat64);
			auto tn = ((1 - target) * (1 - round(output))).sum().to(torch::kFloat64);
			auto fp = ((1 - target) * round(output)).sum().to(torch::kFloat64);
			auto fn = (target * (1 - round(output))).sum().to(torch::kFloat64);

			// TODO CHECKING NULL
			auto precision = 0.0;
			auto recall = 0.0;

			if ( tp.item<double>() > 0 ) {
				precision = (tp / (tp + fp)).to(torch::kFloat64).item<double>();
				recall = (tp / (tp + fn)).to(torch::kFloat64).item<double>(); // alias sensitivit
			} else if ( fp.item<double>() > 0) {
				precision = (tp / (tp + fp)).to(torch::kFloat64).item<double>();
			} else if ( fn.item<double>() > 0) {
				recall = (tp / (tp + fn)).to(torch::kFloat64).item<double>(); // alias sensitivit
			}

			auto f1_score = 0.0;
			if (precision > 0 || recall > 0)
				f1_score = (2 * (precision * recall) / (precision + recall));

			auto specificity = 0.0;
			if ( tn.item<double>() > 0 || fp.item<double>() > 0) 
				specificity = (tn / (tn + fp)).to(torch::kFloat64).item<double>();
			

			/*
			std::cout << "true pos: "<< tp << "\n";
			std::cout << "true neg: "<< tn << "\n";
			std::cout << "false pos: "<< fp << "\n";
			std::cout << "false neg: "<< fn << "\n";
			std::cout << "precision: "<< precision << "\n";
			std::cout << "recall: "<< recall<< "\n";
			std::cout << "f1 score: "<< f1_score<< "\n";
			std::cout << "specificity: " << specificity << "\n";
			*/

			// FOR ACCURACY
			numberOfCorrect += round(output).view({-1, 1}).eq(target).sum().item<int>();
			
			sensitivities += recall;
			specificities += specificity;
			precisions += precision;
			f1scores += f1_score;

			//auto pred = round(output).view({-1, 1}).eq(target).sum().item();
			//std::cout << "output: \n" << round(output).view({-1, 1}) << "\n";
			//std::cout << "correct pred_y: \n" << round(output).view({-1, 1}).eq(target) <<"\n";
			//std::cout << "pred: " << pred << "\n";
			//std::cout << "numCorr: " << numberOfCorrect << "\n";

		 	// Calculate prediction
		 	//auto prediction = output.argmax(1);

		 	// Update number of correctly classified samples
	 	 	//numberOfCorrect += prediction.eq(target).sum().item<int64_t>();
			//numberOfCorrect += output.eq(target).sum();

			// Backward pass and optimize
			loss.backward();
			optimizer.step();
		}

		auto sampleMeanLoss = runningLoss / numberOfTrainSamples;
		//auto accuracy = static_cast<double>(numberOfCorrect) / numberOfTrainSamples;
		auto accuracy = (100* numberOfCorrect) / numberOfTrainSamples;
		accuracies += accuracy;

		sensitivityVec.push_back(sensitivities);
		specificityVec.push_back(specificities);
		precisionVec.push_back(precisions);
		f1scoreVec.push_back(f1scores);
		accuracyVec.push_back(accuracies);

		std::cout << "Epoch [" << (epoch + 1) << "/" << numberOfEpochs << "], Trainset - Loss: "
   	         << sampleMeanLoss << ", Accuracy: " << accuracy << '\n';	
	}
	std::cout << "Training finished!\n\n";

	for(const auto &elem: sensitivityVec) std::cout<< elem << " ";
	std::cout << "\n";
		
	for(const auto &elem: specificityVec) std::cout<< elem << " ";
	std::cout << "\n";

	for(const auto &elem: precisionVec) std::cout<< elem << " ";
	std::cout << "\n";

	for(const auto &elem: f1scoreVec) std::cout<< elem << " ";
	std::cout << "\n";

	for(const auto &elem: accuracyVec) std::cout<< elem << " ";
	std::cout << "\n\n";


 	std::cout << "Testing...\n";
	
	// Test the model
  model->eval();
  torch::NoGradGuard no_grad;

  double running_loss = 0.0;
  double numberOfCorrect = 0.0;
	double sensitivities = 0.0;
	double specificities = 0.0;
	double precisions = 0.0;
	double f1scores = 0.0;
	double accuracies = 0.0;

  for (const auto& batch : *testingDataLoader) {
  	auto data = batch.data.to(torch::kFloat32);
  	auto target = batch.target.to(torch::kFloat32);

    auto output = model->forward(data);

    auto loss = torch::nn::functional::binary_cross_entropy(output, target);

    //running_loss += loss.item<double>() * data.size(0);
    running_loss += loss.item<double>();


		auto tp = (target * round(output)).sum().to(torch::kFloat64);
		auto tn = ((1 - target) * (1 - round(output))).sum().to(torch::kFloat64);
		auto fp = ((1 - target) * round(output)).sum().to(torch::kFloat64);
		auto fn = (target * (1 - round(output))).sum().to(torch::kFloat64);

		// TODO CHECKING NULL
		auto precision = 0.0;
		auto recall = 0.0;


		if ( tp.item<double>() > 0 ) {
			precision = (tp / (tp + fp)).to(torch::kFloat64).item<double>();
			recall = (tp / (tp + fn)).to(torch::kFloat64).item<double>(); // alias sensitivit
		} else if ( fp.item<double>() > 0) {
			precision = (tp / (tp + fp)).to(torch::kFloat64).item<double>();
		} else if ( fn.item<double>() > 0) {
			recall = (tp / (tp + fn)).to(torch::kFloat64).item<double>(); // alias sensitivit
		}

		auto f1_score = 0.0;
		if (precision > 0 || recall > 0)
			f1_score = (2 * (precision * recall) / (precision + recall));

		std::cout<<"checking tn: " <<tn.item<double>() << "\n";
		std::cout<<"checking fp: " <<fp.item<double>() << "\n";
		auto specificity = 0.0;
		if ( tn.item<double>() > 0 || fp.item<double>() > 0) 
			specificity = (tn / (tn + fp)).to(torch::kFloat64).item<double>();
		

		// FOR ACCURACY
		numberOfCorrect += round(output).view({-1, 1}).eq(target).sum().item<int>();
			
		sensitivities += recall;
		specificities += specificity;
		precisions += precision;
		f1scores += f1_score;

  }

  std::cout << "Testing finished!\n";

  //auto test_accuracy = static_cast<double>(num_correct) / numberOfTestSamples;
	auto accuracy = (100*numberOfCorrect) / numberOfTestSamples;
  auto test_sample_mean_loss = running_loss / numberOfTestSamples;
	
  std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << accuracy << '\n';

	std::cout << "sensitivity: " << sensitivities << "\n";	
	std::cout << "specificity: " << specificities << "\n";	
	std::cout << "precision: " << precisions << "\n";	
	std::cout << "f1 score:: " << f1scores << "\n";	
	
	return 0;
}