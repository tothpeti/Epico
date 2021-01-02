# Independent Laboratory Work - Epico

This project was created for my master's independent laboratory work. The aim of the project was to create a tool which helps me run custom simulations on newly created datasets of self-made random dataset generator. Through the results of simulations, I try to conclude how each chosen Machine Learning model behaves in a changed environment/dataset.

## Table of Contents
1. [Why did I create a custom Random Dataset Generator?](#why-did-i-create-a-custom-random-dataset-generator)
	[How did I create this?](#how-did-i-create-this)
2. [Why did I create custom simulations?](#why-did-i-create-custom-simulations)
	[Types of custom simulations](types-of-custom-simulations)
3. [Methodology of research](#methodology-of-research)
4. [Results](#results)
5. [Structure of the project](#structure-of-the-project)


## Why did I create a custom Random Dataset Generator?

I needed to create the generator because:
  - If I use only one dataset, the simulations will not give unforeseen results.
  - I needed a tool which guarantees randomness in value creation, easily customizable, and can generate large number of datasets for my simulations.

### How did I create this?

First of all, to be able to guarantee randomness, I needed to use Monte Carlo sampling, which lets me create values in a random manner. This can be achieved in C++ by using the Mersenne twister engine. Secondly, to be able to customize the dataset structure, I implemented several distributions, which can be seen in the following table.  


| Name of distribution | Parameters                            |
| -------------------- | :------------------------------------ |
| Binomial             | number of trials, probability, weight |
| Bernoulli            | probability, weight                   |
| Normal               | mean, standard deviation, weight      |
| Uniform Discrete     | from, to, weight                      |
| Uniform Real         | from, to, weight                      |
| Gamma                | alpha, beta, weight                   |
<br>
Lastly, to be able to use binary-classification on the generated datasets, I needed binary output column. This could be achieved by using logistic regression's logit function.
<br>
<br>

## Why did I create custom simulations?
Goals of creating custom simulations were 
  - to scale up the number of simulations that were used during my work 
  - to create a plug&play solver that can be easily customized for my needs 
  - to see how each covariates influence the used machine learning models performing ability

### Types of custom simulations 

- Without column excluding:
	- measures the influence of all covariates 
	- fits Machine Learning model on all the covariates 
	
- With column excluding:
	- measures the influence of covariates separately
	- excludes one column in each iteration
	- fits Machine Learning model on the remaining dataset 
	- puts back the excluded column at the end 

## Methodology of research

- 5 Scenarios
- 4 Machine Learning models: 
	- Logistic Regression with default parameters
	- Random Forest with default parameters
	- Random Forest by hyperparameter optimized for accuracy
	- Random Forest by hyperparameter optimized for AUC value of ROC analysis
	
## 6. Results 
The documentation can be found in the __docs__ folder.

## 7. Structure of the project 
- __epico-cpp__ folder contains the implementation of the Random Dataset Generator 
- __epico-python__ folder contains the implementation of custom simulations, and data vizualization files

---
## License & copyright
© Péter Csaba Tóth