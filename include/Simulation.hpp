#pragma once

#include <vector>
#include "torch/torch.h"
#include "RandomDatasetGenerator.hpp"
#include "RandomDataset.hpp"

class Simulation {
private:

public:
    Simulation() = delete;

    template<typename ModelType>
    explicit Simulation(const RandomDatasetGenerator &rdg,
                        ModelType &model,
                        size_t epochs)
    {

    }

    void train();
    void test();

    ~Simulation() = default;
};
