/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
 #pragma once

#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <utility>
#include <vector>

namespace torch
{
namespace serialize
{
class OutputArchive;
class InputArchive;
}  // namespace serialize
}  // namespace torch

namespace torch
{
namespace optim
{

class TORCH_API MyAdam : public Optimizer
{
   public:
    explicit MyAdam(std::vector<OptimizerParamGroup> param_groups, AdamOptions defaults = {})
        : Optimizer(std::move(param_groups), std::make_unique<AdamOptions>(defaults))
    {
        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
        TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
        auto betas = defaults.betas();
        TORCH_CHECK(0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
                    "Invalid beta parameter at index 0: ", std::get<0>(betas));
        TORCH_CHECK(0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
                    "Invalid beta parameter at index 1: ", std::get<1>(betas));
        TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
    }
    explicit MyAdam(std::vector<Tensor> params, AdamOptions defaults = {})
        : MyAdam({OptimizerParamGroup(std::move(params))}, defaults)
    {
    }

    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(serialize::OutputArchive& archive) const override;
    void load(serialize::InputArchive& archive) override;
    void shrinkenInternalState(int param_group_index, torch::Tensor indices_to_keep);
    void appendToInternalState(int param_group_index, int new_size);

   private:
    // template <typename Self, typename Archive>
    // static void serialize(Self& self, Archive& archive)
    //{
    //     _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adam);
    // }
};
}  // namespace optim
}  // namespace torch