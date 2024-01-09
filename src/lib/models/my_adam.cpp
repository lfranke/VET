/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
 #include "my_adam.h"

#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"

#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <cmath>
#include <functional>
#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

namespace torch
{
namespace optim
{

Tensor MyAdam::step(LossClosure closure)
{
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr)
    {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }
    for (auto& group : param_groups_)
    {
        for (auto& p : group.params())
        {
            if (!p.grad().defined())
            {
                continue;
            }
            auto grad = p.grad();
            TORCH_CHECK(!grad.is_sparse(),
                        "Adam does not support sparse gradients" /*, please consider SparseAdam instead*/);
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto& options    = static_cast<AdamOptions&>(group.options());

            // State initialization
            if (param_state == state_.end())
            {
                auto state = std::make_unique<AdamParamState>();
                state->step(0);
                // Exponential moving average of gradient values
                state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
                // Exponential moving average of squared gradient values
                state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
                if (options.amsgrad())
                {
                    // Maintains max of all exp. moving avg. of sq. grad. values
                    state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
                }
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
            }

            auto& state          = static_cast<AdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto& exp_avg        = state.exp_avg();
            auto& exp_avg_sq     = state.exp_avg_sq();
            auto& max_exp_avg_sq = state.max_exp_avg_sq();

            state.step(state.step() + 1);
            auto beta1 = std::get<0>(options.betas());
            auto beta2 = std::get<1>(options.betas());

            auto bias_correction1 = 1 - std::pow(beta1, state.step());
            auto bias_correction2 = 1 - std::pow(beta2, state.step());

            if (options.weight_decay() != 0)
            {
                grad = grad.add(p, options.weight_decay());
            }

            // std::cout << Saiga::TensorInfo(grad) << " " << Saiga::TensorInfo(exp_avg) << " " << beta1 << " "
            //           << 1 - beta1 << std::endl;
            //  Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, 1 - beta1);
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

            Tensor denom;
            if (options.amsgrad())
            {
                // Maintains the maximum of all 2nd moment running avg. till now
                torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
                // Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
            }
            else
            {
                denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
            }

            auto step_size = options.lr() / bias_correction1;
            p.addcdiv_(exp_avg, denom, -step_size);
        }
    }
    return loss;
}
void MyAdam::shrinkenInternalState(int param_group_index, torch::Tensor indices_to_keep)
{
    SAIGA_ASSERT(param_group_index < param_groups_.size());
    auto& group = param_groups_[param_group_index];

    {
        for (auto& p : group.params())
        {
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            // not created yet -> will we initialized later
            if (param_state == state_.end()) continue;
            auto& state          = static_cast<AdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto& exp_avg        = state.exp_avg();
            auto& exp_avg_sq     = state.exp_avg_sq();
            auto& max_exp_avg_sq = state.max_exp_avg_sq();


            auto remove_selected = [&](torch::Tensor& t)
            {
                auto values_keep = t.index_select(t.sizes().size() - 1, indices_to_keep.squeeze().to(t.device()));
                t.set_(values_keep);
            };

            if (exp_avg.defined()) remove_selected(exp_avg);
            if (exp_avg_sq.defined()) remove_selected(exp_avg_sq);
            if (max_exp_avg_sq.defined()) remove_selected(max_exp_avg_sq);
        }
    }
}


void MyAdam::appendToInternalState(int param_group_index, int new_size)
{
    SAIGA_ASSERT(param_group_index < param_groups_.size());
    auto& group = param_groups_[param_group_index];
    {
        for (auto& p : group.params())
        {
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            // not created yet -> will we initialized later
            if (param_state == state_.end()) continue;
            auto& state          = static_cast<AdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto& exp_avg        = state.exp_avg();
            auto& exp_avg_sq     = state.exp_avg_sq();
            auto& max_exp_avg_sq = state.max_exp_avg_sq();

            auto add_selected = [&](torch::Tensor& t)
            {
                torch::Tensor new_vals;
                int new_point_size = new_size - t.size(-1);
                std::vector<long> sizes_tensor(t.sizes().size());
                for (int i = 0; i < t.sizes().size(); ++i) sizes_tensor[i] = t.sizes()[i];
                sizes_tensor[sizes_tensor.size() - 1] = new_point_size;

                new_vals = torch::zeros(sizes_tensor, t.options());
                auto t_n = torch::cat({t.clone(), new_vals}, -1);
                t.set_(t_n);
            };
            if (exp_avg.defined()) add_selected(exp_avg);
            if (exp_avg_sq.defined()) add_selected(exp_avg_sq);
            if (max_exp_avg_sq.defined()) add_selected(max_exp_avg_sq);
        }
    }
}

void MyAdam::save(serialize::OutputArchive& archive) const
{
    SAIGA_ASSERT(false);
}

void MyAdam::load(serialize::InputArchive& archive)
{
    SAIGA_ASSERT(false);
}
}  // namespace optim
}  // namespace torch