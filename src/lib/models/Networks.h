/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/vision/torch/PartialConvUnet2d.h"


/// NEW

class MultiStartBlockImpl : public UnetBlockImpl
{
   public:
    using UnetBlockImpl::forward;

    MultiStartBlockImpl(int in_channels, int out_channels, std::string conv_block, std::string norm_str,
                        std::string pooling_str, std::string activation_str)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);
        conv = UnetBlockFromString(conv_block, in_channels, out_channels, 3, 1, 1, norm_str, activation_str);

        register_module("conv", conv.ptr());
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        auto res = conv.forward<std::pair<at::Tensor, at::Tensor>>(x, mask);
        return res;
    }

    // GatedBlock conv = nullptr;
    torch::nn::AnyModule conv;
};

TORCH_MODULE(MultiStartBlock);



class MultiScaleUnet2dSlimImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dSlimImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dSlim " << std::endl;
        std::vector<int> num_input_channels_per_layer;
        // std::vector<int> filters = {4, 8, 16, 16, 16};
        std::vector<int> filters = params.filters_network;

        std::vector<int> num_input_channels(params.num_input_layers, params.num_input_channels);
        for (int i = params.num_input_layers; i < 5; ++i)
        {
            num_input_channels.push_back(0);
        }
        for (int i = 0; i < 5; ++i)
        {
            auto& f = filters[i];
            f       = f * params.feature_factor;
            if (params.add_input_to_filters && i >= 1)
            {
                f += num_input_channels[i];
            }

            if (i >= 1)
            {
                SAIGA_ASSERT(f >= num_input_channels[0]);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        //  start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");
        //  register_module("start", start.ptr());



        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);

        for (int i = 0; i < params.num_layers; ++i)
        {
            // Down[i] transforms from layer (i) -> (i+1)
            int multistart_in  = params.num_input_channels;  // filters[i];
            int multistart_out = filters[i];
            multistart[i] = MultiStartBlock(multistart_in, multistart_out, params.conv_block, params.norm_layer_down,
                                            params.pooling, params.activation);
            register_module("multistart" + std::to_string(i + 1), multistart[i]);
        }
        for (int i = 0; i < params.num_layers - 1; ++i)
        // for (int i = params.num_layers - 1; i >= 1; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleBlock(up_in, up_out, params.conv_block_up, params.upsample_mode, params.norm_layer_up,
                                       params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }
        // multistart[params.num_layers - 1] = MultiStartBlock(params.num_input_channels, filters[params.num_layers -
        // 1], params.conv_block,
        //     params.norm_layer_down, params.pooling, params.activation);

        multi_channel_masks = params.conv_block == "partial_multi";
        need_up_masks       = params.conv_block_up == "partial_multi";
        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        if (multi_channel_masks)
        {
            torch::NoGradGuard ngg;
            // multi channel partial convolution needs a mask value for each channel.
            // Here, we just repeat the masks along the channel dimension.
            for (int i = 0; i < inputs.size(); ++i)
            {
                auto& ma = masks[i];
                auto& in = inputs[i];
                if (ma.size(1) == 1 && in.size(1) > 1)
                {
                    ma = ma.repeat({1, in.size(1), 1, 1});
                }
            }
        }

        std::pair<torch::Tensor, torch::Tensor> d[MultiScaleUnet2dParams::max_layers - 1];

        //!
        //        d[0] = multistart[0].forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        // Loops Range: [1,2, ... , layers-1]
        // At 5 layers we have only 4 stages
        for (int i = 0; i < params.num_layers; ++i)
        {
            d[i] = multistart[i]->forward(inputs[i]);
        }

        if (!need_up_masks)
        {
            for (int i = 0; i < params.num_layers; ++i)
            {
                d[i].second = torch::Tensor();
            }
        }

        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 1; i >= 1; --i)
        {
            d[i - 1] = up[i - 1]->forward(d[i], d[i - 1]);
        }
        return final->forward(d[0].first);
    }

    MultiScaleUnet2dParams params;
    bool multi_channel_masks = false;
    bool need_up_masks       = false;

    //  torch::nn::AnyModule start;
    torch::nn::Sequential final;

    MultiStartBlock multistart[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
    UpsampleBlock up[MultiScaleUnet2dParams::max_layers - 1]           = {nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dSlim);



class UpsampleUltraBlockImpl : public torch::nn::Module
{
   public:
    UpsampleUltraBlockImpl(int in_channels, int out_channels, int num_input_channels, std::string conv_block,
                           std::string upsample_mode = "deconv", std::string norm_str = "id",
                           std::string activation = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        if (upsample_mode != "deconv")
        {
            if (conv_block == "partial_multi")
            {
                conv1 = torch::nn::AnyModule(
                    PartialConv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1), true));
            }
            else
            {
                conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
            }
        }
        // conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);
        conv2 = UnetBlockFromString(conv_block, out_channels + num_input_channels, out_channels, 3, 1, 1, norm_str,
                                    activation);


        up_mask = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest));

        register_module("up", up);
        register_module("up_mask", up_mask);

        if (!conv1.is_empty())
        {
            register_module("conv1", conv1.ptr());
        }
        register_module("conv2", conv2.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        // SAIGA_ASSERT(skip.first.size(2) == same_layer_as_skip.first.size(2) &&
        //              skip.first.size(3) == same_layer_as_skip.first.size(3));
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
        {
            return torch::cat({below, skip}, 1);
        }
        else
        {
            return torch::cat({below, CenterCrop2D(skip, below.sizes())}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> skip)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(skip.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> same_layer_as_skip;
        same_layer_as_skip.first = up->forward(layer_below.first);

        if (layer_below.second.defined())
        {
            same_layer_as_skip.second = up_mask->forward(layer_below.second);
            // SAIGA_ASSERT(skip.second.size(2) == same_layer_as_skip.second.size(2) &&
            //              skip.second.size(3) == same_layer_as_skip.second.size(3));
        }

        if (!conv1.is_empty())
        {
            same_layer_as_skip =
                conv1.forward<std::pair<at::Tensor, at::Tensor>>(same_layer_as_skip.first, same_layer_as_skip.second);
        }


        std::pair<at::Tensor, at::Tensor> output;
        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        output.first = CombineBridge(same_layer_as_skip.first, skip.first);

        if (layer_below.second.defined())
        {
            // output.second = torch::cat({same_layer_as_skip.second, skip.second}, 1);
            output.second = CombineBridge(same_layer_as_skip.second, skip.second);
        }

        return conv2.forward<std::pair<at::Tensor, at::Tensor>>(output.first, output.second);
    }

    torch::nn::Sequential up;
    torch::nn::Upsample up_mask = nullptr;
    torch::nn::AnyModule conv1;
    torch::nn::AnyModule conv2;
};

TORCH_MODULE(UpsampleUltraBlock);



class MultiScaleUnet2dUltraSlimImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dUltraSlimImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dUltraSlim " << std::endl;
        std::vector<int> num_input_channels_per_layer;
        // std::vector<int> filters = {4, 8, 16, 16, 16};
        std::vector<int> filters = params.filters_network;

        std::vector<int> num_input_channels(params.num_input_layers, params.num_input_channels);
        for (int i = params.num_input_layers; i < 5; ++i)
        {
            num_input_channels.push_back(0);
        }
        for (int i = 0; i < 5; ++i)
        {
            auto& f = filters[i];
            f       = f * params.feature_factor;
            if (params.add_input_to_filters && i >= 1)
            {
                f += num_input_channels[i];
            }

            if (i >= 1)
            {
                SAIGA_ASSERT(f >= num_input_channels[0]);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        //  start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");
        //  register_module("start", start.ptr());



        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);

        for (int i = params.num_layers - 1; i < params.num_layers; ++i)
        {
            // Down[i] transforms from layer (i) -> (i+1)
            int multistart_in  = params.num_input_channels;  // filters[i];
            int multistart_out = filters[i];
            multistart[i] = MultiStartBlock(multistart_in, multistart_out, params.conv_block, params.norm_layer_down,
                                            params.pooling, params.activation);
            register_module("multistart" + std::to_string(i + 1), multistart[i]);
        }
        //  start      = MultiStartBlock(params.num_input_channels, filters[params.num_layers-1], params.conv_block,
        //  params.norm_layer_down, params.pooling,
        //                             params.activation);
        // register_module("start", start.ptr());

        for (int i = 0; i < params.num_layers - 1; ++i)
        // for (int i = params.num_layers - 1; i >= 1; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleUltraBlock(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                            params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }
        // multistart[params.num_layers - 1] = MultiStartBlock(params.num_input_channels, filters[params.num_layers -
        // 1], params.conv_block,
        //     params.norm_layer_down, params.pooling, params.activation);

        multi_channel_masks = params.conv_block == "partial_multi";
        need_up_masks       = params.conv_block_up == "partial_multi";
        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        if (multi_channel_masks)
        {
            torch::NoGradGuard ngg;
            // multi channel partial convolution needs a mask value for each channel.
            // Here, we just repeat the masks along the channel dimension.
            for (int i = 0; i < inputs.size(); ++i)
            {
                auto& ma = masks[i];
                auto& in = inputs[i];
                if (ma.size(1) == 1 && in.size(1) > 1)
                {
                    ma = ma.repeat({1, in.size(1), 1, 1});
                }
            }
        }

        std::pair<torch::Tensor, torch::Tensor> d[MultiScaleUnet2dParams::max_layers - 1];

        //!
        //        d[0] = multistart[0].forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        // Loops Range: [1,2, ... , layers-1]
        // At 5 layers we have only 4 stages

        for (int i = 0; i < params.num_layers; ++i)
        {
            d[i] = std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);  // multistart[i]->forward(inputs[i]);
        }
        for (int i = params.num_layers - 1; i < params.num_layers; ++i)
        {
            d[i] = multistart[i]->forward(inputs[i]);
        }
        // d[params.num_layers-1] = start.forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[params.num_layers-1]);
        //   d[params.num_layers-1] = start.forward<std::pair<torch::Tensor,
        //   torch::Tensor>>(inputs[params.num_layers-1], masks[params.num_layers-1]);


        if (!need_up_masks)
        {
            for (int i = 0; i < params.num_layers; ++i)
            {
                d[i].second = torch::Tensor();
            }
        }

        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 1; i >= 1; --i)
        {
            d[i - 1] = up[i - 1]->forward(d[i], d[i - 1]);
        }
        return final->forward(d[0].first);
    }

    MultiScaleUnet2dParams params;
    bool multi_channel_masks = false;
    bool need_up_masks       = false;

    // torch::nn::AnyModule start;
    torch::nn::Sequential final;

    MultiStartBlock multistart[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
    //  MultiStartBlock multistarts[1] = {nullptr};
    UpsampleUltraBlock up[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dUltraSlim);
