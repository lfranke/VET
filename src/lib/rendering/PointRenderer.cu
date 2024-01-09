/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// #undef CUDA_DEBUG
// #define CUDA_NDEBUG

// #include "saiga/colorize.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"

#include "cooperative_groups.h"
#include <curand_kernel.h>

curandState* curand_state_h;

template <typename IndexType>
__global__ void CombineAndFill(float* background_color, ImageView<float> weight,
                               StaticDeviceTensor<float, 3> out_neural_image)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto cou              = weight(gy, gx);
    auto texture_channels = out_neural_image.sizes[0];

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_neural_image(ci, gy, gx) = background_color[ci];
        }
    }
    else
    {
        // divide by weight
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_neural_image(ci, gy, gx) /= cou;
        }
    }
}

__global__ void CombineAndFillBlend(float* background_color, ImageView<float> weight,
                                    StaticDeviceTensor<float, 3> out_neural_image)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto alpha_dest       = weight(gy, gx);
    auto texture_channels = out_neural_image.sizes[0];

    if (alpha_dest > 0)
    {
        // copy background into output
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_neural_image(ci, gy, gx) += alpha_dest * background_color[ci];
        }
    }
    //    else
    //    {
    //        // divide by weight
    //        for (int ci = 0; ci < texture_channels; ++ci)
    //        {
    //            out_neural_image(ci, gy, gx) /= cou;
    //        }
    //    }
}

template <typename IndexType>
__global__ void CombineAndFillDepth(ImageView<float> depths, StaticDeviceTensor<float, 3> out_depth_image)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= depths.width || gy >= depths.height) return;

    auto dep              = depths(gy, gx);
    auto texture_channels = out_depth_image.sizes[0];

    if (dep == 0)
    {
        // copy background into output
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_depth_image(ci, gy, gx) = 0;
        }
    }
    else
    {
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_depth_image(ci, gy, gx) = (MAX_DEPTH_CONST - dep) / MAX_DEPTH_CONST;
        }
    }
}

__global__ void DebugWeightToColor(ImageView<float> weight, StaticDeviceTensor<float, 3> out_neural_image,
                                   float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto cou = weight(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        // float t = ::saturate(x);
        // vec3 c  = saturate(vec3(sqrt(t), t * t * t, std::max(sin(3.1415 * 1.75 * t), pow(t, 12.0))));

        vec3 c = colorizeTurbo(x);

        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}

__global__ void DebugDepthToColor(ImageView<float> depth, StaticDeviceTensor<float, 3> out_neural_image,
                                  float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= depth.width || gy >= depth.height) return;

    auto cou = depth(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);
        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void DebugCountingsToColor(ImageView<int> counting, StaticDeviceTensor<float, 3> out_neural_image,
                                      float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= counting.width || gy >= counting.height) return;

    auto cou = counting(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);

        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void CreateMask(StaticDeviceTensor<float, 4> in_weight, StaticDeviceTensor<float, 4> out_mask,
                           float background_value, int b)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;


    if (!in_weight.Image().template inImage(gy, gx)) return;

    auto w = in_weight.At({b, 0, gy, gx});

    if (w == 0)
    {
        out_mask.At({b, 0, gy, gx}) = background_value;
    }
    else
    {
        out_mask(b, 0, gy, gx) = 1;
    }
}


void PointRendererCache::Build(NeuralRenderInfo* info, bool forward)
{
    this->info        = info;
    this->num_batches = info->images.size();



    this->render_mode = (RenderMode)info->params.render_mode;



    SAIGA_OPTIONAL_TIME_MEASURE("Build Cache", info->timer_system);
    static_assert(sizeof(Packtype) == 8);

    SAIGA_ASSERT(num_batches > 0);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Allocate", info->timer_system);
        Allocate(info, forward);
    }

    if (forward)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Initialize", info->timer_system);
        InitializeData(forward);
    }
    else
    {
        output_gradient_texture    = torch::zeros_like(info->scene->texture->texture);
        output_gradient_confidence = torch::zeros_like(info->scene->texture->confidence_value_of_point);
        output_gradient_background = torch::zeros_like(info->scene->texture->background_color);


        if (info->scene->point_cloud_cuda->t_position.requires_grad())
        {
            output_gradient_points = torch::zeros_like(info->scene->point_cloud_cuda->t_position);
            output_gradient_point_count =
                torch::zeros({output_gradient_points.size(0)}, output_gradient_points.options());
        }

        if (info->scene->dynamic_refinement_t.sizes().size() > 1)
        {
            output_gradient_dynamic_points = torch::zeros_like(info->scene->dynamic_refinement_t);
            output_gradient_dynamic_point_count =
                torch::zeros({output_gradient_dynamic_points.size(0), output_gradient_dynamic_points.size(1)},
                             output_gradient_points.options());
        }

        if (info->scene->poses->tangent_poses.requires_grad())
        {
            output_gradient_pose_tangent = torch::zeros_like(info->scene->poses->tangent_poses);
            output_gradient_pose_tangent_count =
                torch::zeros({info->scene->poses->tangent_poses.size(0)},
                             info->scene->poses->tangent_poses.options().dtype(torch::kFloat32));
        }

        if (info->scene->intrinsics->intrinsics.requires_grad())
        {
            output_gradient_intrinsics       = torch::zeros_like(info->scene->intrinsics->intrinsics);
            output_gradient_intrinsics_count = torch::zeros({info->scene->intrinsics->intrinsics.size(0)},
                                                            info->scene->intrinsics->intrinsics.options());
        }
    }
}

void PointRendererCache::Allocate(NeuralRenderInfo* info, bool forward)
{
    auto& fd = info->images.front();
    int h    = fd.h;
    int w    = fd.w;

    SAIGA_ASSERT(info->scene->point_cloud_cuda);
    SAIGA_ASSERT(info->scene->texture);

    std::vector<int> new_cache_size = {(int)info->scene->texture->texture.size(0),
                                       info->scene->point_cloud_cuda->Size(),
                                       info->num_layers,
                                       num_batches,
                                       h,
                                       w};


    bool size_changed = new_cache_size != cache_size;

    if (size_changed)
    {
        cache_has_forward  = false;
        cache_has_backward = false;
    }

    bool need_allocate_forward  = !cache_has_forward && forward;
    bool need_allocate_backward = !cache_has_backward && !forward;

    if (!need_allocate_forward && !need_allocate_backward)
    {
        // std::cout << "skip allocate" << std::endl;
        return;
    }

    // std::cout << "allocate render cache " << need_allocate_forward << " " << need_allocate_backward << " "
    //          << size_changed << std::endl;

    if (curand_state_h == nullptr)
    {
        cudaMalloc(&curand_state_h, sizeof(curandState));
        Saiga::CUDA::initRandom(ArrayView<curandState>(curand_state_h, 1), 0);
    }
    if (size_changed)
    {
        layers_cuda.resize(info->num_layers);
    }

    float scale = 1;
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(w > 0 && h > 0);
        auto& l = layers_cuda[i];

        if (need_allocate_forward || need_allocate_backward)
        {
            l.depth     = torch::empty({num_batches, 1, h, w},
                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            l.weight    = torch::empty({num_batches, 1, h, w},
                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            l.max_depth = torch::empty({num_batches, 1, h, w},
                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            l.counting =
                torch::empty({num_batches, 1, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
            l.scanned_counting =
                torch::empty({num_batches, 1, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
            l.per_image_atomic_counters =
                torch::empty({num_batches, 1, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
        }



        l.size  = {w, h};
        l.scale = scale;

        h /= 2;
        w /= 2;
        scale *= 0.5;
    }



    if (need_allocate_forward)
    {
        dropout_points = torch::empty({num_batches, info->scene->point_cloud_cuda->Size()},
                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    cache_size = new_cache_size;
    if (forward)
    {
        cache_has_forward = true;
    }
    else
    {
        cache_has_backward = true;
    }
}

void PointRendererCache::InitializeData(bool forward)
{
    if (forward)
    {
        for (auto& l : layers_cuda)
        {
            l.depth.fill_(MAX_DEPTH_CONST);
            l.weight.zero_();
            l.max_depth.fill_(MAX_DEPTH_CONST);
            l.counting.zero_();
            l.scanned_counting.zero_();
            l.per_image_atomic_counters.zero_();
        }


        // This is created every frame, because we 'move' it to the output
        output_forward.resize(info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            int w                = layers_cuda[i].size(0);
            int h                = layers_cuda[i].size(1);
            int texture_channels = info->params.num_texture_channels;
            output_forward[i]    = torch::zeros({num_batches, texture_channels, h, w},
                                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        }

        if (info->params.add_depth_to_network)
        {
            output_forward_depthbuffer.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                int w                = layers_cuda[i].size(0);
                int h                = layers_cuda[i].size(1);
                int texture_channels = 1;
                output_forward_depthbuffer[i] =
                    torch::zeros({num_batches, texture_channels, h, w},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

        if (info->params.output_background_mask)
        {
            output_forward_blend.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                auto& l = layers_cuda[i];
                output_forward_blend[i] =
                    torch::zeros({num_batches, 1, l.size.y(), l.size.x()},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

#if 0
        for (auto& t : output_forward)
        {
            t.zero_();
        }
#endif

        if (info->params.dropout > 0)
        {
            dropout_points.bernoulli_(info->params.dropout);
        }
        else
        {
            dropout_points.zero_();
        }
    }
}


DeviceRenderParams PointRendererCache::PrepareDeviceRenderParams()
{
    static DeviceRenderParams drp;

    drp = DeviceRenderParams(info->params);
    if (info->scene)
    {
        drp._poses     = (Sophus::SE3d*)info->scene->poses->poses_se3.data_ptr<double>();
        drp.intrinsics = info->scene->intrinsics->intrinsics;
    }
    drp.num_layers = info->num_layers;

    for (int i = 0; i < info->num_layers; ++i)
    {
        drp.depth[i]                     = layers_cuda[i].depth;
        drp.weight[i]                    = layers_cuda[i].weight;
        drp.max_depth[i]                 = layers_cuda[i].max_depth;
        drp.counting[i]                  = layers_cuda[i].counting;
        drp.per_image_atomic_counters[i] = layers_cuda[i].per_image_atomic_counters;
    }

    if (info->params.use_point_adding_and_removing_module)
    {
        if (gradient_of_forward_pass_x.defined())
        {
            drp.gradient_of_forward_pass_x = gradient_of_forward_pass_x;
        }
    }

    drp.curand_state  = curand_state_h;
    drp.current_epoch = info->current_epoch;

    return drp;
}
DeviceTexture PointRendererCache::PrepareDeviceTexture()
{
    static DeviceTexture d_tex;

    d_tex.in_texture = info->scene->texture->texture;
    // std::cout << TensorInfo(info->scene->texture->texture.contiguous()) << std::endl;
    d_tex.points_confidence_value = info->scene->texture->confidence_value_of_point;

    return d_tex;
}


DeviceBackwardParams PointRendererCache::PrepareDeviceBackwardParams()
{
    DeviceBackwardParams dbp = {0};

    if (output_gradient_pose_tangent.defined())
    {
        SAIGA_ASSERT(output_gradient_pose_tangent.size(1) == 6);
        dbp.out_gradient_pose       = (Vec6*)output_gradient_pose_tangent.data_ptr<double>();
        dbp.out_gradient_pose_count = output_gradient_pose_tangent_count.data_ptr<float>();
    }

    if (output_gradient_points.defined())
    {
        SAIGA_ASSERT(output_gradient_points.size(1) == 4);
        dbp.out_gradient_points       = (vec4*)output_gradient_points.data_ptr<float>();
        dbp.out_gradient_points_count = output_gradient_point_count.data_ptr<float>();
    }

    if (output_gradient_dynamic_points.defined())
    {
        SAIGA_ASSERT(output_gradient_dynamic_points.size(2) == 3);
        dbp.out_gradient_dynamic_points       = output_gradient_dynamic_points;
        dbp.out_gradient_dynamic_points_count = output_gradient_dynamic_point_count;
    }
    else
    {
        dbp.out_gradient_dynamic_points.data       = nullptr;
        dbp.out_gradient_dynamic_points_count.data = nullptr;
    }

    if (output_gradient_intrinsics.defined())
    {
        dbp.out_gradient_intrinsics       = output_gradient_intrinsics;
        dbp.out_gradient_intrinsics_count = output_gradient_intrinsics_count.data_ptr<float>();
    }

    dbp.out_gradient_texture    = output_gradient_texture;
    dbp.out_gradient_confidence = output_gradient_confidence;
    // std::cout << "dbp.out_gradient_texture" << TensorInfo(output_gradient_texture) << std::endl;
    // std::cout << "dbp.out_gradient_confidence" << TensorInfo(output_gradient_confidence) << std::endl;
    SAIGA_ASSERT(image_gradients.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(image_gradients[i].dim() == 4);
        dbp.in_gradient_image[i] = image_gradients[i];
    }
    return dbp;
}

void PointRendererCache::CombineAndFill(int batch, torch::Tensor background_color)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    SAIGA_ASSERT(output_forward.size() == info->num_layers);


    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);
        SAIGA_ASSERT(batch < output_forward[i].size(0));
        auto in_out_neural_image = output_forward[i][batch];

        //   std::cout << in_out_neural_image.sizes()<<std::endl;

        auto weights = l.BatchViewWeights(batch);
        ::CombineAndFill<unsigned int><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(background, weights, in_out_neural_image);

        if (info->params.add_depth_to_network)
        {
            auto depth_buff         = l.BatchViewDepth(batch);
            auto in_out_depth_image = output_forward_depthbuffer[i][batch];
            ::CombineAndFillDepth<unsigned int><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(depth_buff, in_out_depth_image);
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::CombineAndFillBlend(int batch, torch::Tensor background_color)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    SAIGA_ASSERT(output_forward.size() == info->num_layers);


    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);
        SAIGA_ASSERT(batch < output_forward[i].size(0));
        auto in_out_neural_image = output_forward[i][batch];

        //   std::cout << in_out_neural_image.sizes()<<std::endl;

        auto weights = l.BatchViewWeights(batch);
        ::CombineAndFillBlend<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(background, weights, in_out_neural_image);

        if (info->params.add_depth_to_network)
        {
            SAIGA_ASSERT(false);
            auto depth_buff         = l.BatchViewDepth(batch);
            auto in_out_depth_image = output_forward_depthbuffer[i][batch];
            ::CombineAndFillDepth<unsigned int><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(depth_buff, in_out_depth_image);
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}


void PointRendererCache::CreateMask(int batch, float background_value)
{
    SAIGA_ASSERT(output_forward_blend.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);

        SAIGA_ASSERT(output_forward_blend[i].size(2) == l.size.y());
        ::CreateMask<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(l.weight, output_forward_blend[i], background_value, batch);
    }
    CUDA_SYNC_CHECK_ERROR();
}



std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> BlendPointCloudForward(
    torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info)
{
    SAIGA_ASSERT(info->cache);

    // int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    PointRendererCache& cache = *info->cache;


    cache.Build(info, true);

    int num_batches = cache.num_batches;

    cache.PushParametersForward();

    if (params.render_outliers)
    {
        if (!scene.outlier_point_cloud_cuda)
        {
            scene.BuildOutlierCloud(params.outlier_count);
        }
    }


    auto add_displacement_tensor_info_to_point_cloud = [&](int b_id)
    {
        // a dummy [1] tensor is used when no displacements are optimized
        if (scene.dynamic_refinement_t.sizes().size() <= 1) return;

        {
            auto displacement = scene.dynamic_refinement_t.slice(0, b_id, b_id + 1).squeeze(0);
            auto d2           = torch::cat({displacement, torch::empty_like(displacement.slice(1, 0, 1))}, 1);

            scene.point_cloud_cuda->t_position_displacement = d2;
        }
    };

    // std::cout << (int)info->images.front().camera_model_type << std::endl;
    // std::cout << info->images.front().crop_transform << std::endl;
    // std::cout << info->images.front().crop_rotation << std::endl;
    // std::cout << (info->scene->intrinsics->intrinsics.slice(0, 0, 1)) << std::endl;

    // only for blending
    //  tensors are shape [2,max_elems]
    std::vector<std::vector<torch::Tensor>> collection_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> per_point_data_buffer(num_batches);

    if (cache.render_mode == PointRendererCache::RenderMode::FUZZY_DT)
    {
        // std::cout << "FUZZY DT" << std::endl;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("DepthPrepass", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                add_displacement_tensor_info_to_point_cloud(b);
                if (params.render_points)
                {
                    cache.DepthPrepassMulti(b, scene.point_cloud_cuda, info->train);
                }
                if (params.render_outliers)
                {
                    cache.DepthPrepassMulti(b, scene.point_cloud_cuda, info->train);
                }
            }
        }

        {
            SAIGA_OPTIONAL_TIME_MEASURE("RenderForward", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                add_displacement_tensor_info_to_point_cloud(b);

                if (params.render_points)
                {
                    cache.RenderForwardMulti(b, scene.point_cloud_cuda, info->train);
                }
                if (params.render_outliers)
                {
                    cache.RenderForwardMulti(b, scene.point_cloud_cuda, info->train);
                }
            }
        }

        {
            SAIGA_OPTIONAL_TIME_MEASURE("CombineAndFill", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                cache.CombineAndFill(b, scene.texture->background_color);
            }
        }

        // dummy fill
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                collection_buffer[b].push_back(
                    torch::zeros({2, 1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)));

                per_point_data_buffer[b].push_back(
                    torch::zeros({1, 3}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)));
            }
        }
    }
    if (cache.render_mode == PointRendererCache::RenderMode::FULL_BLEND ||
        cache.render_mode == PointRendererCache::RenderMode::FUZZY_BLEND ||
        cache.render_mode == PointRendererCache::RenderMode::BILINEAR_BLEND)
    {
        {
            SAIGA_ASSERT(params.render_outliers == false);
            SAIGA_OPTIONAL_TIME_MEASURE("CountingPass", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                if (params.render_points)
                {
                    add_displacement_tensor_info_to_point_cloud(b);

                    if (cache.render_mode == PointRendererCache::RenderMode::BILINEAR_BLEND)
                    {
                        cache.CountingPrepassMultiBilinear(b, scene.point_cloud_cuda, info->train);
                    }
                    else
                    {
                        cache.CountingPrepassMulti(b, scene.point_cloud_cuda, info->train);
                    }
                }
            }
        }
        std::vector<std::vector<int>> max_elements(num_batches);

        {
            SAIGA_OPTIONAL_TIME_MEASURE("CreateAndAllocateBuffers", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                for (int i = 0; i < info->num_layers; ++i)
                {
                    // Allocate result tensor
                    auto& l = cache.layers_cuda[i];
                    {
                        auto countings = l.BatchViewCounting(b);

                        auto scanned_countings                    = l.BatchViewScannedCounting(b);
                        thrust::device_ptr<int> thrust_offset_ptr = thrust::device_pointer_cast(countings.data);
                        thrust::device_ptr<int> thrust_scanned_counts_ptr =
                            thrust::device_pointer_cast(scanned_countings.data);
                        thrust::exclusive_scan(thrust_offset_ptr, thrust_offset_ptr + l.size.x() * l.size.y(),
                                               thrust_scanned_counts_ptr);

                        CUDA_SYNC_CHECK_ERROR();
                    }
                    // add last count to sum (as it is exclusive for better indexing later)
                    int num_elements = l.scanned_counting[b][0][l.size.y() - 1][l.size.x() - 1].item<int>() +
                                       l.counting[b][0][l.size.y() - 1][l.size.x() - 1].item<int>();
                    max_elements[b].push_back(num_elements);

                    // create or update total buffer size, based on scanned last element.
                    // TODO make persistant, only update if it gets larger
                    collection_buffer[b].push_back(torch::zeros(
                        {2, num_elements}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)));

                    per_point_data_buffer[b].push_back(torch::zeros(
                        {num_elements, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)));
                }
            }
        }
#if 0
    std::cout << "Max Elements: ";
    for(auto v : max_elements)
    {
        for(auto elem : v)
        {
            std::cout << elem << "-";
        }
    }
    std::cout << std::endl;
    std::cout <<"before " <<  std::endl;
    for(auto v : collection_buffer)
        for (auto& elem : v)
            std::cout << TensorInfo(elem);
    std::cout << std::endl;
#endif

        {
            SAIGA_OPTIONAL_TIME_MEASURE("CollectionPass", info->timer_system);
            // fill each segment with data: depth | index
            for (int b = 0; b < num_batches; ++b)
            {
                if (params.render_points)
                {
                    add_displacement_tensor_info_to_point_cloud(b);

                    if (cache.render_mode == PointRendererCache::RenderMode::BILINEAR_BLEND)
                    {
                        cache.CollectMultiBilinear(b, scene.point_cloud_cuda, collection_buffer[b],
                                                   per_point_data_buffer[b], info->train);
                    }
                    else
                    {
                        cache.CollectMulti(b, scene.point_cloud_cuda, collection_buffer[b], per_point_data_buffer[b],
                                           info->train);
                    }
                }
            }
        }
#if 0
    std::cout <<"after " <<  std::endl;
    for(auto v : collection_buffer)
        for (auto& elem : v)
            std::cout << TensorInfo(elem);
    std::cout << std::endl;
#endif
        {
            SAIGA_OPTIONAL_TIME_MEASURE("SortingPass", info->timer_system);

            // sort for depth in each pixel front -> back
            for (int b = 0; b < num_batches; ++b)
            {
                if (params.render_points)
                {
                    cache.SortMulti(b, collection_buffer[b], info->train);
                }
            }
        }
    }
    if (cache.render_mode == PointRendererCache::RenderMode::FULL_BLEND)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("BlendingPass", info->timer_system);

        // blend pixels forward
        for (int b = 0; b < num_batches; ++b)
        {
            if (params.render_points)
            {
                add_displacement_tensor_info_to_point_cloud(b);

                cache.BlendMulti(b, scene.point_cloud_cuda, collection_buffer[b], per_point_data_buffer[b],
                                 scene.texture->background_color, info->train, info->params.use_environment_map);
            }
        }
    }
    else if (cache.render_mode == PointRendererCache::RenderMode::FUZZY_BLEND)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("BlendingPassFuzzy", info->timer_system);

        // blend pixels forward
        for (int b = 0; b < num_batches; ++b)
        {
            if (params.render_points)
            {
                add_displacement_tensor_info_to_point_cloud(b);

                cache.BlendMultiFuzzy(b, scene.point_cloud_cuda, collection_buffer[b], per_point_data_buffer[b],
                                      scene.texture->background_color, info->train, info->params.use_environment_map);
            }
        }
    }
    else if (cache.render_mode == PointRendererCache::RenderMode::BILINEAR_BLEND)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("BlendingPassBilinear", info->timer_system);

        // blend pixels forward
        for (int b = 0; b < num_batches; ++b)
        {
            if (params.render_points)
            {
                add_displacement_tensor_info_to_point_cloud(b);

                cache.BlendMultiBilinear(b, scene.point_cloud_cuda, collection_buffer[b], per_point_data_buffer[b],
                                         scene.texture->background_color, info->train,
                                         info->params.use_environment_map);
            }
        }
    }

#if 0
    {
        SAIGA_OPTIONAL_TIME_MEASURE("DebugShowCounts", info->timer_system);

        if (params.render_points)
        {

            for (int b = 0; b < num_batches; ++b)
            {
                for (int i = 0; i < info->num_layers; ++i)
                {
                    // Allocate result tensor
                    auto& l = cache.layers_cuda[i];
                    int bx  = iDivUp(l.size.x(), 16);
                    int by  = iDivUp(l.size.y(), 16);
                    SAIGA_ASSERT(bx > 0 && by > 0);
                    auto in_out_neural_image = cache.output_forward[i][b];

                    //auto countings = l.BatchViewCounting(b);
                    //::DebugCountingsToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(countings, in_out_neural_image,
                    //                                                          1);

                    auto scanned_countings                    = l.BatchViewScannedCounting(b);
                    ::DebugCountingsToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(scanned_countings, in_out_neural_image,
                                                                                  1000);
                }
            }
        }

    }
#endif


    if (info->params.output_background_mask)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            cache.CreateMask(b, info->params.output_background_mask_value);
        }
    }

    if (info->params.debug_weight_color && info->params.num_texture_channels == 4)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                // Allocate result tensor
                auto& l = cache.layers_cuda[i];
                int bx  = iDivUp(l.size.x(), 16);
                int by  = iDivUp(l.size.y(), 16);
                SAIGA_ASSERT(bx > 0 && by > 0);
                auto in_out_neural_image = cache.output_forward[i][b];

                auto weights = l.BatchViewWeights(b);
                ::DebugWeightToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(weights, in_out_neural_image,
                                                                           info->params.debug_max_weight);
            }
        }
    }

    if (info->params.debug_depth_color && info->params.num_texture_channels == 4)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                // Allocate result tensor
                auto& l = cache.layers_cuda[i];
                int bx  = iDivUp(l.size.x(), 16);
                int by  = iDivUp(l.size.y(), 16);
                SAIGA_ASSERT(bx > 0 && by > 0);
                auto in_out_neural_image = cache.output_forward[i][b];

                auto depths = l.BatchViewDepth(b);
                ::DebugDepthToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(depths, in_out_neural_image,
                                                                          info->params.debug_max_weight);
            }
        }
    }

    if (info->params.debug_print_num_rendered_points)
    {
        double weight_sum = 0;
        for (int i = 0; i < info->num_layers; ++i)
        {
            // Allocate result tensor
            auto& l = cache.layers_cuda[i];
            weight_sum += l.weight.sum().item().toFloat();
        }
        std::cout << "# Rasterized Points = " << (int)weight_sum << std::endl;
    }

    if (ctx)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Save in Graph", info->timer_system);
        std::vector<torch::Tensor> save_variables;
        for (auto l : cache.layers_cuda)
        {
            save_variables.push_back(l.depth);
            save_variables.push_back(l.weight);
            save_variables.push_back(l.scanned_counting);
            save_variables.push_back(l.per_image_atomic_counters);
        }
        save_variables.insert(save_variables.end(), cache.output_forward.begin(), cache.output_forward.end());
        for (auto v : collection_buffer)
        {
            for (auto elem : v)
            {
                save_variables.push_back(elem);
            }
        }

        for (auto vx : per_point_data_buffer)
        {
            for (auto elemx : vx)
            {
                save_variables.push_back(elemx);
            }
        }


        save_variables.push_back(cache.dropout_points);
        ctx->save_for_backward(save_variables);
        CUDA_SYNC_CHECK_ERROR();
    }

    if (info->params.add_depth_to_network)
    {
        cache.output_forward.insert(cache.output_forward.end(), cache.output_forward_depthbuffer.begin(),
                                    cache.output_forward_depthbuffer.end());
    }

    // cudaDeviceSynchronize();
    return {std::move(cache.output_forward), std::move(cache.output_forward_blend)};
}


template <typename T, int N>
__global__ void NormalizeGradient(Vector<T, N>* tangent, float* tangent_count, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vector<T, N> t = tangent[tid];
    float c        = tangent_count[tid];

    if (c > 0)
    {
        // if (N == 6)
        //     for (int i = 0; i < 6; ++i) printf("++%f++ ", float(t(i)));
        tangent[tid] = t / c;
        // tangent[tid] = t / T(c);
        // if (N == 6)
        //    for (int i = 0; i < 6; ++i) printf("##%f## ", float(tangent[tid](i)));
    }
}

template <typename T, int N>
__global__ void NormalizeGradientDevTensor(StaticDeviceTensor<T, 2> tangent, StaticDeviceTensor<float, 1> tangent_count,
                                           int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Vector<T, N> t = tangent[tid];
    float c = tangent_count(tid);

    if (c > 0)
    {
        for (int i = 0; i < N; ++i)
        {
            tangent(tid, i) = tangent(tid, i) / c;
        }
    }
}

torch::autograd::variable_list BlendPointCloudBackward(torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info,
                                                       torch::autograd::variable_list _image_gradients)
{
    SAIGA_ASSERT(info->cache);
    for (auto& ig : _image_gradients)
    {
        SAIGA_ASSERT(ig.dtype() == torch::kFloat32);
    }

    // int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    // PointRendererCache cache;
    PointRendererCache& cache = *info->cache;

    int num_batches = cache.num_batches;

    auto add_displacement_tensor_info_to_point_cloud = [&](int b_id)
    {
        // a dummy [1] tensor is used when no displacements are optimized
        if (scene.dynamic_refinement_t.sizes().size() <= 1) return;

        {
            auto displacement = scene.dynamic_refinement_t.slice(0, b_id, b_id + 1).squeeze(0);
            auto d2           = torch::cat({displacement, torch::empty_like(displacement.slice(1, 0, 1))}, 1);
            scene.point_cloud_cuda->t_position_displacement = d2;
        }
    };

    /*
     *  These buffers are large buffers including space for exactly all pixels collected.
     *  Accessing can be done with the scanned countings list.
     *  there exists one for each batch and layer (i.e. 4 batches, 4 layers = [4][4])
     *  gradient_sum_back_buffer is an intermediate buffer for the Jacobians, with num_tex_parameters + 1 channels
     */
    std::vector<std::vector<torch::Tensor>> collection_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> per_point_data_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> gradient_sum_back_buffer(num_batches);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Prepare Backward", info->timer_system);
        cache.Build(info, false);

        // The first [num_layers] gradients are the actual neural image gradients. After that we get the gradients
        // of the mask which does not help us much
        cache.image_gradients =
            std::vector<torch::Tensor>(_image_gradients.begin(), _image_gradients.begin() + info->num_layers);

        auto save_variables = ctx->get_saved_variables();
        int offset_v        = 4;
        SAIGA_ASSERT(save_variables.size() ==
                     info->num_layers * (offset_v + 1) + 1 + 2 * info->num_layers * num_batches);
        cache.output_forward.resize(info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            cache.layers_cuda[i].depth                     = save_variables[i * offset_v + 0];
            cache.layers_cuda[i].weight                    = save_variables[i * offset_v + 1];
            cache.layers_cuda[i].scanned_counting          = save_variables[i * offset_v + 2];
            cache.layers_cuda[i].per_image_atomic_counters = save_variables[i * offset_v + 3];
            cache.output_forward[i]                        = save_variables[info->num_layers * offset_v + i];
        }
        int start_collb = info->num_layers * offset_v + info->num_layers;
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                collection_buffer[b].push_back(save_variables[start_collb + b * info->num_layers + i]);
            }
        }
        int start_data = info->num_layers * offset_v + info->num_layers + num_batches * info->num_layers;
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                per_point_data_buffer[b].push_back(save_variables[start_data + b * info->num_layers + i]);
                // intermediate summation buffer, last element is for alpha_dest storing
                gradient_sum_back_buffer[b].push_back(
                    torch::zeros({params.num_texture_channels + 1, per_point_data_buffer[b].back().size(0)},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)));
            }
        }
        cache.dropout_points = save_variables.back();

        SAIGA_ASSERT(cache.image_gradients.size() == info->num_layers);

        cache.PushParametersBackward();
    }

    if (cache.render_mode == PointRendererCache::RenderMode::FUZZY_DT)
    {
        {
            SAIGA_OPTIONAL_TIME_MEASURE("CombineAndFillBackward", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                cache.CombineAndFillBackward(b, scene.texture->background_color, cache.image_gradients);
            }
        }

        {
            SAIGA_OPTIONAL_TIME_MEASURE("RenderBackward", info->timer_system);
            for (int b = 0; b < num_batches; ++b)
            {
                add_displacement_tensor_info_to_point_cloud(b);
                cache.RenderBackward(b, scene.point_cloud_cuda);
            }
        }
    }
    else if (cache.render_mode == PointRendererCache::RenderMode::FULL_BLEND)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("BlendBackwards", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            add_displacement_tensor_info_to_point_cloud(b);
            cache.BlendBackwards(b, scene.point_cloud_cuda, collection_buffer[b], scene.texture->background_color,
                                 gradient_sum_back_buffer[b], info->params.use_environment_map);
        }
    }
    else if (cache.render_mode == PointRendererCache::RenderMode::FUZZY_BLEND)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("BlendBackwardsFuzzy", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            add_displacement_tensor_info_to_point_cloud(b);
            cache.BlendBackwardsFuzzy(b, scene.point_cloud_cuda, collection_buffer[b], scene.texture->background_color,
                                      gradient_sum_back_buffer[b], info->params.use_environment_map);
        }
    }
    else if (cache.render_mode == PointRendererCache::RenderMode::BILINEAR_BLEND)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("BlendBackwardsBilinear", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            add_displacement_tensor_info_to_point_cloud(b);
            cache.BlendBackwardsBilinear(b, scene.point_cloud_cuda, collection_buffer[b], per_point_data_buffer[b],
                                         scene.texture->background_color, gradient_sum_back_buffer[b],
                                         info->params.use_environment_map);
        }
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Post Process Gradient", info->timer_system);
        if (cache.output_gradient_pose_tangent.defined())
        {
            // std::cout << "POSE NORMALIZATION" << TensorInfo(cache.output_gradient_pose_tangent)
            //           << TensorInfo(cache.output_gradient_pose_tangent_count) << std::endl
            //           << std::endl;
            //  Average pose gradient over all measurements
            int n = cache.output_gradient_pose_tangent.size(0);
            int c = iDivUp(n, 128);
            // NormalizeGradient<double, 6><<<c, 128>>>((Vec6*)cache.output_gradient_pose_tangent.data_ptr<double>(),
            //                                          cache.output_gradient_pose_tangent_count.data_ptr<float>(), n);

            NormalizeGradientDevTensor<double, 6>
                <<<c, 128>>>(cache.output_gradient_pose_tangent, cache.output_gradient_pose_tangent_count, n);
            CUDA_SYNC_CHECK_ERROR();

            // std::cout << std::endl
            //           << "END POSE NORMALIZATION" << TensorInfo(cache.output_gradient_pose_tangent)
            //           << TensorInfo(cache.output_gradient_pose_tangent_count) << std::endl;
        }

        if (cache.output_gradient_points.defined())
        {
            // Average point gradient over all measurements
            int n = cache.output_gradient_points.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 3><<<c, 128>>>((vec3*)cache.output_gradient_points.data_ptr<float>(),
                                                    cache.output_gradient_point_count.data_ptr<float>(), n);
        }
        if (cache.output_gradient_dynamic_points.defined())
        {
            for (int b = 0; b < num_batches; ++b)
            {
                auto tensor_to_normalize = cache.output_gradient_dynamic_points.slice(0, b, b + 1).squeeze();
                auto count_tensor        = cache.output_gradient_dynamic_point_count.slice(0, b, b + 1).squeeze();
                int n                    = tensor_to_normalize.size(0);
                int c                    = iDivUp(n, 128);
                // NormalizeGradient<double, 6><<<c,
                // 128>>>((Vec6*)cache.output_gradient_pose_tangent.data_ptr<double>(),
                //                                          cache.output_gradient_pose_tangent_count.data_ptr<float>(),
                //                                          n);

                NormalizeGradientDevTensor<float, 3><<<c, 128>>>(tensor_to_normalize, count_tensor, n);
                CUDA_SYNC_CHECK_ERROR();
            }
        }

        if (cache.output_gradient_intrinsics.defined())
        {
            // Average intrinsics/distortion gradient over all measurements
            int n = cache.output_gradient_intrinsics.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 13>
                <<<c, 128>>>((Vector<float, 13>*)cache.output_gradient_intrinsics.data_ptr<float>(),
                             cache.output_gradient_intrinsics_count.data_ptr<float>(), n);
        }
    }
    CUDA_SYNC_CHECK_ERROR();

    // gradients for displacement field are equal to point gradients for that batch, as:
    //  point_pos = original_point_pos + displacements
    //  thus:
    //  d_point_pos / d_displacements = 1
    //  d_point_pos / d_original_point_pos = 1

    // std::cout << TensorInfo(cache.output_gradient_dynamic_points) << std::endl;

    return {std::move(cache.output_gradient_texture),       std::move(cache.output_gradient_background),
            std::move(cache.output_gradient_points),        std::move(cache.output_gradient_pose_tangent),
            std::move(cache.output_gradient_intrinsics),    std::move(cache.output_gradient_confidence),
            std::move(cache.output_gradient_dynamic_points)};
}

__global__ void ApplyTangent(Vec6* tangent, Sophus::SE3d* pose, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vec6 t = tangent[tid];
    auto p = pose[tid];
#ifdef _WIN32
    Sophus::SE3d p2(Sophus::se3_expd(t) * p);
    for (int i = 0; i < 7; ++i) pose[tid].data()[i] = p2.data()[i];
#else
    p         = Sophus::se3_expd(t) * p;
    pose[tid] = p;
#endif

    tangent[tid] = Vec6::Zero();
}

void ApplyTangentToPose(torch::Tensor tangent, torch::Tensor pose)
{
    SAIGA_ASSERT(tangent.is_contiguous() && pose.is_contiguous());
    int n = tangent.size(0);
    int c = iDivUp(n, 128);
    ApplyTangent<<<c, 128>>>((Vec6*)tangent.data_ptr<double>(), (Sophus::SE3d*)pose.data_ptr<double>(), n);
}