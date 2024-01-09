/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// functions for backwards rendering
// #include "saiga/colorize.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointBlending.h"
#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"

#include "cooperative_groups.h"
#include <curand_kernel.h>

__device__ __constant__ DeviceRenderParams d_render_params;
__device__ __constant__ DeviceTexture d_texture;
__device__ __constant__ DeviceForwardParams d_forward_params;
__device__ __constant__ DeviceBackwardParams d_backward_params;
__device__ __constant__ DeviceAlphaCompositionParams d_alpha_comp_params_bw;

template <typename IndexType>
__global__ void CombineAndFillBackward(StaticDeviceTensor<float, 3> image_gradient, ImageView<float> weight,
                                       float* out_background_gradient)
{
    int gx        = blockIdx.x * blockDim.x + threadIdx.x;
    int gy        = blockIdx.y * blockDim.y + threadIdx.y;
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    bool in_image = gx < weight.width && gy < weight.height;
    if (!in_image) return;

    gx = min(gx, weight.width - 1);
    gy = min(gy, weight.height - 1);

    __shared__ float bg_grad[32];

    int num_channels = image_gradient.sizes[0];

    if (local_tid < num_channels)
    {
        bg_grad[local_tid] = 0;
    }
    __syncthreads();


    auto w = weight(gy, gx);

    float factor = (w == 0 & in_image);

    // if (w == 0)
    {
        for (int ci = 0; ci < num_channels; ++ci)
        {
            float g = factor * image_gradient(ci, gy, gx);

            g = CUDA::warpReduceSum<float>(g);
            if (local_tid % 32 == 0)
            {
                atomicAdd(&bg_grad[ci], g);
            }
        }
    }

    __syncthreads();

    if (local_tid < num_channels)
    {
        atomicAdd(&out_background_gradient[local_tid], bg_grad[local_tid]);
    }
}



__inline__ __device__ thrust::tuple<vec2, float, float> ProjPoint(vec3 position, vec3 normal, float drop_out_radius,
                                                                  ReducedImageInfo& cam, bool check_normal)
{
    vec2 image_p_a;
    vec2 ip;
    float z;
    float radius_pixels;
    Sophus::SE3f V = d_render_params.Pose(cam.image_index);

    if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
        auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
        thrust::tie(image_p_a, z) =
            ProjectPointPinhole(position, normal, V, K, distortion, check_normal, d_render_params.dist_cutoff);
        radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;
        ip            = cam.crop_transform.normalizedToImage(image_p_a);
    }
    else if (cam.camera_model_type == CameraModel::OCAM)
    {
        auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
        thrust::tie(image_p_a, z) =
            ProjectPointOcam(position, normal, V, aff, poly, check_normal, d_render_params.dist_cutoff);
        radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
        ip            = cam.crop_transform.normalizedToImage(image_p_a);
    }
    else if (cam.camera_model_type == CameraModel::SPHERICAL)
    {
        thrust::tie(image_p_a, z) = ProjectPointSpherical(
            position, normal, V, vec2(d_render_params.depth[0].Image().w, d_render_params.depth[0].Image().h),
            check_normal, d_render_params.dist_cutoff);
        radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
        ip            = image_p_a;
        // ip = cam.crop_transform.normalizedToImage(image_p_a);
    }
    return {ip, z, radius_pixels};
}

__inline__ __device__ thrust::tuple<vec2, float, float> GetProjectedPoint(vec3 position, vec3 normal,
                                                                          float drop_out_radius, int point_id,
                                                                          ReducedImageInfo& cam)
{
    //  vec3 position;
    //  vec3 normal;
    //  float drop_out_radius;

    return ProjPoint(position, normal, drop_out_radius, cam, d_render_params.check_normal);
    // return {ip,z,radius_pixels};
}


__global__ void RenderBackward(DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int layer_,
                               int batch)
{
    int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= point_cloud.Size()) return;
    bool drop_out = dropout_p[point_id] == 1;

    vec2 ip;
    float z;
    float radius_pixels;

    vec3 position;
    vec3 normal;
    float drop_out_radius;

    Sophus::SE3f V = d_render_params.Pose(cam.image_index);
    {
        thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);
        thrust::tie(ip, z, radius_pixels) = GetProjectedPoint(position, normal, drop_out_radius, point_id, cam);

        if (z == 0) return;
    }

    //   auto texture_index = point_cloud.GetIndex(point_id);
    float scale = 1;

    int texture_index = point_cloud.GetIndex(point_id);
    //    if(discard_point_for_confidence(texture_index, point_id, batch, true))
    //        return;

    ivec2 p_imgi_top_level = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

#pragma unroll
    for (int layer = 0; layer < max_layers; ++layer, scale *= 0.5f, radius_pixels *= 0.5f, ip *= 0.5f)
    {
        if (layer < d_render_params.num_layers)
        {
            // ip            = cam.crop_transform.scale(scale).normalizedToImage(image_p_a, nullptr, nullptr);
            // radius_pixels = scale * cam.base_K.fx * cam.crop_transform.fx * drop_out_radius / z;

            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                continue;
            }

            ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

            // Check in image
            if (!d_render_params.depth[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;

            auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, p_imgi(1), p_imgi(0)));
            float w              = *dst_pos_weight;

            float image_depth = d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0));

            // no rasterized point
            if (w == 0) continue;
            float iw = 1.f / w;

            if (z > image_depth * (d_render_params.depth_accept + 1))
            {
                continue;
            }

            if (!drop_out)
            {
                // This is currently necessary because somehow the results cam.crop_transform transformation gives
                // different results here compared to the forwrad function even though the inputs are the same.
                if (w == 0) continue;

                float iw = 1.f / w;
                CUDA_DEBUG_ASSERT(w > 0);

                {
                    // dont use point confidence or flowback
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        float g = iw * d_backward_params.in_gradient_image[layer](batch, ci, p_imgi.y(), p_imgi.x());
                        CUDA_KERNEL_ASSERT(isfinite(g));
                        // g = clamp(g * 100, -10, 10);
                        atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);
                    }
                }
            }
            bool point_grad =
                (d_render_params.ghost_gradients && drop_out) | (!d_render_params.ghost_gradients && !drop_out);
            if (point_grad && d_render_params.depth[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
            {
                // We have the following pixel constellation were p is the pixel of the current point and px0, px1,...
                // are the left, right, bottom and top neighbors.
                //
                //        py1
                //         |
                //   px0 - p - px1
                //         |
                //        py0
                //
                // The output neural image of the render operation is I.
                // The intensity of a pixel in I is for example I(px1). The texture of a point is T(p). The background
                // color is B(p). We now compute the gradient of p w.r.t. I.
                //
                // If the point moves from p to px1 and I(px1) was previously the background color
                // then I(px1) is now colored by the texture of p. The new value NV of px1 is then:
                //   NV(px1) = T(p)
                //
                // The change of I at px1 is then:
                // Motion p -> px1 (positive X):
                //   dI(x)/dp|x=px1 = NV(px1) - I(px1)
                //
                // There is a special case that if I(px1) is already colored by one or more points which have a similar
                // z-value then the point is blended into the neighbors instead of overriding them. This change is then
                // defined using the weight at that pixel W(p).
                //
                // New value if p -> px1:
                //   NV(px1) = W(px1)/(W(px1)+1)*I(px1)+1/(W(px1)+1)*T(p)
                // The gradient is therefore:
                //   dI(x)/dp|x=px1 = NV(px1) - I(px1)
                //

                ivec2 px0 = p_imgi + ivec2(-1, 0);
                ivec2 px1 = p_imgi + ivec2(1, 0);
                ivec2 py0 = p_imgi + ivec2(0, -1);
                ivec2 py1 = p_imgi + ivec2(0, 1);

                float iw = 1.f / (w + 1);

                float dR_dpx = 0;
                float dR_dpy = 0;

                auto sample_grad = [&](int ci, ivec2 p) -> float
                { return d_backward_params.in_gradient_image[layer](batch, ci, p.y(), p.x()); };
                auto sample_forward = [&](int ci, ivec2 p) -> float
                { return d_forward_params.neural_out[layer](batch, ci, p.y(), p.x()); };
                auto sample_tex = [&](int ci, int uv) -> float { return d_texture.in_texture(ci, uv); };

                auto compute_dR_dp_at_x = [&](ivec2 x, int ci, float W_x, float D_x, float T_p) -> float
                {
                    auto I_x = sample_forward(ci, x);
                    auto G_x = sample_grad(ci, x);

                    if (d_render_params.test_backward_mode == 4)
                    {
                        float dI_dp_at_x;
                        if (W_x == 0 || z * (d_render_params.depth_accept + 1) < D_x)
                        {
                            // Full override
                            dI_dp_at_x = T_p - I_x;
                        }
                        else if (z > D_x * (d_render_params.depth_accept + 1))
                        {
                            // Discard
                            dI_dp_at_x = 0;
                            //                             dI_dp_at_x = T_p - I_x;
                        }
                        else
                        {
                            // Blend
                            dI_dp_at_x = (W_x / (W_x + 1) * I_x + 1 / (W_x + 1) * T_p - I_x);
                            //                            dI_dp_at_x = T_p - I_x;
                        }

                        float dR_dp_at_x = dI_dp_at_x * G_x;
                        return dR_dp_at_x;
                    }
                    else
                    {
                        float dI_dp_at_x = T_p - I_x;

                        float dR_dp_at_x = dI_dp_at_x * G_x;
                        return dR_dp_at_x;
                    }
                };

                float W_px0 = d_render_params.weight[layer](batch, 0, px0(1), px0(0));
                float W_px1 = d_render_params.weight[layer](batch, 0, px1(1), px1(0));
                float W_py0 = d_render_params.weight[layer](batch, 0, py0(1), py0(0));
                float W_py1 = d_render_params.weight[layer](batch, 0, py1(1), py1(0));

                float D_px0 = d_render_params.depth[layer](batch, 0, px0(1), px0(0));
                float D_px1 = d_render_params.depth[layer](batch, 0, px1(1), px1(0));
                float D_py0 = d_render_params.depth[layer](batch, 0, py0(1), py0(0));
                float D_py1 = d_render_params.depth[layer](batch, 0, py1(1), py1(0));

                int texture_channels = d_render_params.num_texture_channels;

#pragma unroll
                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto g = sample_grad(ci, p_imgi);

                    auto T_p = sample_tex(ci, texture_index);

                    // The spatial derivatives at the neighboring points.
                    float dI_dp_at_px0 = 0;  //-(T_p - I_px0);
                    float dI_dp_at_px1 = 0;  // T_p - I_px1;
                    float dI_dp_at_py0 = 0;  //-(T_p - I_py0);
                    float dI_dp_at_py1 = 0;  // T_p - I_py1;

                    dI_dp_at_px0 = -compute_dR_dp_at_x(px0, ci, W_px0, D_px0, T_p);
                    dI_dp_at_px1 = compute_dR_dp_at_x(px1, ci, W_px1, D_px1, T_p);
                    dI_dp_at_py0 = -compute_dR_dp_at_x(py0, ci, W_py0, D_py0, T_p);
                    dI_dp_at_py1 = compute_dR_dp_at_x(py1, ci, W_py1, D_py1, T_p);

                    // Average between forward and backward diff. to get symmetric central diff.
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);
                }

                vec2 dR_dp = vec2(dR_dpx, dR_dpy) / float(texture_channels);
                //   CUDA_KERNEL_ASSERT((dR_dp.x() == 0 && dR_dp.y() == 0));

                float grad_scale    = 1.f;
                auto cam2           = cam;
                cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


                if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
                {
                    auto [K, distortion]               = d_render_params.PinholeIntrinsics(cam.camera_index);
                    auto [g_point, g_pose, g_k, g_dis] = ProjectPointPinholeBackward(
                        position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                        d_render_params.dist_cutoff, cam2.crop_rotation);

                    if (d_backward_params.out_gradient_points)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                    }
                    if (d_backward_params.out_gradient_dynamic_points.data)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                      g_point(k));  // * 0.1);
                        }
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                    }
                    if (d_backward_params.out_gradient_pose)
                    {
                        // Extrinsics
                        for (int k = 0; k < g_pose.rows(); ++k)
                        {
                            //         printf("%f", g_pose(k));
                            atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f);
                    }

                    if (d_backward_params.out_gradient_intrinsics_count)
                    {
                        float k_factor = d_render_params.K_gradient_factor;
                        // Intrinsics
                        // g_k(2) *= 0.5;
                        // g_k(3) *= 0.5;

                        // sheer
                        g_k(4) *= 0.1;
                        g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                        for (int k = 0; k < 5; ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k),
                                      k_factor * g_k(k));
                        }

                        float distortion_factor = d_render_params.distortion_gradient_factor;

                        // k3
                        g_dis(2) *= 0.25;

                        // k4 - 6
                        g_dis(3) *= 0.1;
                        g_dis(4) *= 0.1;
                        g_dis(5) *= 0.1;

                        // tangential distortion
                        g_dis(6) *= 0.1;
                        g_dis(7) *= 0.1;
                        for (int k = 0; k < 8; ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                      distortion_factor * g_dis(k));
                        }
                        // Note we add a value less than 1 to increase float precision
                        float factor = 1.f / 1024.f;
                        atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                    }
                }
                else if (cam.camera_model_type == CameraModel::OCAM)
                {
                    auto [aff, poly]                 = d_render_params.OcamIntrinsics(cam.camera_index);
                    auto [g_point, g_pose, g_affine] = ProjectPointOcamBackward(
                        position, normal, dR_dp, V, cam2.crop_transform, aff, poly, d_render_params.check_normal,
                        d_render_params.dist_cutoff, cam2.crop_rotation);

                    if (d_backward_params.out_gradient_points)
                    {
                        // Points
                        for (int k = 0; k < g_point.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                    }
                    if (d_backward_params.out_gradient_dynamic_points.data)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                      g_point(k));  // * 0.1);
                        }
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                    }

                    if (d_backward_params.out_gradient_pose)
                    {
                        // Extrinsics
                        for (int k = 0; k < g_pose.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                    }

                    if (d_backward_params.out_gradient_intrinsics_count)
                    {
                        // Intrinsics
                        for (int k = 0; k < 5; ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                    }
                }
                else if (cam.camera_model_type == CameraModel::SPHERICAL)
                {
                    CUDA_KERNEL_ASSERT(false);
                }
            }
        }
    }
}

void PointRendererCache::PushParametersBackward()
{
    SAIGA_OPTIONAL_TIME_MEASURE("Param Upload", info->timer_system);

    {
        static DeviceForwardParams dfp;
        for (int i = 0; i < info->num_layers; ++i)
        {
            dfp.neural_out[i] = output_forward[i];
        }
        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_forward_params, &dfp, sizeof(dfp)));
    }
    {
        DeviceBackwardParams dbp = PrepareDeviceBackwardParams();
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_backward_params, &dbp, sizeof(dbp)));
    }
    {
        static DeviceRenderParams drp;
        drp = PrepareDeviceRenderParams();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_render_params, &drp, sizeof(drp)));
        CUDA_SYNC_CHECK_ERROR();
    }
    {
        static DeviceTexture d_tex;
        d_tex = PrepareDeviceTexture();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_texture, &d_tex, sizeof(d_tex)));
        CUDA_SYNC_CHECK_ERROR();
    }
}


void PointRendererCache::CombineAndFillBackward(int batch, torch::Tensor background_color,
                                                std::vector<torch::Tensor> image_gradient)
{
    for (int i = 0; i < info->num_layers; ++i)
    {
        //       std::cout << info->timer_system- << std::endl;
        //      SAIGA_OPTIONAL_TIME_MEASURE("CombineAndFillBackward " + std::to_string(i), info->timer_system);
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);

        auto out_info            = output_gradient_background.data_ptr<float>();
        auto image_gradient_info = image_gradient[i][batch];

        //   std::cout << "in: " << image_gradient_info.sizes() << std::endl;
        //   std::cout << "out: " <<output_gradient_background.sizes() << std::endl;

        ::CombineAndFillBackward<unsigned int>
            <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(image_gradient_info, l.BatchViewWeights(batch), out_info);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::RenderBackward(int batch, NeuralPointCloudCuda point_cloud)
{
    SAIGA_ASSERT(point_cloud);
    {
        int image_batch_id = batch;

        auto cam = info->images[image_batch_id];

        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id;
        int c          = iDivUp(point_cloud->Size(), default_block_size);
        ::RenderBackward<<<c, default_block_size>>>(point_cloud, dropout, cam, 0, batch);
    }
    CUDA_SYNC_CHECK_ERROR();
}


void PointRendererCache::UploadCollectionBuffersBackwards(std::vector<torch::Tensor> buffers,
                                                          std::vector<torch::Tensor> data_buffer,
                                                          std::vector<torch::Tensor> grad_sum_back_buffers,
                                                          int batch_num)
{
    static DeviceAlphaCompositionParams dacp;
    // buffer.size == layers used
    for (int i = 0; i < buffers.size(); ++i)
    {
        dacp.collections[i]            = buffers[i];
        dacp.gradient_sum_backwards[i] = grad_sum_back_buffers[i];
        if (!data_buffer.empty()) dacp.per_point_data[i] = data_buffer[i];
    }
    for (int i = 0; i < info->num_layers; ++i)
    {
        dacp.scanned_countings[i] = layers_cuda[i].scanned_counting[batch_num];
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_alpha_comp_params_bw, &dacp, sizeof(dacp)));
}


template <int num_descriptors>
__global__ void BlendBackwardsFuzzy(DevicePointCloud point_cloud, float* background_color,
                                    float* out_background_gradient, int batch, int layer, ReducedImageInfo cam,
                                    bool need_point_gradients, bool use_environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
        gy >= d_backward_params.in_gradient_image[layer].size(2))
        return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);

    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);
    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    // helper functions
    auto sample_depth = [&](int buffer_pos) -> float
    {
        int depth_i = d_alpha_comp_params_bw.collections[layer](0, buffer_pos);
        float depth = reinterpret_cast<float*>(&depth_i)[0];
        return depth;
    };
    auto sample_point_id = [&](int buffer_pos) -> int
    {
        int point_id_b = d_alpha_comp_params_bw.collections[layer](1, buffer_pos);
        return point_cloud.GetIndex(point_id_b);
    };

    {
        // fuzzy
        float accumulation_last_min_d = 0.f;
        float accumulation_desc[num_descriptors];
        float accumulation_conf_val = 0.f;
        int accumulation_num_elems  = 0;
        int accum_start_index       = 0;
        float alpha_dest            = 1.f;

        auto accumulateCollectedBackwards = [&](bool is_last)
        {
            float inv_weight = 1.f / float(accumulation_num_elems);

            float confidence_val = accumulation_conf_val * inv_weight;

            float colors[num_descriptors];
            float grad_in[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                colors[ci]  = accumulation_desc[ci] * inv_weight;
                grad_in[ci] = d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
            }

            // compute gradients
            float g_alpha = 0;
            float g_colors[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                g_colors[ci] = alpha_dest * confidence_val * grad_in[ci];
                g_alpha += colors[ci] * grad_in[ci];
            }
            g_alpha *= alpha_dest;

            // add gradients to all points
            for (int t = accum_start_index; t < accum_start_index + accumulation_num_elems; ++t)
            {
                bool is_foreground = true;
                if (t == size_of_chunk) is_foreground = false;
                int texture_index = 0;

                if (is_foreground) texture_index = sample_point_id(offset_in_buffer + t);

                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    float* grad_write_address = is_foreground
                                                    ? &d_backward_params.out_gradient_texture(ci, texture_index)
                                                    : &out_background_gradient[ci];
                    atomicAdd(grad_write_address, g_colors[ci] * inv_weight);
                }
                if (is_foreground)
                {
                    atomicAdd(&d_backward_params.out_gradient_confidence(0, texture_index), g_alpha * inv_weight);

                    // save averaged confidence to be used with updates
                    d_alpha_comp_params_bw.gradient_sum_backwards[layer](d_render_params.num_texture_channels,
                                                                         offset_in_buffer + t) = confidence_val;
                }
                //  return;
            }

            // propergate to all previous (before fuzzy clunk) elements
            for (int j = 0; j < accum_start_index; ++j)
            {
                int full_buffer_pos_iter = offset_in_buffer + j;
                int texture_index_iter   = sample_point_id(full_buffer_pos_iter);
                // float confidence_val_iter = d_texture.points_confidence_value(0, texture_index_iter);
                float confidence_val_iter = d_alpha_comp_params_bw.gradient_sum_backwards[layer](
                    d_render_params.num_texture_channels, full_buffer_pos_iter);

                float g_iter = 0;
                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    const float epsilon = 1e-9;
                    float dem           = 1 / (1 - confidence_val_iter + epsilon);
                    float g_alpha_iter =
                        (grad_in[ci] * colors[ci] * alpha_dest * confidence_val / (1 - confidence_val_iter + epsilon));
                    g_iter -= g_alpha_iter;
                    // g += -grad_in[ci] * color_iter * alpha_dest * confidence_val * dem;
                }
                float* grad_address_iter = &d_backward_params.out_gradient_confidence(0, texture_index_iter);
                atomicAdd(grad_address_iter, g_iter);
            }
            alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);
        };  // end lambda


        // blend background color if no env map
        // int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;
        int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;

        // blend all together
        for (int i = 0; i < list_size; ++i)
        {
            bool is_foreground = true;
            if (i == size_of_chunk) is_foreground = false;
            int full_buffer_pos = offset_in_buffer + i;
            float d             = is_foreground ? sample_depth(full_buffer_pos) : MAX_DEPTH_CONST;

            if (!(d - accumulation_last_min_d < d_render_params.depth_accept_blend))
            {
                if (i != 0) accumulateCollectedBackwards(false);
                {
                    // reset state
                    accumulation_conf_val  = 0.f;
                    accumulation_num_elems = 0;
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        accumulation_desc[ci] = 0.f;
                    }
                    accumulation_last_min_d = d;
                    accum_start_index       = i;
                }
            }

            // accumulate fuzzy
            {
                int texture_index = 0;
                if (is_foreground) texture_index = sample_point_id(full_buffer_pos);

                CUDA_KERNEL_ASSERT(texture_index >= 0);
                CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

                // if (accumulation_last_min_d == 0.f)
                //{
                //     accumulation_last_min_d = d;
                //     accum_start_index       = i;
                // }

                float confidence_val = 1.f;

                if (is_foreground) confidence_val = d_texture.points_confidence_value(0, texture_index);

                accumulation_conf_val += confidence_val;

                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    float color = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];
                    accumulation_desc[ci] += color;
                }
                ++accumulation_num_elems;
            }
        }
        // accumulate last elements
        if (accumulation_num_elems > 0)
        {
            accumulateCollectedBackwards(true);
        }
    }

    if (!need_point_gradients) return;

    // approximate gradient for points
    ivec2 p_imgi = ivec2(gx, gy);
    if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
    {
        float alpha_dest = 1.f;

        for (int i = 0; i < size_of_chunk; ++i)
        {
            float intensities_p_0_0_point_gradient[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci) intensities_p_0_0_point_gradient[ci] = 0.f;

            {
                // compute intensity without point at own position
                float alpha_dest_point_gradient = 1.f;
                int offset_in_buffer_b          = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);
                for (int j = 0; j < size_of_chunk; ++j)
                {
                    if (j != i)
                    {
                        int full_buffer_pos_b = offset_in_buffer_b + j;
                        int texture_index_b   = sample_point_id(full_buffer_pos_b);

                        float confidence_val = d_texture.points_confidence_value(0, texture_index_b);
                        for (int ci = 0; ci < num_descriptors; ++ci)
                        {
                            float color                          = d_texture.in_texture(ci, texture_index_b);
                            intensities_p_0_0_point_gradient[ci] = compute_blend(
                                alpha_dest_point_gradient, confidence_val, color, intensities_p_0_0_point_gradient[ci]);
                        }
                        alpha_dest_point_gradient = compute_new_alphadest(alpha_dest_point_gradient, confidence_val);
                    }
                }
                if (!use_environment_map)
                {
                    // background
                    for (int ci = 0; ci < num_descriptors; ++ci)
                    {
                        intensities_p_0_0_point_gradient[ci] =
                            compute_blend(alpha_dest, 1.f, background_color[ci], intensities_p_0_0_point_gradient[ci]);
                    }
                }
            }

            int full_buffer_pos = offset_in_buffer + i;
            int point_id        = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
            int texture_index   = point_cloud.GetIndex(point_id);

            float confidence_val = d_texture.points_confidence_value(0, texture_index);

            ivec2 px0 = p_imgi + ivec2(-1, 0);
            ivec2 px1 = p_imgi + ivec2(1, 0);
            ivec2 py0 = p_imgi + ivec2(0, -1);
            ivec2 py1 = p_imgi + ivec2(0, 1);

            auto sample_grad = [&](int ci, ivec2 p) -> float
            { return d_backward_params.in_gradient_image[layer](batch, ci, p.y(), p.x()); };
            auto sample_forward = [&](int ci, ivec2 p) -> float
            { return d_forward_params.neural_out[layer](batch, ci, p.y(), p.x()); };
            auto sample_tex = [&](int ci, int uv) -> float { return d_texture.in_texture(ci, uv); };

#if 0
            // compute neighboring pixel with current point inserted into the blend
            auto compute_blend_with_point_at_x = [&](ivec2 x)  //, vec4 result)
            {
                float point_depth = sample_depth(full_buffer_pos);

                int size_of_aux_chunk  = d_render_params.per_image_atomic_counters[layer](batch, 0, x.y(), x.x());
                int offset_in_buffer_c = d_alpha_comp_params_bw.scanned_countings[layer](0, x.y(), x.x());

                float intensities_aux_point[MAXCHANNELS];
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) intensities_aux_point[ci] = 0.f;
                float alpha_dest_aux_point = 1.f;
                bool inserted_point        = false;
                for (int j = 0; j < size_of_aux_chunk; ++j)
                {
                    int full_buffer_pos_c = offset_in_buffer_c + j;

                    float depth_aux = sample_depth(full_buffer_pos_c);

                    if (!inserted_point && point_depth < depth_aux)
                    {
                        // insert point
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val);
                        inserted_point       = true;
                        --j;
                    }
                    else
                    {
                        // compute blend with other points
                        int texture_index_c    = sample_point_id(full_buffer_pos_c);
                        float confidence_val_c = d_texture.points_confidence_value(0, texture_index_c);
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index_c);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val_c, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val_c);
                    }
                }
                if (!use_environment_map)
                {
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        intensities_aux_point[ci] =
                            compute_blend(alpha_dest_aux_point, 1.f, background_color[ci], intensities_aux_point[ci]);
                    }
                }
                vec4 result;
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    result[ci] = intensities_aux_point[ci];
                }
                return result;
            };
#endif
            int texture_channels = num_descriptors;  // d_render_params.num_texture_channels;

            float dR_dpx = 0;
            float dR_dpy = 0;

#if 0
            if (d_render_params.test_backward_mode == 4)
            {
                // float I_px0[MAXCHANNELS];
                // float I_px1[MAXCHANNELS];
                // float I_py0[MAXCHANNELS];
                // float I_py1[MAXCHANNELS];
                // vec4 I_px0 = vec4(0, 0, 0, 0);
                // vec4 I_px1 = vec4(0, 0, 0, 0);
                // vec4 I_py0 = vec4(0, 0, 0, 0);
                // vec4 I_py1 = vec4(0, 0, 0, 0);

                vec4 I_px0 = compute_blend_with_point_at_x(px0);  //, I_px0);
                vec4 I_px1 = compute_blend_with_point_at_x(px1);  //, I_px1);
                vec4 I_py0 = compute_blend_with_point_at_x(py0);  //, I_py0);
                vec4 I_py1 = compute_blend_with_point_at_x(py1);  //, I_py1);

                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    float dI_dp_at_px0 = -(intensities_p_0_0_point_gradient[ci] - I_px0[ci]) * G_px0;
                    float dI_dp_at_px1 = (intensities_p_0_0_point_gradient[ci] - I_px1[ci]) * G_px1;
                    float dI_dp_at_py0 = -(intensities_p_0_0_point_gradient[ci] - I_py0[ci]) * G_py0;
                    float dI_dp_at_py1 = (intensities_p_0_0_point_gradient[ci] - I_py1[ci]) * G_py1;
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  // * alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  // * alpha_dest;
                }
            }
            else
#endif
            {
#pragma unroll
                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto g = sample_grad(ci, p_imgi);

                    auto T_p = sample_tex(ci, texture_index);

                    auto I_px0 = sample_forward(ci, px0);
                    auto I_px1 = sample_forward(ci, px1);
                    auto I_py0 = sample_forward(ci, py0);
                    auto I_py1 = sample_forward(ci, py1);
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    // The spatial derivatives at the neighboring points.
                    float dI_dp_at_px0 = -(T_p - I_px0) * G_px0;
                    float dI_dp_at_px1 = (T_p - I_px1) * G_px1;
                    float dI_dp_at_py0 = -(T_p - I_py0) * G_py0;
                    float dI_dp_at_py1 = (T_p - I_py1) * G_py1;
                    // Average between forward and backward diff. to get symmetric central diff.
                    // multiply by alpha_dest for individual point contribution
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  //* alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  //* alpha_dest;
                }
            }
            vec2 dR_dp = vec2(dR_dpx, dR_dpy) / float(texture_channels);
            // dR_dp*=alpha_dest;

            alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);


            // adapt for multiresolution rendering
            float scale = 1 * powf(0.5f, float(layer));

            float grad_scale    = 1.f;
            auto cam2           = cam;
            cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            vec3 position;
            vec3 normal;
            float drop_out_radius;
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion]               = d_render_params.PinholeIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_k, g_dis] = ProjectPointPinholeBackward(
                    position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    float k_factor = d_render_params.K_gradient_factor;

                    g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), k_factor * g_k(k));
                    }

                    float distortion_factor = d_render_params.distortion_gradient_factor;

                    // k3
                    g_dis(2) *= 0.25;

                    // k4 - 6
                    g_dis(3) *= 0.1;
                    g_dis(4) *= 0.1;
                    g_dis(5) *= 0.1;

                    // tangential distortion
                    g_dis(6) *= 0.1;
                    g_dis(7) *= 0.1;
                    for (int k = 0; k < 8; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                  distortion_factor * g_dis(k));
                    }
                    // Note we add a value less than 1 to increase float precision
                    float factor = 1.f / 1024.f;
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]                 = d_render_params.OcamIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_affine] = ProjectPointOcamBackward(
                    position, normal, dR_dp, V, cam2.crop_transform, aff, poly, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    // Points
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    // Intrinsics
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                }
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                CUDA_KERNEL_ASSERT(false);
            }
        }
    }
}


template <int num_descriptors>
__global__ void BlendBackwards2(DevicePointCloud point_cloud, float* background_color, float* out_background_gradient,
                                int batch, int layer, ReducedImageInfo cam, bool need_point_gradients,
                                bool use_environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
        gy >= d_backward_params.in_gradient_image[layer].size(2))
        return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);

    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);
    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);


    // helper functions
    auto sample_depth = [&](int buffer_pos) -> float
    {
        int depth_i = d_alpha_comp_params_bw.collections[layer](0, buffer_pos);
        float depth = reinterpret_cast<float*>(&depth_i)[0];
        return depth;
    };
    auto sample_point_id = [&](int buffer_pos) -> int
    {
        int point_id_b = d_alpha_comp_params_bw.collections[layer](1, buffer_pos);
        return point_cloud.GetIndex(point_id_b);
    };



    {
        float alpha_dest = 1.f;

        // blend background color if no env map
        int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;
        // blend all together
        for (int i = 0; i < list_size; ++i)
        {
            bool is_foreground = true;
            if (i == size_of_chunk) is_foreground = false;

            int full_buffer_pos = offset_in_buffer + i;
            int texture_index   = 0;
            if (is_foreground) texture_index = sample_point_id(full_buffer_pos);

            CUDA_KERNEL_ASSERT(texture_index >= 0);
            CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

            float confidence_val = 1.f;

            if (is_foreground) confidence_val = d_texture.points_confidence_value(0, texture_index);
            float colors[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci)
                colors[ci] = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];


            float grad_in[num_descriptors];
            float g_alpha = 0;
            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                grad_in[ci]               = d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                float g_col               = alpha_dest * confidence_val * grad_in[ci];
                float* grad_write_address = is_foreground ? &d_backward_params.out_gradient_texture(ci, texture_index)
                                                          : &out_background_gradient[ci];
                atomicAdd(grad_write_address, g_col);

                g_alpha += colors[ci] * grad_in[ci];
            }
            g_alpha *= alpha_dest;
            atomicAdd(&d_backward_params.out_gradient_confidence(0, texture_index), g_alpha);


            for (int j = 0; j < i; ++j)
            {
                int full_buffer_pos_iter  = offset_in_buffer + j;
                int texture_index_iter    = sample_point_id(full_buffer_pos_iter);
                float confidence_val_iter = d_texture.points_confidence_value(0, texture_index_iter);

                float g_iter = 0;
                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    const float epsilon = 1e-9;
                    float dem           = 1 / (1 - confidence_val_iter + epsilon);
                    float g_alpha_iter =
                        (grad_in[ci] * colors[ci] * alpha_dest * confidence_val / (1 - confidence_val_iter + epsilon));
                    g_iter -= g_alpha_iter;
                    // g += -grad_in[ci] * color_iter * alpha_dest * confidence_val * dem;
                }
                float* grad_address_iter = &d_backward_params.out_gradient_confidence(0, texture_index_iter);
                atomicAdd(grad_address_iter, g_iter);
            }
            alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);
        }
    }

    if (!need_point_gradients) return;

    // approximate gradient for points
    ivec2 p_imgi = ivec2(gx, gy);
    if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
    {
        float alpha_dest = 1.f;

        for (int i = 0; i < size_of_chunk; ++i)
        {
            float intensities_p_0_0_point_gradient[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci) intensities_p_0_0_point_gradient[ci] = 0.f;

            {
                // compute intensity without point at own position
                float alpha_dest_point_gradient = 1.f;
                int offset_in_buffer_b          = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);
                for (int j = 0; j < size_of_chunk; ++j)
                {
                    if (j != i)
                    {
                        int full_buffer_pos_b = offset_in_buffer_b + j;
                        int texture_index_b   = sample_point_id(full_buffer_pos_b);

                        float confidence_val = d_texture.points_confidence_value(0, texture_index_b);
                        for (int ci = 0; ci < num_descriptors; ++ci)
                        {
                            float color                          = d_texture.in_texture(ci, texture_index_b);
                            intensities_p_0_0_point_gradient[ci] = compute_blend(
                                alpha_dest_point_gradient, confidence_val, color, intensities_p_0_0_point_gradient[ci]);
                        }
                        alpha_dest_point_gradient = compute_new_alphadest(alpha_dest_point_gradient, confidence_val);
                    }
                }
                if (!use_environment_map)
                {
                    // background
                    for (int ci = 0; ci < num_descriptors; ++ci)
                    {
                        intensities_p_0_0_point_gradient[ci] =
                            compute_blend(alpha_dest, 1.f, background_color[ci], intensities_p_0_0_point_gradient[ci]);
                    }
                }
            }

            int full_buffer_pos = offset_in_buffer + i;
            int point_id        = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
            int texture_index   = point_cloud.GetIndex(point_id);

            float confidence_val = d_texture.points_confidence_value(0, texture_index);

            ivec2 px0 = p_imgi + ivec2(-1, 0);
            ivec2 px1 = p_imgi + ivec2(1, 0);
            ivec2 py0 = p_imgi + ivec2(0, -1);
            ivec2 py1 = p_imgi + ivec2(0, 1);

            auto sample_grad = [&](int ci, ivec2 p) -> float
            { return d_backward_params.in_gradient_image[layer](batch, ci, p.y(), p.x()); };
            auto sample_forward = [&](int ci, ivec2 p) -> float
            { return d_forward_params.neural_out[layer](batch, ci, p.y(), p.x()); };
            auto sample_tex = [&](int ci, int uv) -> float { return d_texture.in_texture(ci, uv); };

#if 0
            // compute neighboring pixel with current point inserted into the blend
            auto compute_blend_with_point_at_x = [&](ivec2 x)  //, vec4 result)
            {
                float point_depth = sample_depth(full_buffer_pos);

                int size_of_aux_chunk  = d_render_params.per_image_atomic_counters[layer](batch, 0, x.y(), x.x());
                int offset_in_buffer_c = d_alpha_comp_params_bw.scanned_countings[layer](0, x.y(), x.x());

                float intensities_aux_point[MAXCHANNELS];
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) intensities_aux_point[ci] = 0.f;
                float alpha_dest_aux_point = 1.f;
                bool inserted_point        = false;
                for (int j = 0; j < size_of_aux_chunk; ++j)
                {
                    int full_buffer_pos_c = offset_in_buffer_c + j;

                    float depth_aux = sample_depth(full_buffer_pos_c);

                    if (!inserted_point && point_depth < depth_aux)
                    {
                        // insert point
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val);
                        inserted_point       = true;
                        --j;
                    }
                    else
                    {
                        // compute blend with other points
                        int texture_index_c    = sample_point_id(full_buffer_pos_c);
                        float confidence_val_c = d_texture.points_confidence_value(0, texture_index_c);
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index_c);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val_c, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val_c);
                    }
                }
                if (!use_environment_map)
                {
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        intensities_aux_point[ci] =
                            compute_blend(alpha_dest_aux_point, 1.f, background_color[ci], intensities_aux_point[ci]);
                    }
                }
                vec4 result;
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    result[ci] = intensities_aux_point[ci];
                }
                return result;
            };
#endif
            int texture_channels = num_descriptors;  // d_render_params.num_texture_channels;

            float dR_dpx = 0;
            float dR_dpy = 0;

#if 0
            if (d_render_params.test_backward_mode == 4)
            {
                // float I_px0[MAXCHANNELS];
                // float I_px1[MAXCHANNELS];
                // float I_py0[MAXCHANNELS];
                // float I_py1[MAXCHANNELS];
                // vec4 I_px0 = vec4(0, 0, 0, 0);
                // vec4 I_px1 = vec4(0, 0, 0, 0);
                // vec4 I_py0 = vec4(0, 0, 0, 0);
                // vec4 I_py1 = vec4(0, 0, 0, 0);

                vec4 I_px0 = compute_blend_with_point_at_x(px0);  //, I_px0);
                vec4 I_px1 = compute_blend_with_point_at_x(px1);  //, I_px1);
                vec4 I_py0 = compute_blend_with_point_at_x(py0);  //, I_py0);
                vec4 I_py1 = compute_blend_with_point_at_x(py1);  //, I_py1);

                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    float dI_dp_at_px0 = -(intensities_p_0_0_point_gradient[ci] - I_px0[ci]) * G_px0;
                    float dI_dp_at_px1 = (intensities_p_0_0_point_gradient[ci] - I_px1[ci]) * G_px1;
                    float dI_dp_at_py0 = -(intensities_p_0_0_point_gradient[ci] - I_py0[ci]) * G_py0;
                    float dI_dp_at_py1 = (intensities_p_0_0_point_gradient[ci] - I_py1[ci]) * G_py1;
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  // * alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  // * alpha_dest;
                }
            }
            else
#endif
            {
#pragma unroll
                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto g = sample_grad(ci, p_imgi);

                    auto T_p = sample_tex(ci, texture_index);

                    auto I_px0 = sample_forward(ci, px0);
                    auto I_px1 = sample_forward(ci, px1);
                    auto I_py0 = sample_forward(ci, py0);
                    auto I_py1 = sample_forward(ci, py1);
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    // The spatial derivatives at the neighboring points.
                    float dI_dp_at_px0 = -(T_p - I_px0) * G_px0;
                    float dI_dp_at_px1 = (T_p - I_px1) * G_px1;
                    float dI_dp_at_py0 = -(T_p - I_py0) * G_py0;
                    float dI_dp_at_py1 = (T_p - I_py1) * G_py1;
                    // Average between forward and backward diff. to get symmetric central diff.
                    // multiply by alpha_dest for individual point contribution
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  //* alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  //* alpha_dest;
                }
            }
            vec2 dR_dp = vec2(dR_dpx, dR_dpy) / float(texture_channels);
            // dR_dp*=alpha_dest;

            alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);


            // adapt for multiresolution rendering
            float scale = 1 * powf(0.5f, float(layer));

            float grad_scale    = 1.f;
            auto cam2           = cam;
            cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            vec3 position;
            vec3 normal;
            float drop_out_radius;
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion]               = d_render_params.PinholeIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_k, g_dis] = ProjectPointPinholeBackward(
                    position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    float k_factor = d_render_params.K_gradient_factor;

                    g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), k_factor * g_k(k));
                    }

                    float distortion_factor = d_render_params.distortion_gradient_factor;

                    // k3
                    g_dis(2) *= 0.25;

                    // k4 - 6
                    g_dis(3) *= 0.1;
                    g_dis(4) *= 0.1;
                    g_dis(5) *= 0.1;

                    // tangential distortion
                    g_dis(6) *= 0.1;
                    g_dis(7) *= 0.1;
                    for (int k = 0; k < 8; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                  distortion_factor * g_dis(k));
                    }
                    // Note we add a value less than 1 to increase float precision
                    float factor = 1.f / 1024.f;
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]                 = d_render_params.OcamIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_affine] = ProjectPointOcamBackward(
                    position, normal, dR_dp, V, cam2.crop_transform, aff, poly, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    // Points
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }

                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    // Intrinsics
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                }
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                CUDA_KERNEL_ASSERT(false);
            }
        }
    }
}

template <int num_descriptors>
__global__ void BlendBackwards(DevicePointCloud point_cloud, float* background_color, float* out_background_gradient,
                               int batch, int layer, ReducedImageInfo cam, bool need_point_gradients,
                               bool use_environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
        gy >= d_backward_params.in_gradient_image[layer].size(2))
        return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);

    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);
    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);


    // helper functions
    auto sample_depth = [&](int buffer_pos) -> float
    {
        int depth_i = d_alpha_comp_params_bw.collections[layer](0, buffer_pos);
        float depth = reinterpret_cast<float*>(&depth_i)[0];
        return depth;
    };
    auto sample_point_id = [&](int buffer_pos) -> int
    {
        int point_id_b = d_alpha_comp_params_bw.collections[layer](1, buffer_pos);
        return point_cloud.GetIndex(point_id_b);
    };


    {
        float alpha_dest = 1.f;
        float color_out[num_descriptors];
        CUDA_KERNEL_ASSERT(d_render_params.num_texture_channels <= num_descriptors);
        for (int ci = 0; ci < num_descriptors; ++ci) color_out[ci] = 0.f;

        // Jacobians
        float J_cdest_alpha[num_descriptors], J_cdest_col[num_descriptors], J_cdest_alphadest[num_descriptors],
            J_cdest_oldcdest[num_descriptors], J_alphadest_alpha, J_alphadest_alphadestold;

        // blend background color if no env map
        int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;
        // blend all together
        for (int i = 0; i < list_size; ++i)
        {
            bool is_foreground = true;
            if (i == size_of_chunk) is_foreground = false;

            int full_buffer_pos = offset_in_buffer + i;
            int texture_index   = 0;
            if (is_foreground) texture_index = sample_point_id(full_buffer_pos);

            CUDA_KERNEL_ASSERT(texture_index >= 0);
            CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

            float confidence_val = 1.f;

            if (is_foreground) confidence_val = d_texture.points_confidence_value(0, texture_index);

            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                float color   = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];
                color_out[ci] = compute_blend(alpha_dest, confidence_val, color, color_out[ci], &J_cdest_alpha[ci],
                                              &J_cdest_col[ci], &J_cdest_alphadest[ci], &J_cdest_oldcdest[ci]);
            }
            alpha_dest =
                compute_new_alphadest(alpha_dest, confidence_val, &J_alphadest_alpha, &J_alphadest_alphadestold);

            // write texture gradient
            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                float g = J_cdest_col[ci] * d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                float* grad_write_address = is_foreground ? &d_backward_params.out_gradient_texture(ci, texture_index)
                                                          : &out_background_gradient[ci];
                atomicAdd(grad_write_address, g);
            }

            // update Jacobians for confidence
            //  current running Js
            //  last "running" Js (for background not needed, as they are still stored in the J matrices
            if (is_foreground)
            {
                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    d_alpha_comp_params_bw.gradient_sum_backwards[layer](ci, offset_in_buffer + i) = J_cdest_alpha[ci];
                }
                d_alpha_comp_params_bw.gradient_sum_backwards[layer](d_render_params.num_texture_channels,
                                                                     offset_in_buffer + i) = J_alphadest_alpha;
            }
            // run Jacobians
            for (int j = 0; j < i; ++j)
            {
                float* alpha_run_dest = &d_alpha_comp_params_bw.gradient_sum_backwards[layer](
                    d_render_params.num_texture_channels, offset_in_buffer + j);
                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    float* intermediate_Jacobien =
                        &d_alpha_comp_params_bw.gradient_sum_backwards[layer](ci, offset_in_buffer + j);
                    *intermediate_Jacobien = 1 * intermediate_Jacobien[0] + J_cdest_alphadest[ci] * alpha_run_dest[0];
                }
                *alpha_run_dest = J_alphadest_alphadestold * alpha_run_dest[0];
            }
        }

        // compute final gradients, confidence for the background is currently not computed
        for (int i = 0; i < size_of_chunk; ++i)
        {
            int full_buffer_pos = offset_in_buffer + i;
            int point_id        = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
            int texture_index   = point_cloud.GetIndex(point_id);

            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                float J_conf = d_alpha_comp_params_bw.gradient_sum_backwards[layer](ci, full_buffer_pos);
                float g      = J_conf * d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                // write out gradient of confidence
                atomicAdd(&d_backward_params.out_gradient_confidence(0, texture_index), g);
            }
        }
    }

    if (!need_point_gradients) return;

    // approximate gradient for points
    ivec2 p_imgi = ivec2(gx, gy);
    if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
    {
        float alpha_dest = 1.f;

        for (int i = 0; i < size_of_chunk; ++i)
        {
            float intensities_p_0_0_point_gradient[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci) intensities_p_0_0_point_gradient[ci] = 0.f;

            {
                // compute intensity without point at own position
                float alpha_dest_point_gradient = 1.f;
                int offset_in_buffer_b          = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);
                for (int j = 0; j < size_of_chunk; ++j)
                {
                    if (j != i)
                    {
                        int full_buffer_pos_b = offset_in_buffer_b + j;
                        int texture_index_b   = sample_point_id(full_buffer_pos_b);

                        float confidence_val = d_texture.points_confidence_value(0, texture_index_b);
                        for (int ci = 0; ci < num_descriptors; ++ci)
                        {
                            float color                          = d_texture.in_texture(ci, texture_index_b);
                            intensities_p_0_0_point_gradient[ci] = compute_blend(
                                alpha_dest_point_gradient, confidence_val, color, intensities_p_0_0_point_gradient[ci]);
                        }
                        alpha_dest_point_gradient = compute_new_alphadest(alpha_dest_point_gradient, confidence_val);
                    }
                }
                if (!use_environment_map)
                {
                    // background
                    for (int ci = 0; ci < num_descriptors; ++ci)
                    {
                        intensities_p_0_0_point_gradient[ci] =
                            compute_blend(alpha_dest, 1.f, background_color[ci], intensities_p_0_0_point_gradient[ci]);
                    }
                }
            }

            int full_buffer_pos = offset_in_buffer + i;
            int point_id        = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
            int texture_index   = point_cloud.GetIndex(point_id);

            float confidence_val = d_texture.points_confidence_value(0, texture_index);

            ivec2 px0 = p_imgi + ivec2(-1, 0);
            ivec2 px1 = p_imgi + ivec2(1, 0);
            ivec2 py0 = p_imgi + ivec2(0, -1);
            ivec2 py1 = p_imgi + ivec2(0, 1);

            auto sample_grad = [&](int ci, ivec2 p) -> float
            { return d_backward_params.in_gradient_image[layer](batch, ci, p.y(), p.x()); };
            auto sample_forward = [&](int ci, ivec2 p) -> float
            { return d_forward_params.neural_out[layer](batch, ci, p.y(), p.x()); };
            auto sample_tex = [&](int ci, int uv) -> float { return d_texture.in_texture(ci, uv); };

#if 0
            // compute neighboring pixel with current point inserted into the blend
            auto compute_blend_with_point_at_x = [&](ivec2 x)  //, vec4 result)
            {
                float point_depth = sample_depth(full_buffer_pos);

                int size_of_aux_chunk  = d_render_params.per_image_atomic_counters[layer](batch, 0, x.y(), x.x());
                int offset_in_buffer_c = d_alpha_comp_params_bw.scanned_countings[layer](0, x.y(), x.x());

                float intensities_aux_point[MAXCHANNELS];
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) intensities_aux_point[ci] = 0.f;
                float alpha_dest_aux_point = 1.f;
                bool inserted_point        = false;
                for (int j = 0; j < size_of_aux_chunk; ++j)
                {
                    int full_buffer_pos_c = offset_in_buffer_c + j;

                    float depth_aux = sample_depth(full_buffer_pos_c);

                    if (!inserted_point && point_depth < depth_aux)
                    {
                        // insert point
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val);
                        inserted_point       = true;
                        --j;
                    }
                    else
                    {
                        // compute blend with other points
                        int texture_index_c    = sample_point_id(full_buffer_pos_c);
                        float confidence_val_c = d_texture.points_confidence_value(0, texture_index_c);
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index_c);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val_c, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val_c);
                    }
                }
                if (!use_environment_map)
                {
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        intensities_aux_point[ci] =
                            compute_blend(alpha_dest_aux_point, 1.f, background_color[ci], intensities_aux_point[ci]);
                    }
                }
                vec4 result;
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    result[ci] = intensities_aux_point[ci];
                }
                return result;
            };
#endif
            int texture_channels = num_descriptors;  // d_render_params.num_texture_channels;

            float dR_dpx = 0;
            float dR_dpy = 0;

#if 0
            if (d_render_params.test_backward_mode == 4)
            {
                // float I_px0[MAXCHANNELS];
                // float I_px1[MAXCHANNELS];
                // float I_py0[MAXCHANNELS];
                // float I_py1[MAXCHANNELS];
                // vec4 I_px0 = vec4(0, 0, 0, 0);
                // vec4 I_px1 = vec4(0, 0, 0, 0);
                // vec4 I_py0 = vec4(0, 0, 0, 0);
                // vec4 I_py1 = vec4(0, 0, 0, 0);

                vec4 I_px0 = compute_blend_with_point_at_x(px0);  //, I_px0);
                vec4 I_px1 = compute_blend_with_point_at_x(px1);  //, I_px1);
                vec4 I_py0 = compute_blend_with_point_at_x(py0);  //, I_py0);
                vec4 I_py1 = compute_blend_with_point_at_x(py1);  //, I_py1);

                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    float dI_dp_at_px0 = -(intensities_p_0_0_point_gradient[ci] - I_px0[ci]) * G_px0;
                    float dI_dp_at_px1 = (intensities_p_0_0_point_gradient[ci] - I_px1[ci]) * G_px1;
                    float dI_dp_at_py0 = -(intensities_p_0_0_point_gradient[ci] - I_py0[ci]) * G_py0;
                    float dI_dp_at_py1 = (intensities_p_0_0_point_gradient[ci] - I_py1[ci]) * G_py1;
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  // * alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  // * alpha_dest;
                }
            }
            else
#endif
            {
#pragma unroll
                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto g = sample_grad(ci, p_imgi);

                    auto T_p = sample_tex(ci, texture_index);

                    auto I_px0 = sample_forward(ci, px0);
                    auto I_px1 = sample_forward(ci, px1);
                    auto I_py0 = sample_forward(ci, py0);
                    auto I_py1 = sample_forward(ci, py1);
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    // The spatial derivatives at the neighboring points.
                    float dI_dp_at_px0 = -(T_p - I_px0) * G_px0;
                    float dI_dp_at_px1 = (T_p - I_px1) * G_px1;
                    float dI_dp_at_py0 = -(T_p - I_py0) * G_py0;
                    float dI_dp_at_py1 = (T_p - I_py1) * G_py1;
                    // Average between forward and backward diff. to get symmetric central diff.
                    // multiply by alpha_dest for individual point contribution
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  //* alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  //* alpha_dest;
                }
            }
            vec2 dR_dp = vec2(dR_dpx, dR_dpy) / float(texture_channels);
            // dR_dp*=alpha_dest;

            alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);


            // adapt for multiresolution rendering
            float scale = 1 * powf(0.5f, float(layer));

            float grad_scale    = 1.f;
            auto cam2           = cam;
            cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            vec3 position;
            vec3 normal;
            float drop_out_radius;
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion]               = d_render_params.PinholeIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_k, g_dis] = ProjectPointPinholeBackward(
                    position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    float k_factor = d_render_params.K_gradient_factor;

                    g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), k_factor * g_k(k));
                    }

                    float distortion_factor = d_render_params.distortion_gradient_factor;

                    // k3
                    g_dis(2) *= 0.25;

                    // k4 - 6
                    g_dis(3) *= 0.1;
                    g_dis(4) *= 0.1;
                    g_dis(5) *= 0.1;

                    // tangential distortion
                    g_dis(6) *= 0.1;
                    g_dis(7) *= 0.1;
                    for (int k = 0; k < 8; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                  distortion_factor * g_dis(k));
                    }
                    // Note we add a value less than 1 to increase float precision
                    float factor = 1.f / 1024.f;
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]                 = d_render_params.OcamIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_affine] = ProjectPointOcamBackward(
                    position, normal, dR_dp, V, cam2.crop_transform, aff, poly, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    // Points
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    // Intrinsics
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                }
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                CUDA_KERNEL_ASSERT(false);
            }
        }
    }
}
/*
__global__ void BlendBackwards(DevicePointCloud point_cloud, float* background_color, float* out_background_gradient,
                               int batch, int layer, ReducedImageInfo cam, bool need_point_gradients,
                               bool use_environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
        gy >= d_backward_params.in_gradient_image[layer].size(2))
        return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);

    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);
    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    {
        float alpha_dest = 1.f;
#define MAXCHANNELS 4
        float color_out[MAXCHANNELS];
        CUDA_KERNEL_ASSERT(d_render_params.num_texture_channels <= 4);
        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) color_out[ci] = 0.f;

        // Jacobians
        Matrix<double, 1, 1> J_cdest_alpha[MAXCHANNELS], J_cdest_col[MAXCHANNELS], J_cdest_alphadest[MAXCHANNELS],
            J_cdest_oldcdest[MAXCHANNELS], J_alphadest_alpha, J_alphadest_alphadestold;

        // blend background color if no env map
        int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;
        // blend all together
        for (int i = 0; i < list_size; ++i)
        {
            bool is_foreground = true;
            if (i == size_of_chunk) is_foreground = false;

            int full_buffer_pos = offset_in_buffer + i;
            int texture_index   = 0;
            if (is_foreground)
            {
                int point_id  = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
                texture_index = point_cloud.GetIndex(point_id);
            }
            CUDA_KERNEL_ASSERT(texture_index >= 0);
            CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

            float confidence_val = 1.f;
            if (is_foreground) confidence_val = d_texture.points_confidence_value(0, texture_index);

            for (int ci = 0; ci < MAXCHANNELS; ++ci)
            {
                J_cdest_alpha[ci].setZero();
                J_cdest_col[ci].setZero();
                J_cdest_alphadest[ci].setZero();
                J_cdest_oldcdest[ci].setZero();
            }
            J_alphadest_alpha.setZero();
            J_alphadest_alphadestold.setZero();

            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
            {
                float color   = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];
                color_out[ci] = compute_blend(alpha_dest, confidence_val, color, color_out[ci], &J_cdest_alpha[ci],
                                              &J_cdest_col[ci], &J_cdest_alphadest[ci], &J_cdest_oldcdest[ci]);
            }
            alpha_dest =
                compute_new_alphadest(alpha_dest, confidence_val, &J_alphadest_alpha, &J_alphadest_alphadestold);

            // write texture gradient
            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
            {
                float g = J_cdest_col[ci](0, 0) * d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                if (is_foreground)
                    atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);
                else
                    atomicAdd(&out_background_gradient[ci], g);
            }

            // update Jacobians for confidence
            //  current running Js
            //  last "running" Js (for background not needed, as they are still stored in the J matrices
            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
            {
                if (is_foreground)
                    d_alpha_comp_params_bw.gradient_sum_backwards[layer](ci, offset_in_buffer + i) =
                        J_cdest_alpha[ci](0, 0);
            }
            if (is_foreground)
                d_alpha_comp_params_bw.gradient_sum_backwards[layer](d_render_params.num_texture_channels,
                                                                     offset_in_buffer + i) = J_alphadest_alpha(0, 0);

            // run Jacobians
            for (int j = 0; j < i; ++j)
            {
                float* alpha_run_dest = &d_alpha_comp_params_bw.gradient_sum_backwards[layer](
                    d_render_params.num_texture_channels, offset_in_buffer + j);
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    float* intermediate_Jacobien =
                        &d_alpha_comp_params_bw.gradient_sum_backwards[layer](ci, offset_in_buffer + j);
                    intermediate_Jacobien[0] =
                        1 * intermediate_Jacobien[0] + J_cdest_alphadest[ci](0, 0) * alpha_run_dest[0];
                }
                alpha_run_dest[0] = J_alphadest_alphadestold(0, 0) * alpha_run_dest[0];
            }
        }


        // compute final gradients, confidence for the background is currently not computed
        for (int i = 0; i < size_of_chunk; ++i)
        {
            int full_buffer_pos = offset_in_buffer + i;
            int point_id        = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
            int texture_index   = point_cloud.GetIndex(point_id);

            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
            {
                float J_conf = d_alpha_comp_params_bw.gradient_sum_backwards[layer](ci, full_buffer_pos);
                float g      = J_conf * d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                // write out gradient of confidence
                atomicAdd(&d_backward_params.out_gradient_confidence(0, texture_index), g);
            }
        }
        // auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, gy, gx));
        // atomicAdd(dst_pos_weight, alpha_dest);
    }

    if (!need_point_gradients) return;

    // approximate gradient for points
    ivec2 p_imgi = ivec2(gx, gy);
    if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
    {
        float alpha_dest = 1.f;

        for (int i = 0; i < size_of_chunk; ++i)
        {
            auto sample_depth = [&](int buffer_pos) -> float
            {
                int depth_i = d_alpha_comp_params_bw.collections[layer](0, buffer_pos);
                float depth = reinterpret_cast<float*>(&depth_i)[0];
                return depth;
            };
            auto sample_point_id = [&](int buffer_pos) -> int
            {
                int point_id_b = d_alpha_comp_params_bw.collections[layer](1, buffer_pos);
                return point_cloud.GetIndex(point_id_b);
            };

            float intensities_p_0_0_point_gradient[MAXCHANNELS];
            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                intensities_p_0_0_point_gradient[ci] = 0.f;

            // compute intensity without point at own position
            float alpha_dest_point_gradient = 1.f;
            for (int j = 0; j < size_of_chunk; ++j)
            {
                if (j != i)
                {
                    int offset_in_buffer_b = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);
                    int full_buffer_pos_b  = offset_in_buffer_b + j;
                    int texture_index_b    = sample_point_id(full_buffer_pos_b);

                    float confidence_val = d_texture.points_confidence_value(0, texture_index_b);
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        float color                          = d_texture.in_texture(ci, texture_index_b);
                        intensities_p_0_0_point_gradient[ci] = compute_blend(
                            alpha_dest_point_gradient, confidence_val, color, intensities_p_0_0_point_gradient[ci]);
                    }
                    alpha_dest_point_gradient = compute_new_alphadest(alpha_dest_point_gradient, confidence_val);
                }
            }
            if (!use_environment_map)
            {
                // background
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    intensities_p_0_0_point_gradient[ci] =
                        compute_blend(alpha_dest, 1.f, background_color[ci], intensities_p_0_0_point_gradient[ci]);
                }
            }

            int full_buffer_pos = offset_in_buffer + i;
            int point_id        = d_alpha_comp_params_bw.collections[layer](1, full_buffer_pos);
            int texture_index   = point_cloud.GetIndex(point_id);

            float confidence_val = d_texture.points_confidence_value(0, texture_index);

            ivec2 px0 = p_imgi + ivec2(-1, 0);
            ivec2 px1 = p_imgi + ivec2(1, 0);
            ivec2 py0 = p_imgi + ivec2(0, -1);
            ivec2 py1 = p_imgi + ivec2(0, 1);

            auto sample_grad = [&](int ci, ivec2 p) -> float
            { return d_backward_params.in_gradient_image[layer](batch, ci, p.y(), p.x()); };
            auto sample_forward = [&](int ci, ivec2 p) -> float
            { return d_forward_params.neural_out[layer](batch, ci, p.y(), p.x()); };
            auto sample_tex = [&](int ci, int uv) -> float { return d_texture.in_texture(ci, uv); };

            // compute neighboring pixel with current point inserted into the blend
            auto compute_blend_with_point_at_x = [&](ivec2 x)  //, vec4 result)
            {
                float point_depth = sample_depth(full_buffer_pos);

                int size_of_aux_chunk  = d_render_params.per_image_atomic_counters[layer](batch, 0, x.y(), x.x());
                int offset_in_buffer_c = d_alpha_comp_params_bw.scanned_countings[layer](0, x.y(), x.x());

                float intensities_aux_point[MAXCHANNELS];
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) intensities_aux_point[ci] = 0.f;
                float alpha_dest_aux_point = 1.f;
                bool inserted_point        = false;
                for (int j = 0; j < size_of_aux_chunk; ++j)
                {
                    int full_buffer_pos_c = offset_in_buffer_c + j;

                    float depth_aux = sample_depth(full_buffer_pos_c);

                    if (!inserted_point && point_depth < depth_aux)
                    {
                        // insert point
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val);
                        inserted_point       = true;
                        --j;
                    }
                    else
                    {
                        // compute blend with other points
                        int texture_index_c    = sample_point_id(full_buffer_pos_c);
                        float confidence_val_c = d_texture.points_confidence_value(0, texture_index_c);
                        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                        {
                            float color = d_texture.in_texture(ci, texture_index_c);
                            intensities_aux_point[ci] =
                                compute_blend(alpha_dest_aux_point, confidence_val_c, color, intensities_aux_point[ci]);
                        }
                        alpha_dest_aux_point = compute_new_alphadest(alpha_dest_aux_point, confidence_val_c);
                    }
                }
                if (!use_environment_map)
                {
                    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                    {
                        intensities_aux_point[ci] =
                            compute_blend(alpha_dest_aux_point, 1.f, background_color[ci], intensities_aux_point[ci]);
                    }
                }
                vec4 result;
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    result[ci] = intensities_aux_point[ci];
                }
                return result;
            };

            int texture_channels = d_render_params.num_texture_channels;

            float dR_dpx = 0;
            float dR_dpy = 0;

            if (d_render_params.test_backward_mode == 4)
            {
                // float I_px0[MAXCHANNELS];
                // float I_px1[MAXCHANNELS];
                // float I_py0[MAXCHANNELS];
                // float I_py1[MAXCHANNELS];
                // vec4 I_px0 = vec4(0, 0, 0, 0);
                // vec4 I_px1 = vec4(0, 0, 0, 0);
                // vec4 I_py0 = vec4(0, 0, 0, 0);
                // vec4 I_py1 = vec4(0, 0, 0, 0);

                vec4 I_px0 = compute_blend_with_point_at_x(px0);  //, I_px0);
                vec4 I_px1 = compute_blend_with_point_at_x(px1);  //, I_px1);
                vec4 I_py0 = compute_blend_with_point_at_x(py0);  //, I_py0);
                vec4 I_py1 = compute_blend_with_point_at_x(py1);  //, I_py1);

                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    float dI_dp_at_px0 = -(intensities_p_0_0_point_gradient[ci] - I_px0[ci]) * G_px0;
                    float dI_dp_at_px1 = (intensities_p_0_0_point_gradient[ci] - I_px1[ci]) * G_px1;
                    float dI_dp_at_py0 = -(intensities_p_0_0_point_gradient[ci] - I_py0[ci]) * G_py0;
                    float dI_dp_at_py1 = (intensities_p_0_0_point_gradient[ci] - I_py1[ci]) * G_py1;
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  // * alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  // * alpha_dest;
                }
            }
            else
            {
#pragma unroll
                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto g = sample_grad(ci, p_imgi);

                    auto T_p = sample_tex(ci, texture_index);

                    auto I_px0 = sample_forward(ci, px0);
                    auto I_px1 = sample_forward(ci, px1);
                    auto I_py0 = sample_forward(ci, py0);
                    auto I_py1 = sample_forward(ci, py1);
                    auto G_px0 = sample_grad(ci, px0);
                    auto G_px1 = sample_grad(ci, px1);
                    auto G_py0 = sample_grad(ci, py0);
                    auto G_py1 = sample_grad(ci, py1);

                    // The spatial derivatives at the neighboring points.
                    float dI_dp_at_px0 = -(T_p - I_px0) * G_px0;
                    float dI_dp_at_px1 = (T_p - I_px1) * G_px1;
                    float dI_dp_at_py0 = -(T_p - I_py0) * G_py0;
                    float dI_dp_at_py1 = (T_p - I_py1) * G_py1;
                    // Average between forward and backward diff. to get symmetric central diff.
                    // multiply by alpha_dest for individual point contribution
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);  //* alpha_dest;
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);  //* alpha_dest;
                }
            }
            vec2 dR_dp = vec2(dR_dpx, dR_dpy) / float(texture_channels);
            // dR_dp*=alpha_dest;

            alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);


            // adapt for multiresolution rendering
            float scale = 1 * powf(0.5f, float(layer));

            float grad_scale    = 1.f;
            auto cam2           = cam;
            cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            vec3 position;
            vec3 normal;
            float drop_out_radius;
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_k, g_dis] =
                    ProjectPointPinholeBackward(position, normal, dR_dp, V, K, cam2.crop_transform, distortion,
                                                d_render_params.check_normal, d_render_params.dist_cutoff,
cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
if (d_backward_params.out_gradient_dynamic_points.data)
{
    for (int k = 0; k < g_point.rows(); ++k)
    {
        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                  g_point(k));  // * 0.1);
    }
    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
}
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    float k_factor = d_render_params.K_gradient_factor;
                    // Intrinsics
                    // g_k(2) *= 0.5;
                    // g_k(3) *= 0.5;

                    // sheer
                    g_k(4) *= 0.1;
                    g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), k_factor * g_k(k));
                    }

                    float distortion_factor = d_render_params.distortion_gradient_factor;

                    // k3
                    g_dis(2) *= 0.25;

                    // k4 - 6
                    g_dis(3) *= 0.1;
                    g_dis(4) *= 0.1;
                    g_dis(5) *= 0.1;

                    // tangential distortion
                    g_dis(6) *= 0.1;
                    g_dis(7) *= 0.1;
                    for (int k = 0; k < 8; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                  distortion_factor * g_dis(k));
                    }
                    // Note we add a value less than 1 to increase float precision
                    float factor = 1.f / 1024.f;
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_affine] =
                    ProjectPointOcamBackward(position, normal, dR_dp, V, cam2.crop_transform, aff, poly,
                                             d_render_params.check_normal, d_render_params.dist_cutoff,
cam2.crop_rotation);


                if (d_backward_params.out_gradient_points)
                {
                    // Points
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
if (d_backward_params.out_gradient_dynamic_points.data)
{
    for (int k = 0; k < g_point.rows(); ++k)
    {
        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                  g_point(k));  // * 0.1);
    }
    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
}
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f / 1024.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    // Intrinsics
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                }
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                CUDA_KERNEL_ASSERT(false);
            }
        }
    }
}*/

void PointRendererCache::BlendBackwards(int batch, NeuralPointCloudCuda point_cloud,
                                        std::vector<torch::Tensor> collectionbuffers, torch::Tensor background_color,
                                        std::vector<torch::Tensor> grad_sum_back_buffers, bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    auto out_gradient_background = output_gradient_background.data_ptr<float>();


    SAIGA_ASSERT(output_forward.size() == info->num_layers);

    UploadCollectionBuffersBackwards(collectionbuffers, std::vector<torch::Tensor>(), grad_sum_back_buffers, batch);

    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);
        SAIGA_ASSERT(batch < output_forward[i].size(0));
        auto in_out_neural_image = output_forward[i][batch];

        int image_batch_id = batch;
        auto cam           = info->images[image_batch_id];

        switch (info->params.num_texture_channels)
        {
            case 3:
            {
                ::BlendBackwards2<3>
                    <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, background, out_gradient_background, batch, i,
                                                           cam, info->params.need_point_gradient, use_environment_map);
                break;
            }
            case 4:
            {
                ::BlendBackwards2<4>
                    <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, background, out_gradient_background, batch, i,
                                                           cam, info->params.need_point_gradient, use_environment_map);
                break;
            }
            default:
                SAIGA_ASSERT(false);
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}
void PointRendererCache::BlendBackwardsFuzzy(int batch, NeuralPointCloudCuda point_cloud,
                                             std::vector<torch::Tensor> collectionbuffers,
                                             torch::Tensor background_color,
                                             std::vector<torch::Tensor> grad_sum_back_buffers, bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    auto out_gradient_background = output_gradient_background.data_ptr<float>();


    SAIGA_ASSERT(output_forward.size() == info->num_layers);

    UploadCollectionBuffersBackwards(collectionbuffers, std::vector<torch::Tensor>(), grad_sum_back_buffers, batch);

    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);
        SAIGA_ASSERT(batch < output_forward[i].size(0));
        auto in_out_neural_image = output_forward[i][batch];

        int image_batch_id = batch;
        auto cam           = info->images[image_batch_id];
        switch (info->params.num_texture_channels)
        {
            case 3:
            {
                ::BlendBackwardsFuzzy<3>
                    <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, background, out_gradient_background, batch, i,
                                                           cam, info->params.need_point_gradient, use_environment_map);

                break;
            }
            case 4:
            {
                ::BlendBackwardsFuzzy<4>
                    <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, background, out_gradient_background, batch, i,
                                                           cam, info->params.need_point_gradient, use_environment_map);
                break;
            }
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}


#if 0
template <int num_descriptors>
__global__ void BlendBackwardsBilinear(DevicePointCloud point_cloud, float* background_color,
                                       float* out_background_gradient, int batch, int layer, ReducedImageInfo cam,
                                       bool need_point_gradients, bool use_environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
        gy >= d_backward_params.in_gradient_image[layer].size(2))
        return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);

    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);
    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    /*
    auto get_point_id = [&](int pos)
    {
        int buf_id          = d_alpha_comp_params_bw.collections[layer](1, pos);
        float p_id_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 0);
        int p_id            = reinterpret_cast<int*>(&p_id_as_float)[0];
        return p_id;
    };

    auto sample_texture_id = [&](int buffer_pos) -> int
    {
        int point_id_b = get_point_id(buffer_pos);
        return point_cloud.GetIndex(point_id_b);
    };
    auto get_subpixel_weight = [&](int pos)
    {
        int buf_id                   = d_alpha_comp_params_bw.collections[layer](1, pos);
        float uv_x                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 1);
        float uv_y                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 2);
        float p_blend_index_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 3);
        int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
        vec4 blend_facs              = compute_blending_fac(vec2(uv_x, uv_y));
        return blend_facs[blend_index];
    };*/

    auto get_point_id_and_subpixel_weights = [&](int pos)
    {
        int buf_id                   = d_alpha_comp_params_bw.collections[layer](1, pos);
        float p_id_as_float          = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 0);
        float uv_x                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 1);
        float uv_y                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 2);
        float p_blend_index_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 3);
        int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
        vec4 blend_facs              = compute_blending_fac(vec2(uv_x, uv_y));
        return vec2(p_id_as_float, blend_facs[blend_index]);
    };

    {
        float alpha_dest = 1.f;

        // blend background color if no env map
        int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;
        // blend all together
        for (int i = 0; i < list_size; ++i)
        {
            bool is_foreground = !(i == size_of_chunk);

            int full_buffer_pos = offset_in_buffer + i;

            // int point_id       = 0;
            float bilinear_fac = 1.f;
            int texture_index  = 0;
            if (is_foreground)
            {
                vec2 stored_data = get_point_id_and_subpixel_weights(full_buffer_pos);
                int point_id     = reinterpret_cast<int*>(&stored_data.x())[0];
                bilinear_fac     = stored_data.y();
                texture_index    = point_cloud.GetIndex(point_id);
            }

            CUDA_KERNEL_ASSERT(texture_index >= 0);
            CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

            float alpha_val = 1.f;
            if (is_foreground) alpha_val = bilinear_fac * d_texture.points_confidence_value(0, texture_index);

            float colors[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci)
                colors[ci] = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];

            float grad_in[num_descriptors];
            float g_alpha = 0;
            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                grad_in[ci] = d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                // texture gradient
                float g                   = alpha_dest * alpha_val * grad_in[ci];
                float* grad_write_address = is_foreground ? &d_backward_params.out_gradient_texture(ci, texture_index)
                                                          : &out_background_gradient[ci];
                atomicAdd(grad_write_address, g);

                // alpha gradient
                g_alpha += colors[ci] * grad_in[ci];
            }
            g_alpha *= alpha_dest;

            if (is_foreground) d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos) += g_alpha;

            for (int j = 0; j < i; ++j)
            {
                int full_buffer_pos_iter = offset_in_buffer + j;

                vec2 stored_data        = get_point_id_and_subpixel_weights(full_buffer_pos);
                int point_id_iter       = reinterpret_cast<int*>(&stored_data.x())[0];
                float bilinear_fac_iter = stored_data.y();

                int texture_index_iter = point_cloud.GetIndex(point_id_iter);
                // float confidence_val_iter = d_texture.points_confidence_value(0, texture_index_iter);
                float alpha_val_iter = bilinear_fac_iter * d_texture.points_confidence_value(0, texture_index_iter);
                float g_iter         = 0;
                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    const float epsilon = 1e-9;
                    float dem           = 1 / (1 - alpha_val_iter + epsilon);
                    float g_alpha_iter =
                        (grad_in[ci] * colors[ci] * alpha_dest * alpha_val / (1 - alpha_val_iter + epsilon));
                    g_iter -= g_alpha_iter;
                    // g += -grad_in[ci] * color_iter * alpha_dest * confidence_val * dem;
                }
                d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos_iter) += g_iter;
            }
            alpha_dest = compute_new_alphadest(alpha_dest, alpha_val);

            // break loop if contibution is very low
            if (alpha_dest < 0.01)
            {
                // list_size     = i + 1;
                if (is_foreground) size_of_chunk = i + 1;
                break;
            }
        }
    }
    /*
        auto get_subpixel_position = [&](int pos)
        {
            int buf_id = d_alpha_comp_params_bw.collections[layer](1, pos);
            float uv_x = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 1);
            float uv_y = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 2);
            return vec2(uv_x, uv_y);
        };
        auto get_subpixel_blend_index = [&](int pos)
        {
            int buf_id                   = d_alpha_comp_params_bw.collections[layer](1, pos);
            float p_blend_index_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 3);
            int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
            return blend_index;
        };*/

    auto get_point_id_and_subpixel_pos_and_float_blend_index = [&](int pos)
    {
        int buf_id                   = d_alpha_comp_params_bw.collections[layer](1, pos);
        float p_id_as_float          = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 0);
        float uv_x                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 1);
        float uv_y                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 2);
        float p_blend_index_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 3);

        return vec4(p_id_as_float, uv_x, uv_y, p_blend_index_as_float);
    };



    // return;

    ivec2 p_imgi = ivec2(gx, gy);
    if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
    {
        for (int i = 0; i < size_of_chunk; ++i)
        {
            /*
             * Gradients from each pixel are accumulated independently to one point:
             * The gradient from color w.r.t. the point alpha is computed before this step
             * Here the gradient w.r.t the confidence and the gradient w.r.t. the blend_factors (thus the spatial
             * derivatives) are computed
             */

            int full_buffer_pos = offset_in_buffer + i;
            // int texture_index   = sample_texture_id(full_buffer_pos);

            vec4 stored_data = get_point_id_and_subpixel_pos_and_float_blend_index(full_buffer_pos);
            int point_id     = reinterpret_cast<int*>(&stored_data.x())[0];
            vec2 uv          = vec2(stored_data.y(), stored_data.z());
            int blend_index  = reinterpret_cast<int*>(&stored_data.w())[0];

            int texture_index = point_cloud.GetIndex(point_id);
            // vec2 uv         = get_subpixel_position(full_buffer_pos);
            // int blend_index = get_subpixel_blend_index(full_buffer_pos);

            float grad_alpha = d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos);


            Matrix<float, 4, 2> J_uv_b;
            vec4 blend_factors = compute_blending_fac(uv, &J_uv_b);

            // write point confidence gradient, dA_dc = blend_fac
            float* point_confidence_grad_address = &d_backward_params.out_gradient_confidence(0, texture_index);
            float grad_point_confidence          = blend_factors[blend_index] * grad_alpha;
            atomicAdd(point_confidence_grad_address, grad_point_confidence);

            if (!need_point_gradients) continue;

            // if (i > 0) continue;

            // compute dR_dp by singling out the contribution by this pixel, dA_db = confidence
            float confidence_val    = d_texture.points_confidence_value(0, texture_index);
            float grad_blend_single = confidence_val * grad_alpha;


            // compute dP_dA : J_uv_b^T * blend_fac[index]
            vec4 grad_blending;
            grad_blending.setZero();
            grad_blending[blend_index] = grad_blend_single;

            vec2 dR_dp = J_uv_b.transpose() * grad_blending;

            // adapt for multiresolution rendering
            float scale = 1 * powf(0.5f, float(layer));

            float grad_scale    = 1.f;
            auto cam2           = cam;
            cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);

            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            vec3 position;
            vec3 normal;
            float drop_out_radius;
            // int point_id = get_point_id(full_buffer_pos);

            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion]               = d_render_params.PinholeIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_k, g_dis] = ProjectPointPinholeBackward(
                    position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }

                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f);
                    // atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 0.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    float k_factor = d_render_params.K_gradient_factor;

                    g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), k_factor * g_k(k));
                    }

                    float distortion_factor = d_render_params.distortion_gradient_factor;

                    // k3
                    g_dis(2) *= 0.25;

                    // k4 - 6
                    g_dis(3) *= 0.1;
                    g_dis(4) *= 0.1;
                    g_dis(5) *= 0.1;

                    // tangential distortion
                    g_dis(6) *= 0.1;
                    g_dis(7) *= 0.1;
                    for (int k = 0; k < 8; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                  distortion_factor * g_dis(k));
                    }
                    // Note we add a value less than 1 to increase float precision
                    float factor = 1.f / 1024.f;
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]                 = d_render_params.OcamIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_affine] = ProjectPointOcamBackward(
                    position, normal, dR_dp, V, cam2.crop_transform, aff, poly, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    // Points
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }

                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    // Intrinsics
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                }
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                CUDA_KERNEL_ASSERT(false);
            }
        }
    }
}

#endif

template <int num_descriptors>
__global__ void BlendBackwardsBilinear(DevicePointCloud point_cloud, float* background_color,
                                       float* out_background_gradient, int batch, int layer, ReducedImageInfo cam,
                                       bool need_point_gradients, bool use_environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
        gy >= d_backward_params.in_gradient_image[layer].size(2))
        return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params_bw.scanned_countings[layer](0, gy, gx);

    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);
    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    auto get_point_id = [&](int pos)
    {
        int buf_id          = d_alpha_comp_params_bw.collections[layer](1, pos);
        float p_id_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 0);
        int p_id            = reinterpret_cast<int*>(&p_id_as_float)[0];
        return p_id;
    };

    auto sample_texture_id = [&](int buffer_pos) -> int
    {
        int point_id_b = get_point_id(buffer_pos);
        return point_cloud.GetIndex(point_id_b);
    };
    auto get_subpixel_weight = [&](int pos)
    {
        int buf_id                   = d_alpha_comp_params_bw.collections[layer](1, pos);
        float uv_x                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 1);
        float uv_y                   = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 2);
        float p_blend_index_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 3);
        int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
        vec4 blend_facs              = compute_blending_fac(vec2(uv_x, uv_y));
        return blend_facs[blend_index];
    };

    {
        float alpha_dest = 1.f;

        // blend background color if no env map
        int list_size = use_environment_map ? size_of_chunk : size_of_chunk + 1;
        // blend all together
        for (int i = 0; i < list_size; ++i)
        {
            bool is_foreground = true;
            if (i == size_of_chunk) is_foreground = false;

            int full_buffer_pos = offset_in_buffer + i;
            int texture_index   = 0;
            if (is_foreground) texture_index = sample_texture_id(full_buffer_pos);
            // return;
            // return;

            CUDA_KERNEL_ASSERT(texture_index >= 0);
            CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

            float alpha_val = 1.f;

            // if (is_foreground) alpha_val = d_texture.points_confidence_value(0, texture_index);
            // return;
            if (is_foreground)
            {
                alpha_val = get_subpixel_weight(full_buffer_pos) * d_texture.points_confidence_value(0, texture_index);
            }
            // return;

            float colors[num_descriptors];
            for (int ci = 0; ci < num_descriptors; ++ci)
                colors[ci] = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];

            float grad_in[num_descriptors];
            float g_alpha = 0;
            for (int ci = 0; ci < num_descriptors; ++ci)
            {
                grad_in[ci]               = d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                float g                   = alpha_dest * alpha_val * grad_in[ci];
                float* grad_write_address = is_foreground ? &d_backward_params.out_gradient_texture(ci, texture_index)
                                                          : &out_background_gradient[ci];
                atomicAdd(grad_write_address, g);

                g_alpha += colors[ci] * grad_in[ci];
            }
            g_alpha *= alpha_dest;
            //  atomicAdd(&d_backward_params.out_gradient_confidence(0, texture_index), g_alpha);

            if (is_foreground) d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos) += g_alpha;


            for (int j = 0; j < i; ++j)
            {
                int full_buffer_pos_iter = offset_in_buffer + j;
                int texture_index_iter   = sample_texture_id(full_buffer_pos_iter);
                // float confidence_val_iter = d_texture.points_confidence_value(0, texture_index_iter);
                float alpha_val_iter = get_subpixel_weight(full_buffer_pos_iter) *
                                       d_texture.points_confidence_value(0, texture_index_iter);
                float g_iter = 0;
                for (int ci = 0; ci < num_descriptors; ++ci)
                {
                    const float epsilon = 1e-9;
                    float dem           = 1 / (1 - alpha_val_iter + epsilon);
                    float g_alpha_iter =
                        (grad_in[ci] * colors[ci] * alpha_dest * alpha_val / (1 - alpha_val_iter + epsilon));
                    g_iter -= g_alpha_iter;
                    // g += -grad_in[ci] * color_iter * alpha_dest * confidence_val * dem;
                }
                // float* grad_address_iter = &d_backward_params.out_gradient_confidence(0, texture_index_iter);
                // atomicAdd(grad_address_iter, g_iter);
                d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos_iter) += g_iter;
            }
            alpha_dest = compute_new_alphadest(alpha_dest, alpha_val);

            // break loop if contibution is very low
            if (alpha_dest < 0.001)
            {
                // list_size     = i + 1;
                if (is_foreground) size_of_chunk = i + 1;
                break;
            }
        }
    }
    /* for (int i = 0; i < size_of_chunk; ++i)
     {
         int full_buffer_pos = offset_in_buffer + i;
         int texture_index   = sample_texture_id(full_buffer_pos);

         float* grad_address = &d_backward_params.out_gradient_confidence(0, texture_index);

         float grad_point_alpha = d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos);


         atomicAdd(grad_address, d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos));
     }
 */
    // if (!need_point_gradients) return;

    auto get_subpixel_position = [&](int pos)
    {
        int buf_id = d_alpha_comp_params_bw.collections[layer](1, pos);
        float uv_x = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 1);
        float uv_y = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 2);
        return vec2(uv_x, uv_y);
    };
    auto get_subpixel_blend_index = [&](int pos)
    {
        int buf_id                   = d_alpha_comp_params_bw.collections[layer](1, pos);
        float p_blend_index_as_float = d_alpha_comp_params_bw.per_point_data[layer](buf_id, 3);
        int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
        return blend_index;
    };

    // return;

    ivec2 p_imgi = ivec2(gx, gy);
    if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
    {
        for (int i = 0; i < size_of_chunk; ++i)
        {
            /*
             * Gradients from each pixel are accumulated independently to one point:
             * The gradient from color w.r.t. the point alpha is computed before this step
             * Here the gradient w.r.t the confidence and the gradient w.r.t. the blend_factors (thus the spatial
             * derivatives) are computed
             */

            int full_buffer_pos = offset_in_buffer + i;
            int texture_index   = sample_texture_id(full_buffer_pos);

            vec2 uv         = get_subpixel_position(full_buffer_pos);
            int blend_index = get_subpixel_blend_index(full_buffer_pos);

            float grad_alpha = d_alpha_comp_params_bw.gradient_sum_backwards[layer](0, full_buffer_pos);


            Matrix<float, 4, 2> J_uv_b;
            vec4 blend_factors = compute_blending_fac(uv, &J_uv_b);

            // write point confidence gradient, dA_dc = blend_fac
            float* point_confidence_grad_address = &d_backward_params.out_gradient_confidence(0, texture_index);
            float grad_point_confidence          = blend_factors[blend_index] * grad_alpha;
            atomicAdd(point_confidence_grad_address, grad_point_confidence);

            if (!need_point_gradients) continue;

            // if (i > 0) continue;

            // compute dR_dp by singling out the contribution by this pixel, dA_db = confidence
            float confidence_val    = d_texture.points_confidence_value(0, texture_index);
            float grad_blend_single = confidence_val * grad_alpha;


            // compute dP_dA : J_uv_b^T * blend_fac[index]
            vec4 grad_blending;
            grad_blending.setZero();
            grad_blending[blend_index] = grad_blend_single;

            vec2 dR_dp = J_uv_b.transpose() * grad_blending;

            // adapt for multiresolution rendering
            float scale = 1 * powf(0.5f, float(layer));

            float grad_scale    = 1.f;
            auto cam2           = cam;
            cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            vec3 position;
            vec3 normal;
            float drop_out_radius;
            int point_id = get_point_id(full_buffer_pos);

            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion]               = d_render_params.PinholeIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_k, g_dis] = ProjectPointPinholeBackward(
                    position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k), g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }

                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f);
                    // atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 0.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    float k_factor = d_render_params.K_gradient_factor;

                    g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), k_factor * g_k(k));
                    }

                    float distortion_factor = d_render_params.distortion_gradient_factor;

                    // k3
                    g_dis(2) *= 0.25;

                    // k4 - 6
                    g_dis(3) *= 0.1;
                    g_dis(4) *= 0.1;
                    g_dis(5) *= 0.1;

                    // tangential distortion
                    g_dis(6) *= 0.1;
                    g_dis(7) *= 0.1;
                    for (int k = 0; k < 8; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                  distortion_factor * g_dis(k));
                    }
                    // Note we add a value less than 1 to increase float precision
                    float factor = 1.f / 1024.f;
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]                 = d_render_params.OcamIntrinsics(cam.camera_index);
                auto [g_point, g_pose, g_affine] = ProjectPointOcamBackward(
                    position, normal, dR_dp, V, cam2.crop_transform, aff, poly, d_render_params.check_normal,
                    d_render_params.dist_cutoff, cam2.crop_rotation);

                if (d_backward_params.out_gradient_points)
                {
                    // Points
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                }
                if (d_backward_params.out_gradient_dynamic_points.data)
                {
                    for (int k = 0; k < g_point.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k),
                                  g_point(k));  // * 0.1);
                    }
                    atomicAdd(&d_backward_params.out_gradient_dynamic_points_count(batch, point_id), 1);
                }
                if (d_backward_params.out_gradient_pose)
                {
                    // Extrinsics
                    for (int k = 0; k < g_pose.rows(); ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1.f);
                }

                if (d_backward_params.out_gradient_intrinsics_count)
                {
                    // Intrinsics
                    for (int k = 0; k < 5; ++k)
                    {
                        atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                }
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                CUDA_KERNEL_ASSERT(false);
            }
        }
    }
}


void PointRendererCache::BlendBackwardsBilinear(int batch, NeuralPointCloudCuda point_cloud,
                                                std::vector<torch::Tensor> collectionbuffers,
                                                std::vector<torch::Tensor> per_point_data_buffer,
                                                torch::Tensor background_color,
                                                std::vector<torch::Tensor> grad_sum_back_buffers,
                                                bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    auto out_gradient_background = output_gradient_background.data_ptr<float>();


    // always use background descriptors
    if (info->params.no_envmap_at_points) use_environment_map = false;

    SAIGA_ASSERT(output_forward.size() == info->num_layers);

    UploadCollectionBuffersBackwards(collectionbuffers, per_point_data_buffer, grad_sum_back_buffers, batch);

    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);
        SAIGA_ASSERT(batch < output_forward[i].size(0));
        auto in_out_neural_image = output_forward[i][batch];

        int image_batch_id = batch;
        auto cam           = info->images[image_batch_id];

        switch (info->params.num_texture_channels)
        {
            case 3:
            {
                // std::cout << TensorInfo(background_color) << TensorInfo(output_gradient_background) << std::endl;
                ::BlendBackwardsBilinear<3>
                    <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, background, out_gradient_background, batch, i,
                                                           cam, info->params.need_point_gradient, use_environment_map);
                break;
            }
            case 4:
            {
                ::BlendBackwardsBilinear<4>
                    <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, background, out_gradient_background, batch, i,
                                                           cam, info->params.need_point_gradient, use_environment_map);
                break;
            }
            default:
                SAIGA_ASSERT(false);
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}