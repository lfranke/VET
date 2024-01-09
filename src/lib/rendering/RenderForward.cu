/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// functions for forward rendering
// #include "saiga/colorize.h"
#include "saiga/cuda/bitonicSort.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "AlphaListSort.h"
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

__device__ __constant__ DeviceAlphaCompositionParams d_alpha_comp_params;


__device__ inline bool checkBoundingBox(int point_b_id, vec3 bb_min, vec3 bb_len, ReducedImageInfo& cam,
                                        Sophus::SE3f& V)
{
    return false;

    const vec3 bb_facs[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

    /*
     *  (-1,1) , (0,1) , (1,1)
     *  (-1,0) , (0,0) , (1,0)
     *  (-1,-1), (0,-1), (1,-1)
     */

    ivec2 in_out(0, 0);
    bool first_point = true;

    for (int i = 0; i < 8; ++i)
    {
        vec3 f = bb_facs[i].array() * bb_len.array();
        vec3 p = bb_min + f;
        vec3 ip_z;

        if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
        {
            CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
            auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
            ip_z                 = ProjWorldToPinhole(p, V, K, distortion, d_render_params.dist_cutoff);
        }
        else if (cam.camera_model_type == CameraModel::OCAM)
        {
            auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
            ip_z             = ProjWorldToOCAM(p, V, aff, poly, d_render_params.dist_cutoff);
        }
        else if (cam.camera_model_type == CameraModel::SPHERICAL)
        {
            return false;
        }

        vec2 ip      = cam.crop_transform.normalizedToImage(ip_z.head<2>());
        ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

        ivec2 image_dimensions = d_render_params.depth[0].Image().Vector();

        // Check in image ( if one corner is in the image, it needs to be drawn
        if (d_render_params.depth[0].Image().inImage2(p_imgi(1), p_imgi(0)))
        {
            return false;
        }
        else
        {
            if (first_point)
            {
                // signed bit if left, center or right
                in_out(0)   = (p_imgi.x() < 0 ? -1 : (p_imgi.x() >= image_dimensions.x() ? 1 : 0));
                in_out(1)   = (p_imgi.y() < 0 ? -1 : (p_imgi.y() >= image_dimensions.y() ? 1 : 0));
                first_point = false;
            }
            else
            {
                int f_x = (p_imgi.x() < 0 ? -1 : (p_imgi.x() >= image_dimensions.x() ? 1 : 0));
                int f_y = (p_imgi.y() < 0 ? -1 : (p_imgi.y() >= image_dimensions.y() ? 1 : 0));
                // if same quadrant, ok and next
                if (f_x == in_out.x() && f_y == in_out.y()) continue;
                // if completely different indices -> uncullable without further investigation how the edges lie exactly
                if (f_x != in_out.x() && f_y != in_out.y()) return false;
                // only one coordinate is different:
                //  if the unchanged coordinate is zero: crossing the view port
                if (f_x != in_out.x() && in_out.y() == 0) return false;
                if (f_y != in_out.y() && in_out.x() == 0) return false;
                // set it to zero so the check above filters edges intersecting the screen
                if (f_x != in_out.x())
                    in_out(0) = 0;
                else
                    in_out(1) = 0;
            }
        }
    }
    return true;
}

__device__ inline vec2 rotateCropAroundCenter(vec2 point, vec2 center, ReducedImageInfo& cam)
{
    point -= center;
    point = cam.crop_rotation * point;
    point += center;
    return point;
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
    }
    else if (cam.camera_model_type == CameraModel::OCAM)
    {
        auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
        thrust::tie(image_p_a, z) =
            ProjectPointOcam(position, normal, V, aff, poly, check_normal, d_render_params.dist_cutoff);
        radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
    }
    else if (cam.camera_model_type == CameraModel::SPHERICAL)
    {
        thrust::tie(image_p_a, z) = ProjectPointSpherical(
            position, normal, V,
            vec2(d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h), check_normal,
            d_render_params.dist_cutoff);
        radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
        ip            = image_p_a;
        return {ip, z, radius_pixels};
    }
    else
    {
        CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
    }
    ip = cam.crop_transform.normalizedToImage(image_p_a);
    ip = rotateCropAroundCenter(ip, vec2(cam.w / 2, cam.h / 2), cam);
    return {ip, z, radius_pixels};
}

__inline__ __device__ thrust::tuple<vec2, float, float> GetProjectedPoint(vec3 position, vec3 normal,
                                                                          float drop_out_radius, int point_id,
                                                                          ReducedImageInfo& cam)
{
    return ProjPoint(position, normal, drop_out_radius, cam, d_render_params.check_normal);
}


__device__ inline bool discard_point_for_confidence(int texture_index)
{
    float confidence = d_texture.points_confidence_value(0, texture_index);
    if (confidence < d_render_params.stability_cutoff_value && d_render_params.viewer_only)
    {
        return true;
    }
    return false;
}



template <int num_layers, bool opt_test, bool opt_ballot, bool opt_early_z>
__global__ void DepthPrepassMulti(DevicePointCloud point_cloud, ReducedImageInfo cam, int batch, int points_per_thread,
                                  bool train)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)

    Sophus::SE3f _V = d_render_params.Pose(cam.image_index);
#if 0
    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {

    for (int iter = 0, point_id = grid.thread_rank()*point_per_thread; point_id < point_cloud.Size() && iter < point_per_thread; ++iter, point_id += 1)
    {
#endif
    for (int p_nr = 0; p_nr < points_per_thread; ++p_nr)
    {
        int point_id = threadNumInBlock + p_nr * threadsPerBlock + blockNumInGrid * threadsPerBlock * points_per_thread;
        if (point_id >= point_cloud.Size()) break;

        int conf_id = point_cloud.GetIndex(point_id);
        if (discard_point_for_confidence(conf_id)) continue;

        vec2 ip;
        float z;
        float radius_pixels;

        if (opt_test)
        {
            float* dst    = &d_render_params.tmp_projections.Get({batch, point_id, 0});
            float4 res    = ((float4*)dst)[0];
            ip(0)         = res.x;
            ip(1)         = res.y;
            z             = res.z;
            radius_pixels = res.w;

            if (z <= 0) continue;
        }
        else
        {
            vec3 position;
            vec3 normal;
            vec2 image_p_a;
            float drop_out_radius;

            CUDA_KERNEL_ASSERT(cam.image_index >= 0);
            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            // return;

            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);
            thrust::tie(ip, z, radius_pixels) = GetProjectedPoint(position, normal, drop_out_radius, point_id, cam);

            /*  if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
              {
                  CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                  auto [K, distortion]      = d_render_params.PinholeIntrinsics(cam.camera_index);
                  thrust::tie(image_p_a, z) = ProjectPointPinhole(
                      position, normal, V, K, distortion, d_render_params.check_normal, d_render_params.dist_cutoff);
                  radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;
                  ip            = cam.crop_transform.normalizedToImage(image_p_a);
              }
              else if (cam.camera_model_type == CameraModel::OCAM)
              {
                  auto [aff, poly]          = d_render_params.OcamIntrinsics(cam.camera_index);
                  thrust::tie(image_p_a, z) = ProjectPointOcam(position, normal, V, aff, poly,
                                                               d_render_params.check_normal,
              d_render_params.dist_cutoff); radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx *
              drop_out_radius / z; ip            = cam.crop_transform.normalizedToImage(image_p_a);
              }
              else if (cam.camera_model_type == CameraModel::SPHERICAL)
              {
                  thrust::tie(image_p_a, z) = ProjectPointSpherical(
                      position, normal, V, vec2(d_render_params.depth[0].Image().w, d_render_params.depth[0].Image().h),
                      d_render_params.check_normal, d_render_params.dist_cutoff);
                  radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                  ip            = image_p_a;
                  // ip = cam.crop_transform.normalizedToImage(image_p_a);
              }
              */
            if (z == 0) continue;
        }

#pragma unroll
        for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
        {
            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                break;
            }

            ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

            // Check in image
            if (!d_render_params.depth[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;

            float* dst_pos = &(d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0)));

            if (opt_early_z)
            {
                // earlyz
                if (z >= *dst_pos) continue;
            }

            int i_depth = reinterpret_cast<int*>(&z)[0];

#ifdef _CG_HAS_MATCH_COLLECTIVE
            if constexpr (opt_ballot)
            {
                auto ballot   = subgroupPartitionNV(p_imgi);
                int min_depth = subgroupPartitionedMinNV(i_depth, ballot);
                if (ballot.thread_rank() == 0)
                {
                    atomicMin((int*)dst_pos, min_depth);
                }
            }
            else
#endif

            {
                atomicMin((int*)dst_pos, i_depth);
            }
        }
    }
}



template <int num_layers, bool opt_test>
__global__ void RenderForwardMulti(DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int batch,
                                   int points_per_thread, bool train)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)

    // int batch_fetch_index = d_render_params.use_point_confidence ? (batch/2) : batch;

    Sophus::SE3f _V = d_render_params.Pose(cam.image_index);
#if 0
    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
    for (int iter = 0, point_id = grid.thread_rank()*points_per_thread; point_id < point_cloud.Size() && iter < points_per_thread; ++iter, point_id += 1)
    {
#endif
    for (int p_nr = 0; p_nr < points_per_thread; ++p_nr)
    {
        int point_id = threadNumInBlock + p_nr * threadsPerBlock + blockNumInGrid * threadsPerBlock * points_per_thread;
        if (point_id >= point_cloud.Size()) break;

        int conf_id = point_cloud.GetIndex(point_id);
        if (discard_point_for_confidence(conf_id)) continue;

        bool drop_out = dropout_p[point_id] == 1;

        if (drop_out) continue;

        vec2 ip;
        float z;
        float radius_pixels;
        vec3 dir_for_view_based_descs;

        if (opt_test)
        {
            float* dst    = &d_render_params.tmp_projections.Get({batch, point_id, 0});
            float4 res    = ((float4*)dst)[0];
            ip(0)         = res.x;
            ip(1)         = res.y;
            z             = res.z;
            radius_pixels = res.w;

            if (z <= 0) continue;
        }
        else
        {
            vec3 position;
            vec3 normal;
            vec2 image_p_a;
            float drop_out_radius;

            Sophus::SE3f V                                 = d_render_params.Pose(cam.image_index);
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);
            thrust::tie(ip, z, radius_pixels) = GetProjectedPoint(position, normal, drop_out_radius, point_id, cam);

            if (z == 0) continue;
        }

#pragma unroll
        for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
        {
            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                break;
            }

            ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

            // Check in image
            if (!d_render_params.depth[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;


            float image_depth = d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0));
            if (z > image_depth * (d_render_params.depth_accept + 1)) continue;

            int texture_index = point_cloud.GetIndex(point_id);
            //  CUDA_KERNEL_ASSERT(texture_index == 15);

            CUDA_KERNEL_ASSERT(texture_index >= 0);
            CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);
            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
            {
                float t = d_texture.in_texture(ci, texture_index);
                atomicAdd(&d_forward_params.neural_out[layer](batch, ci, p_imgi(1), p_imgi(0)), t);
            }
            auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, p_imgi(1), p_imgi(0)));
            atomicAdd(dst_pos_weight, 1);
        }
    }
}

void PointRendererCache::PushParametersForward()
{
    SAIGA_OPTIONAL_TIME_MEASURE("Param Upload", info->timer_system);
    {
        static DeviceForwardParams dfp;
        for (int i = 0; i < info->num_layers; ++i)
        {
            dfp.neural_out[i] = output_forward[i];
            // dfp.blend_out[i] = output_forward_blend[i];
        }
        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_forward_params, &dfp, sizeof(dfp)));
    }

    {
        static DeviceRenderParams drp;
        drp = PrepareDeviceRenderParams();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_render_params, &drp, sizeof(drp)));
        CUDA_SYNC_CHECK_ERROR();
    }
    if (info->scene)
    {
        static DeviceTexture d_tex;
        d_tex = PrepareDeviceTexture();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_texture, &d_tex, sizeof(d_tex)));
        CUDA_SYNC_CHECK_ERROR();
    }
}


void PointRendererCache::DepthPrepassMulti(int batch, NeuralPointCloudCuda point_cloud, bool train)
{
    SAIGA_ASSERT(point_cloud);

    // ImGui::Begin("test");
    // ImGui::SliderInt("points per thread", &POINT_PER_THREAD, 1, 256);
    // ImGui::End();

    // std::cout <<"pointcloud positions"<< TensorInfo(point_cloud->t_position) << std::endl;
    {
        int image_batch_id = batch;

        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);


        int c = iDivUp(point_cloud->Size(), default_block_size * POINT_PER_THREAD);

        if (info->num_layers == 1)
        {
            ::DepthPrepassMulti<1, false, false, true>
                <<<c, default_block_size>>>(point_cloud, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 2)
        {
            ::DepthPrepassMulti<2, false, false, true>
                <<<c, default_block_size>>>(point_cloud, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 3)
        {
            ::DepthPrepassMulti<3, false, false, true>
                <<<c, default_block_size>>>(point_cloud, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 4)
        {
            ::DepthPrepassMulti<4, false, false, true>
                <<<c, default_block_size>>>(point_cloud, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 5)
        {
            ::DepthPrepassMulti<5, false, false, true>
                <<<c, default_block_size>>>(point_cloud, cam, batch, POINT_PER_THREAD, train);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}


void PointRendererCache::RenderForwardMulti(int batch, NeuralPointCloudCuda point_cloud, bool train)
{
    SAIGA_ASSERT(point_cloud);

    int image_batch_id = batch;


    auto& cam = info->images[image_batch_id];

    // SAIGA_ASSERT(info->scene->texture->texture.is_cuda());

    // std::cout << "render forwrad " << cam.w << "x" << cam.h << " " << cam.camera_index << " " << cam.image_index << "
    // "
    //           << cam.crop_transform << std::endl;

    {
        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id;
        int c          = iDivUp(point_cloud->Size(), default_block_size * POINT_PER_THREAD);
        if (info->num_layers == 1)
        {
            ::RenderForwardMulti<1, false>
                <<<c, default_block_size>>>(point_cloud, dropout, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 2)
        {
            ::RenderForwardMulti<2, false>
                <<<c, default_block_size>>>(point_cloud, dropout, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 3)
        {
            ::RenderForwardMulti<3, false>
                <<<c, default_block_size>>>(point_cloud, dropout, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 4)
        {
            ::RenderForwardMulti<4, false>
                <<<c, default_block_size>>>(point_cloud, dropout, cam, batch, POINT_PER_THREAD, train);
        }
        else if (info->num_layers == 5)
        {
            ::RenderForwardMulti<5, false>
                <<<c, default_block_size>>>(point_cloud, dropout, cam, batch, POINT_PER_THREAD, train);
        }
        else
        {
            SAIGA_EXIT_ERROR("sdf");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

__global__ void FillSceneGridWithLoss(DevicePointCloud point_cloud, ReducedImageInfo cam,
                                      StaticDeviceTensor<float, 3> forward_loss)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    //  int local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    int b_z = blockIdx.z;

    if (gx > forward_loss.size(2) || gy > forward_loss.size(1)) return;

    float grad = 0.f;
    for (int i = 0; i < 3; ++i) grad += abs(forward_loss(i, gy, gx));

    Sophus::SE3f V = d_render_params.Pose(cam.image_index);
    for (int cell_id = b_z; cell_id < point_cloud.n_cells; cell_id += blockDim.z)
    {
        vec3 bb_min, bb_len;
        float value;

        thrust::tie(bb_min, bb_len, value) = point_cloud.GetCellBB(cell_id);
        const vec3 bb_facs[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
        vec2 max_ip          = vec2(-10000000, -10000000);
        vec2 min_ip          = vec2(10000000, 1000000);
        //   vec2 mid_p;
        //   float mid_z;
        //   float mid_rp;
        //   thrust::tie(mid_p, mid_z, mid_rp) = ProjPoint(bb_min+bb_len/2, vec3(0, 1, 0), 0, cam, false);
        bool full_box = true;
        for (int i = 0; i < 8; ++i)
        {
            vec3 p = bb_min.array() + bb_len.array() * bb_facs[i].array();

            vec2 ip;
            float z;
            float radius_pixels;
            thrust::tie(ip, z, radius_pixels) = ProjPoint(p, vec3(0, 1, 0), 0, cam, false);

            // ignore complete box if one element is behind the camera -> unreliable data
            if (z <= 0)
            {
                full_box = false;
                break;
            }

            max_ip.x() = fmax(ip.x(), max_ip.x());
            max_ip.y() = fmax(ip.y(), max_ip.y());
            min_ip.x() = fmin(ip.x(), min_ip.x());
            min_ip.y() = fmin(ip.y(), min_ip.y());
        }
        // inside bb of bb
        if (full_box && gx > min_ip.x() && gx < max_ip.x() && gy > min_ip.y() && gy < max_ip.y())
        {
            // int i_grad = reinterpret_cast<int*>(&grad)[0];

            // atomicMax((int *)point_cloud.GetPointerForValueForCell(cell_id), i_grad);
            // atomicAdd(point_cloud.GetPointerForValueForCell(cell_id), i_grad);
            atomicAdd(point_cloud.GetPointerForValueForCell(cell_id), grad);
            atomicAdd(point_cloud.GetPointerForAccessCountForCell(cell_id), 1);
            // point_cloud.GetPointerForValueForCell(cell_id)[0] = cell_id;
        }
    }
}

// https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
__device__ float sign(vec2 p1, vec2 p2, vec2 p3)
{
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
}

__device__ bool PointInTriangle(vec2 pt, vec2 v1, vec2 v2, vec2 v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

__device__ bool PointInQuad(vec2 p, vec2 q1, vec2 q2, vec2 q3, vec2 q4)
{
    return PointInTriangle(p, q1, q2, q3) || PointInTriangle(p, q1, q2, q4) || PointInTriangle(p, q1, q3, q4) ||
           PointInTriangle(p, q2, q3, q4);
}


#define block_size 32
#define amount_of_warps_in_block 1  // (block_size / 32)

// constexpr int block_y_size = 1;
// scanline
__global__ void FillSceneGridWithLossRasterize(DevicePointCloud point_cloud, ReducedImageInfo cam,
                                               StaticDeviceTensor<float, 3> loss_img)
{
    // border pixels not use for loss projection (error prone)
    const int border_pixels = 32;

    // int gx        = blockIdx.x * blockDim.x + threadIdx.x;
    // int gy        = blockIdx.y * blockDim.y + threadIdx.y;
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    //  int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);
    // int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y));
    //    int index = (blockIdx.x * blockDim.x) + threadIdx.x % 32;
    int task_number_in_block = local_tid / 32;

    int block_idx = (blockIdx.y * blockDim.x + blockIdx.x) * amount_of_warps_in_block + task_number_in_block;
    // int thread_face_id = threadIdx.y;
    if (block_idx >= point_cloud.n_cells) return;
    // each warp one cell
    int thread_id_in_warp      = local_tid % 32;
    int index_of_warp_in_block = threadIdx.x / 32;
    // int index_in_block = threadIdx.x;



    //  const ivec4 quad_indices[] = {{0,2,4,1},{1,5,7,4},{2,4,7,6},{0,3,6,2},{0,1,5,3},{3,5,7,6}};

    const ivec3 tris_indices[] = {{0, 1, 4}, {0, 4, 2}, {1, 5, 7}, {1, 7, 4}, {5, 3, 6}, {5, 6, 7},
                                  {3, 0, 6}, {0, 6, 2}, {3, 0, 5}, {0, 5, 1}, {2, 4, 6}, {4, 6, 7}};

    //   vec2 mid_p;
    //   float mid_z;
    //   float mid_rp;
    //   thrust::tie(mid_p, mid_z, mid_rp) = ProjPoint(bb_min+bb_len/2, vec3(0, 1, 0), 0, cam, false);
    // bool full_box = true;
    __shared__ vec2 proj_points[amount_of_warps_in_block][8];
    __shared__ bool min_depths[amount_of_warps_in_block];
    if (thread_id_in_warp == 0) min_depths[index_of_warp_in_block] = false;
    //   __syncwarp();

    if (thread_id_in_warp < 8)
    {
        vec3 bb_min, bb_len;
        float value;

        thrust::tie(bb_min, bb_len, value) = point_cloud.GetCellBB(block_idx);
        const vec3 bb_facs[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
        // for (int i = 0; i < 8; ++i)
        {
            vec3 p = bb_min.array() + bb_len.array() * bb_facs[thread_id_in_warp].array();

            vec2 ip;
            float z;
            float radius_pixels;
            thrust::tie(ip, z, radius_pixels) = ProjPoint(p, vec3(0, 1, 0), 0, cam, false);

            // ignore complete box if one element is behind the camera -> unreliable data
            if (z <= 0)
            {
                // full_box = false;
                // return;
                min_depths[index_of_warp_in_block] = true;
            }
            proj_points[index_of_warp_in_block][thread_id_in_warp] = ip;
        }
    }
    //  __syncwarp();
    if (min_depths[index_of_warp_in_block]) return;
    vec2 max_ip = vec2(-10000000, -10000000);
    vec2 min_ip = vec2(10000000, 10000000);
    for (int i = 0; i < 8; ++i)
    {
        vec2 ip = proj_points[index_of_warp_in_block][i];

        max_ip.x() = fmax(ip.x(), max_ip.x());
        max_ip.y() = fmax(ip.y(), max_ip.y());
        min_ip.x() = fmin(ip.x(), min_ip.x());
        min_ip.y() = fmin(ip.y(), min_ip.y());
    }

    if (max_ip.x() < border_pixels || min_ip.x() >= loss_img.size(2) - border_pixels || max_ip.y() < border_pixels ||
        min_ip.y() >= loss_img.size(1) - border_pixels)
        return;
    min_ip.x() = fmax(min_ip.x(), border_pixels);
    min_ip.y() = fmax(min_ip.y(), border_pixels);
    max_ip.x() = fmin(max_ip.x(), loss_img.size(2) - border_pixels);
    max_ip.y() = fmin(max_ip.y(), loss_img.size(1) - border_pixels);


    int y_size = int(max_ip.y() - min_ip.y());
    int x_size = int(max_ip.x() - min_ip.x());
    for (int xy_c = thread_id_in_warp; xy_c < y_size * x_size; xy_c += 32)
    {
        int x_c = (xy_c % x_size) + min_ip.x();
        int y_c = (xy_c / x_size) + min_ip.y();
        //   for(int y_c = min_ip.y(); y_c<max_ip.y(); ++y_c){
        //       for(int x_c=min_ip.x(); x_c<max_ip.x(); ++x_c){
        bool is_in = false;
        for (int tris_id = 0; tris_id < 12; ++tris_id)
        //  int quad_id = face;
        {
            //       if (PointInQuad(vec2(x_c, y_c), proj_points[quad_indices[quad_id].x()],
            //                       proj_points[quad_indices[quad_id].y()], proj_points[quad_indices[quad_id].z()],
            //                       proj_points[quad_indices[quad_id].w()]))
            if (PointInTriangle(vec2(x_c, y_c), proj_points[index_of_warp_in_block][tris_indices[tris_id].x()],
                                proj_points[index_of_warp_in_block][tris_indices[tris_id].y()],
                                proj_points[index_of_warp_in_block][tris_indices[tris_id].z()]))
            {
                is_in = true;
                // break;
            }
        }
        if (is_in)
        {
            //  return;
            float grad = 0.f;
            for (int i = 0; i < loss_img.size(0); ++i) grad += abs(loss_img(i, y_c, x_c));

            atomicAdd(point_cloud.GetPointerForValueForCell(block_idx), grad);
            atomicAdd(point_cloud.GetPointerForAccessCountForCell(block_idx), 1);
        }
    }
}

void PointRendererCache::FillSceneGridWithLoss(int batch, NeuralPointCloudCuda point_cloud,
                                               std::vector<ReducedImageInfo> images, torch::Tensor loss_img)
{
    // using namespace std::chrono_literals;
    // std::this_thread::sleep_for(10000ms);
    int image_batch_id = batch;
    auto cam           = images[image_batch_id];  // info->images[image_batch_id];;


    // std::cout << TensorInfo(loss_img) << std::endl;
    // std::cout << "FILL grid start" << std::endl;
    // int bx  = iDivUp(loss_img.size(2), 16);
    // int by  = iDivUp(loss_img.size(1), 16);
    //::FillSceneGridWithLoss<<<dim3(bx, by, 128), dim3(16, 16, 1)>>>(point_cloud, cam,loss_img);

    int size_of_cells         = point_cloud->t_cell_bb_min.size(0);
    const int cells_per_block = block_size / 32;

    auto grid_dim  = dim3(std::ceil(size_of_cells) / cells_per_block, 1, 1);
    auto block_dim = dim3(16, 2, 1);

    // std::cout <<size_of_cells<< ": " << grid_dim.x <<" - "<< grid_dim.y <<" - "<< grid_dim.z <<" --- " << block_dim.x
    // <<" - "<< block_dim.y <<" - "<< block_dim.z <<" - "<< std::endl;

    ::FillSceneGridWithLossRasterize<<<grid_dim, block_dim>>>(point_cloud, cam, loss_img);

    CUDA_SYNC_CHECK_ERROR();
    //  std::cout << "FILL grid end" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_ELEMENTS_PER_PIXEL 16 * 512


template <int num_layers, bool bilinear>
__global__ void CountingPrepassMulti(DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int batch)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)

    Sophus::SE3f _V = d_render_params.Pose(cam.image_index);
    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
#if 0
    for (int iter = 0, point_id = grid.thread_rank()*point_per_thread; point_id < point_cloud.Size() && iter < point_per_thread; ++iter, point_id += 1)
    {
    for (int p_nr = 0; p_nr < points_per_thread; ++p_nr)
    {
        int point_id = threadNumInBlock + p_nr * threadsPerBlock + blockNumInGrid * threadsPerBlock * points_per_thread;
#endif
        if (point_id >= point_cloud.Size()) continue;

        bool drop_out = dropout_p[point_id] == 1;
        if (drop_out) continue;

        int conf_id = point_cloud.GetIndex(point_id);
        if (discard_point_for_confidence(conf_id)) continue;

        vec2 ip;
        float z;
        float radius_pixels;

        {
            vec3 position;
            vec3 normal;
            vec2 image_p_a;
            float drop_out_radius;

            CUDA_KERNEL_ASSERT(cam.image_index >= 0);
            // Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            //  return;

            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);
            thrust::tie(ip, z, radius_pixels) = GetProjectedPoint(position, normal, drop_out_radius, point_id, cam);

            if (z == 0) continue;
        }

#pragma unroll
        for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
        {
            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                break;
            }

            if (bilinear)
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                for (int y_j = 0; y_j <= 1; ++y_j)
                {
                    for (int x_i = 0; x_i <= 1; ++x_i)
                    {
                        ivec2 p_imgi = p_rd + ivec2(x_i, y_j);
                        if (!d_render_params.counting[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;

                        // {
                        //     float* dst_pos = &(d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0)));
                        //     int i_depth    = reinterpret_cast<int*>(&z)[0];
                        //     atomicMin((int*)dst_pos, i_depth);
                        // }
                        {
                            int* dst_pos = &(d_render_params.counting[layer](batch, 0, p_imgi(1), p_imgi(0)));
                            atomicAdd(dst_pos, 1);
                        }
                    }
                }
            }
            else
            {
                ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

                // Check in image
                if (!d_render_params.counting[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;

                {
                    float* dst_pos = &(d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0)));
                    int i_depth    = reinterpret_cast<int*>(&z)[0];
                    atomicMin((int*)dst_pos, i_depth);
                }
                {
                    int* dst_pos = &(d_render_params.counting[layer](batch, 0, p_imgi(1), p_imgi(0)));
                    atomicAdd(dst_pos, 1);
                }
            }
        }
    }
}

void PointRendererCache::CountingPrepassMulti(int batch, NeuralPointCloudCuda point_cloud, bool train)
{
    SAIGA_ASSERT(point_cloud);


    // ImGui::Begin("test");
    // ImGui::SliderInt("points per thread", &POINT_PER_THREAD, 1, 256);
    // ImGui::End();

    static constexpr int points_per_thread_counting_pass = 16;

    // std::cout <<"pointcloud positions"<< TensorInfo(point_cloud->t_position) << std::endl;
    {
        int image_batch_id = batch;

        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id;


        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);


        int c = iDivUp(point_cloud->Size(), default_block_size * points_per_thread_counting_pass);

        if (info->num_layers == 1)
        {
            ::CountingPrepassMulti<1, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 2)
        {
            ::CountingPrepassMulti<2, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 3)
        {
            ::CountingPrepassMulti<3, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 4)
        {
            ::CountingPrepassMulti<4, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 5)
        {
            ::CountingPrepassMulti<5, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::CountingPrepassMultiBilinear(int batch, NeuralPointCloudCuda point_cloud, bool train)
{
    SAIGA_ASSERT(point_cloud);


    // ImGui::Begin("test");
    // ImGui::SliderInt("points per thread", &POINT_PER_THREAD, 1, 256);
    // ImGui::End();

    static constexpr int points_per_thread_counting_pass = 16;

    // std::cout <<"pointcloud positions"<< TensorInfo(point_cloud->t_position) << std::endl;
    {
        int image_batch_id = batch;

        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id;


        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);


        int c = iDivUp(point_cloud->Size(), default_block_size * points_per_thread_counting_pass);

        if (info->num_layers == 1)
        {
            ::CountingPrepassMulti<1, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 2)
        {
            ::CountingPrepassMulti<2, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 3)
        {
            ::CountingPrepassMulti<3, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 4)
        {
            ::CountingPrepassMulti<4, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 5)
        {
            ::CountingPrepassMulti<5, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

template <int num_layers, bool bilinear>
__global__ void CollectMulti(DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int batch)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)

    Sophus::SE3f _V = d_render_params.Pose(cam.image_index);
    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
#if 0

    for (int iter = 0, point_id = grid.thread_rank()*point_per_thread; point_id < point_cloud.Size() && iter < point_per_thread; ++iter, point_id += 1)
    {
    for (int p_nr = 0; p_nr < points_per_thread; ++p_nr)
    {
        int point_id = threadNumInBlock + p_nr * threadsPerBlock + blockNumInGrid * threadsPerBlock * points_per_thread;
#endif
        if (point_id >= point_cloud.Size()) continue;


        bool drop_out = dropout_p[point_id] == 1;
        if (drop_out) continue;

        int conf_id = point_cloud.GetIndex(point_id);
        if (discard_point_for_confidence(conf_id)) continue;
        float confidence = d_texture.points_confidence_value(0, conf_id);

        vec2 ip;
        float z;
        float radius_pixels;

        {
            vec3 position;
            vec3 normal;
            vec2 image_p_a;
            float drop_out_radius;

            CUDA_KERNEL_ASSERT(cam.image_index >= 0);
            Sophus::SE3f V = d_render_params.Pose(cam.image_index);
            // return;

            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);
            thrust::tie(ip, z, radius_pixels) = GetProjectedPoint(position, normal, drop_out_radius, point_id, cam);

            if (z == 0) continue;
        }

        auto get_next_free_pos_in_buffer = [&](int x, int y, int layer)
        {
            // get start of chunk for this pixel
            int offset_in_buffer = d_alpha_comp_params.scanned_countings[layer](0, y, x);
            // get pos in chunk
            int* atomic_c_pos   = &(d_render_params.per_image_atomic_counters[layer](batch, 0, y, x));
            int pos_in_pixel    = atomicAdd(atomic_c_pos, 1);
            int full_buffer_pos = offset_in_buffer + pos_in_pixel;
            return full_buffer_pos;
        };


#pragma unroll
        for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
        {
            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                break;
            }
            int i_depth = reinterpret_cast<int*>(&z)[0];

            if (!bilinear)
            {
                ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

                // Check in image
                if (!d_render_params.per_image_atomic_counters[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;


                int full_buffer_pos = get_next_free_pos_in_buffer(p_imgi(0), p_imgi(1), layer);

                d_alpha_comp_params.collections[layer](0, full_buffer_pos) = i_depth;
                d_alpha_comp_params.collections[layer](1, full_buffer_pos) = point_id;
            }
            else
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                vec4 blend_vec = compute_blending_fac(ip);
#pragma unroll
                for (int y_j = 0; y_j <= 1; ++y_j)
                {
#pragma unroll
                    for (int x_i = 0; x_i <= 1; ++x_i)
                    {
                        ivec2 p_imgi = p_rd + ivec2(x_i, y_j);
                        if (!d_render_params.counting[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;

                        int full_buffer_pos = get_next_free_pos_in_buffer(p_imgi(0), p_imgi(1), layer);

                        d_alpha_comp_params.collections[layer](0, full_buffer_pos) = i_depth;
                        d_alpha_comp_params.collections[layer](1, full_buffer_pos) = full_buffer_pos;

                        float f_point_id = reinterpret_cast<float*>(&point_id)[0];

                        int index_blend_vec     = y_j * 2 + x_i;
                        float f_index_blend_vec = reinterpret_cast<float*>(&index_blend_vec)[0];

                        // d_alpha_comp_params.per_point_data[layer](1, full_buffer_pos) = blend_vec[index_blend_vec];
                        // d_alpha_comp_params.per_point_data[layer](2, full_buffer_pos) = confidence *
                        // blend_vec[index_blend_vec];
                        d_alpha_comp_params.per_point_data[layer](full_buffer_pos, 0) = f_point_id;
                        d_alpha_comp_params.per_point_data[layer](full_buffer_pos, 1) = ip.x();
                        d_alpha_comp_params.per_point_data[layer](full_buffer_pos, 2) = ip.y();
                        d_alpha_comp_params.per_point_data[layer](full_buffer_pos, 3) = f_index_blend_vec;

                        // vec4 data_to_store = vec4(f_point_id, ip.x(),ip.y(),f_index_blend_vec);
                        // vec4 *loc = (vec4*) &d_alpha_comp_params.per_point_data[layer](full_buffer_pos, 0);
                        //*loc = data_to_store;
                    }
                }
            }
        }
    }
}



void PointRendererCache::UploadCollectionBuffers(std::vector<torch::Tensor> collection_buffers,
                                                 std::vector<torch::Tensor> data_buffers, int batch_num,
                                                 std::vector<torch::Tensor> atomics)
{
    static DeviceAlphaCompositionParams dacp;
    // buffer.size == layers used
    for (int i = 0; i < collection_buffers.size(); ++i)
    {
        dacp.collections[i] = collection_buffers[i];
        if (atomics.size() > 0) dacp.ticket_counter[i] = atomics[i];
        if (!data_buffers.empty()) dacp.per_point_data[i] = data_buffers[i];
    }
    for (int i = 0; i < info->num_layers; ++i)
    {
        dacp.scanned_countings[i] = layers_cuda[i].scanned_counting[batch_num];
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_alpha_comp_params, &dacp, sizeof(dacp)));
}


void PointRendererCache::CollectMulti(int batch, NeuralPointCloudCuda point_cloud,
                                      std::vector<torch::Tensor> collectionbuffers,
                                      std::vector<torch::Tensor> data_buffer, bool train)
{
    SAIGA_ASSERT(point_cloud);

    // ImGui::Begin("test");
    // ImGui::SliderInt("points per thread", &POINT_PER_THREAD, 1, 256);
    // ImGui::End();

    static constexpr int points_per_thread_collection_pass = 16;


    // std::cout <<"pointcloud positions"<< TensorInfo(point_cloud->t_position) << std::endl;
    {
        int image_batch_id = batch;

        UploadCollectionBuffers(collectionbuffers, data_buffer, batch);

        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id;


        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);


        //     std::cout << cam.crop_transform << std::endl;

        int c = iDivUp(point_cloud->Size(), default_block_size * points_per_thread_collection_pass);

        if (info->num_layers == 1)
        {
            ::CollectMulti<1, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 2)
        {
            ::CollectMulti<2, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 3)
        {
            ::CollectMulti<3, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 4)
        {
            ::CollectMulti<4, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 5)
        {
            ::CollectMulti<5, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::CollectMultiBilinear(int batch, NeuralPointCloudCuda point_cloud,
                                              std::vector<torch::Tensor> collectionbuffers,
                                              std::vector<torch::Tensor> data_buffer, bool train)
{
    SAIGA_ASSERT(point_cloud);

    // ImGui::Begin("test");
    // ImGui::SliderInt("points per thread", &POINT_PER_THREAD, 1, 256);
    // ImGui::End();

    static constexpr int points_per_thread_collection_pass = 16;


    // std::cout <<"pointcloud positions"<< TensorInfo(point_cloud->t_position) << std::endl;
    {
        int image_batch_id = batch;

        UploadCollectionBuffers(collectionbuffers, data_buffer, batch);

        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id;


        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);


        //     std::cout << cam.crop_transform << std::endl;

        int c = iDivUp(point_cloud->Size(), default_block_size * points_per_thread_collection_pass);

        if (info->num_layers == 1)
        {
            ::CollectMulti<1, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 2)
        {
            ::CollectMulti<2, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 3)
        {
            ::CollectMulti<3, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 4)
        {
            ::CollectMulti<4, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 5)
        {
            ::CollectMulti<5, true><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::SortMulti(int batch, std::vector<torch::Tensor> collectionbuffers, bool train)
{
    SAIGA_ASSERT(collectionbuffers.size() == info->num_layers);
    std::vector<torch::Tensor> atomics;
    for (int i = 0; i < info->num_layers; ++i)
        atomics.push_back(torch::zeros({int(collectionbuffers.size())},
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)));

    UploadCollectionBuffers(collectionbuffers, std::vector<torch::Tensor>(), batch, atomics);

    for (int i = 0; i < info->num_layers; ++i)
    {
        auto& l          = layers_cuda[i];
        int num_elements = l.size.x() * l.size.y();

        CHECK((l.per_image_atomic_counters == l.counting).all().item<bool>());

        auto l_counting_element        = l.counting.slice(0, batch, batch + 1);
        auto l_scannedcounting_element = l.scanned_counting.slice(0, batch, batch + 1);

        SegmentedSortBitonicHelper2(l_counting_element.view({-1}), l_scannedcounting_element.view({-1}),
                                    collectionbuffers[i]);
    }
}


__global__ void BlendMulti(DevicePointCloud point_cloud, StaticDeviceTensor<float, 3> out_neural_image,
                           float* background_color, int batch, int layer, bool environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= out_neural_image.size(2) || gy >= out_neural_image.size(1)) return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params.scanned_countings[layer](0, gy, gx);

    // size of chunk:
    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);

    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    float alpha_dest = 1.f;
    float color_out[4];
    CUDA_KERNEL_ASSERT(d_render_params.num_texture_channels <= 4);

    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) color_out[ci] = 0.f;

    // blend all together
    for (int i = 0; i < size_of_chunk; ++i)
    {
        int full_buffer_pos = offset_in_buffer + i;
        // do blend
        //  int i_depth = d_alpha_comp_params.collections[layer](0,full_buffer_pos);
        int point_id      = d_alpha_comp_params.collections[layer](1, full_buffer_pos);
        int texture_index = point_cloud.GetIndex(point_id);

        CUDA_KERNEL_ASSERT(texture_index >= 0);
        CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

        float confidence_val = d_texture.points_confidence_value(0, texture_index);

        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
        {
            float color   = d_texture.in_texture(ci, texture_index);
            color_out[ci] = compute_blend(alpha_dest, confidence_val, color, color_out[ci]);
        }
        alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);
    }

    // blend background (opacity 1) and write out
    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
    {
        if (!environment_map) color_out[ci] = compute_blend(alpha_dest, 1.f, background_color[ci], color_out[ci]);
        d_forward_params.neural_out[layer](batch, ci, gy, gx) = color_out[ci];
    }

    auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, gy, gx));
    atomicAdd(dst_pos_weight, alpha_dest);
}

void PointRendererCache::BlendMulti(int batch, DevicePointCloud point_cloud,
                                    std::vector<torch::Tensor> collection_buffers,
                                    std::vector<torch::Tensor> data_buffer, torch::Tensor background_color, bool train,
                                    bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    SAIGA_ASSERT(output_forward.size() == info->num_layers);

    UploadCollectionBuffers(collection_buffers, data_buffer, batch);

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

        // auto weights = l.BatchViewWeights(batch);
        ::BlendMulti<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, in_out_neural_image, background, batch, i,
                                                           use_environment_map);
    }

    CUDA_SYNC_CHECK_ERROR();
}


__global__ void BlendMultiFuzzy(DevicePointCloud point_cloud, StaticDeviceTensor<float, 3> out_neural_image,
                                float* background_color, int batch, int layer, bool environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    const int MAX_DESCRIPTORS = 4;

    if (gx >= out_neural_image.size(2) || gy >= out_neural_image.size(1)) return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params.scanned_countings[layer](0, gy, gx);

    // size of chunk:
    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);

    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    float alpha_dest = 1.f;
    float color_out[MAX_DESCRIPTORS];
    CUDA_KERNEL_ASSERT(d_render_params.num_texture_channels <= 4);

    float accumulation_last_min_d = 0.f;
    float accumulation_desc[MAX_DESCRIPTORS];
    float accumulation_conf_val = 0.f;
    int accumulation_num_elems  = 0;

    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
    {
        accumulation_desc[ci] = 0.f;
        color_out[ci]         = 0.f;
    }

    auto sample_depth = [&](int buffer_pos) -> float
    {
        int depth_i = d_alpha_comp_params.collections[layer](0, buffer_pos);
        float depth = reinterpret_cast<float*>(&depth_i)[0];
        return depth;
    };
    auto accumulateCollected = [&]()
    {
        float confidence_val = accumulation_conf_val / float(accumulation_num_elems);
        // blend
        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
        {
            float color   = accumulation_desc[ci] / float(accumulation_num_elems);
            color_out[ci] = compute_blend(alpha_dest, confidence_val, color, color_out[ci]);
        }
        alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);
    };

    // blend all together fuzzy
    for (int i = 0; i < size_of_chunk; ++i)
    {
        int full_buffer_pos = offset_in_buffer + i;
        // do blend
        //  int i_depth = d_alpha_comp_params.collections[layer](0,full_buffer_pos);
        int point_id      = d_alpha_comp_params.collections[layer](1, full_buffer_pos);
        int texture_index = point_cloud.GetIndex(point_id);
        float d           = sample_depth(full_buffer_pos);

        CUDA_KERNEL_ASSERT(texture_index >= 0);
        CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

        if (accumulation_last_min_d == 0.f) accumulation_last_min_d = d;

        if (!(d - accumulation_last_min_d < d_render_params.depth_accept_blend))
        {
            {
                accumulateCollected();
            }
            {
                // reset state
                accumulation_last_min_d = -100000.f;
                accumulation_conf_val   = 0.f;
                accumulation_num_elems  = 0;
                for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                {
                    accumulation_desc[ci] = 0.f;
                }
            }
        }
        // accumulate fuzzy
        {
            float confidence_val = d_texture.points_confidence_value(0, texture_index);
            accumulation_conf_val += confidence_val;
            for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
            {
                float color = d_texture.in_texture(ci, texture_index);
                accumulation_desc[ci] += color;
            }
            ++accumulation_num_elems;
        }
    }
    if (accumulation_num_elems > 0)
    {
        // accumulate rest
        accumulateCollected();
    }

    // blend background (opacity 1) and write out
    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
    {
        if (!environment_map) color_out[ci] = compute_blend(alpha_dest, 1.f, background_color[ci], color_out[ci]);
        d_forward_params.neural_out[layer](batch, ci, gy, gx) = color_out[ci];
    }

    auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, gy, gx));
    atomicAdd(dst_pos_weight, alpha_dest);
}

void PointRendererCache::BlendMultiFuzzy(int batch, DevicePointCloud point_cloud,
                                         std::vector<torch::Tensor> collection_buffers,
                                         std::vector<torch::Tensor> data_buffer, torch::Tensor background_color,
                                         bool train, bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    SAIGA_ASSERT(output_forward.size() == info->num_layers);

    UploadCollectionBuffers(collection_buffers, data_buffer, batch);

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

        // auto weights = l.BatchViewWeights(batch);
        ::BlendMultiFuzzy<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, in_out_neural_image, background, batch, i,
                                                                use_environment_map);
    }

    CUDA_SYNC_CHECK_ERROR();
}


__global__ void BlendMultiBilinear(DevicePointCloud point_cloud, StaticDeviceTensor<float, 3> out_neural_image,
                                   float* background_color, int batch, int layer, bool environment_map)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= out_neural_image.size(2) || gy >= out_neural_image.size(1)) return;

    // get start of chunk for this pixel
    int offset_in_buffer = d_alpha_comp_params.scanned_countings[layer](0, gy, gx);

    // size of chunk:
    int size_of_chunk = d_render_params.per_image_atomic_counters[layer](batch, 0, gy, gx);

    if (d_render_params.debug_max_list_length > 0)
        size_of_chunk = min(d_render_params.debug_max_list_length, size_of_chunk);

    float alpha_dest = 1.f;
    float color_out[4];
    CUDA_KERNEL_ASSERT(d_render_params.num_texture_channels <= 4);

    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci) color_out[ci] = 0.f;
    /*
        auto get_point_id = [&](int buf_id)
        {
            //int buf_id          = d_alpha_comp_params.collections[layer](1, pos);
            float p_id_as_float = d_alpha_comp_params.per_point_data[layer](buf_id, 0);
            int p_id            = reinterpret_cast<int*>(&p_id_as_float)[0];
            return p_id;
        };
        auto get_subpixel_weight = [&](int buf_id)
        {
           // int buf_id = d_alpha_comp_params.collections[layer](1, pos);

            // vec4 data = ((vec4*)&d_alpha_comp_params.per_point_data[layer](buf_id, 0))[0];
            //  float uv_x                   = data.y();
            //  float uv_y                   = data.z();
            //  float p_blend_index_as_float = data.w();
            float uv_x                   = d_alpha_comp_params.per_point_data[layer](buf_id, 1);
            float uv_y                   = d_alpha_comp_params.per_point_data[layer](buf_id, 2);
            float p_blend_index_as_float = d_alpha_comp_params.per_point_data[layer](buf_id, 3);
            int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
            vec4 blend_facs              = compute_blending_fac(vec2(uv_x, uv_y));
            return blend_facs[blend_index];
        };*/

    auto get_point_id_and_subpixel_weights = [&](int pos)
    {
        int buf_id                   = d_alpha_comp_params.collections[layer](1, pos);
        float p_id_as_float          = d_alpha_comp_params.per_point_data[layer](buf_id, 0);
        float uv_x                   = d_alpha_comp_params.per_point_data[layer](buf_id, 1);
        float uv_y                   = d_alpha_comp_params.per_point_data[layer](buf_id, 2);
        float p_blend_index_as_float = d_alpha_comp_params.per_point_data[layer](buf_id, 3);
        int blend_index              = reinterpret_cast<int*>(&p_blend_index_as_float)[0];
        vec4 blend_facs              = compute_blending_fac(vec2(uv_x, uv_y));
        return vec2(p_id_as_float, blend_facs[blend_index]);
    };

    // blend all together
    for (int i = 0; i < size_of_chunk; ++i)
    {
        int full_buffer_pos = offset_in_buffer + i;
        // do blend
        //  int i_depth = d_alpha_comp_params.collections[layer](0,full_buffer_pos);
        // int buf_id          = d_alpha_comp_params.collections[layer](1, pos);

        vec2 stored_data   = get_point_id_and_subpixel_weights(full_buffer_pos);
        int point_id       = reinterpret_cast<int*>(&stored_data.x())[0];
        float bilinear_fac = stored_data.y();


        // int point_id      = get_point_id(full_buffer_pos);
        int texture_index = point_cloud.GetIndex(point_id);

        // CUDA_KERNEL_ASSERT(texture_index >= 0);
        //  if (texture_index >= d_texture.in_texture.sizes[1]) printf("%d\n", texture_index);
        // CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);

        // float bilinear_fac = get_subpixel_weight(full_buffer_pos);

        float confidence_val = bilinear_fac * d_texture.points_confidence_value(0, texture_index);

        for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
        {
            float color   = d_texture.in_texture(ci, texture_index);
            color_out[ci] = compute_blend(alpha_dest, confidence_val, color, color_out[ci]);
        }
        alpha_dest = compute_new_alphadest(alpha_dest, confidence_val);
        if (alpha_dest < 0.001) break;
    }


    // blend background (opacity 1) and write out
    for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
    {
        if (alpha_dest >= 0.001)
        {
            if (!environment_map) color_out[ci] = compute_blend(alpha_dest, 1.f, background_color[ci], color_out[ci]);
        }
        d_forward_params.neural_out[layer](batch, ci, gy, gx) = color_out[ci];
    }


    auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, gy, gx));
    atomicAdd(dst_pos_weight, alpha_dest);
}

void PointRendererCache::BlendMultiBilinear(int batch, DevicePointCloud point_cloud,
                                            std::vector<torch::Tensor> collection_buffers,
                                            std::vector<torch::Tensor> data_buffer, torch::Tensor background_color,
                                            bool train, bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    SAIGA_ASSERT(output_forward.size() == info->num_layers);

    UploadCollectionBuffers(collection_buffers, data_buffer, batch);

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

        // auto weights = l.BatchViewWeights(batch);
        ::BlendMultiBilinear<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(point_cloud, in_out_neural_image, background, batch,
                                                                   i, use_environment_map);
    }

    CUDA_SYNC_CHECK_ERROR();
}