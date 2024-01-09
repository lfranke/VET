/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Dataset.h"

#include "saiga/vision/torch/RandomCrop.h"


TemplatedImage<vec2> InitialUVImage(int h, int w)
{
    TemplatedImage<vec2> uv_target;
    uv_target.create(h, w);


    for (int i : uv_target.rowRange())
    {
        for (int j : uv_target.colRange())
        {
            // int y = h - i - 1;
            vec2 texel(j, i);
            texel.x() /= (w - 1);
            texel.y() /= (h - 1);
            vec2 centered_uv = (vec2(texel) - vec2(0.5, 0.5)) * 2;

            uv_target(i, j) = centered_uv;
        }
    }

    SAIGA_ASSERT(uv_target.getImageView().isFinite());
    return uv_target;
}


SceneDataTrainSampler::SceneDataTrainSampler(std::shared_ptr<SceneData> dataset, std::vector<int> indices,
                                             bool down_scale, ivec2 crop_size, int inner_batch_size,
                                             bool use_image_mask, bool crop_rotate)
    : inner_batch_size(inner_batch_size),
      scene(dataset),
      indices(indices),
      down_scale(down_scale),
      use_image_mask(use_image_mask),
      crop_rotate(crop_rotate)
{
    // image_size_input = ivec2(dataset->scene_cameras.front().w, dataset->scene_cameras.front().h);
    for (auto& cam : dataset->scene_cameras)
    {
        // all cameras must have the same image size
        // SAIGA_ASSERT(cam.w == image_size_input(0));
        // SAIGA_ASSERT(cam.h == image_size_input(1));
        // ivec2 wh = ivec2(cam.w * scene->dataset_params.render_scale, cam.h * scene->dataset_params.render_scale);
        // std::cout << "Train Sampler wh:" << wh.x() << "x" << wh.y() << std::endl;

        // image_size_input.push_back(wh);
        image_size_input.push_back({cam.w, cam.h});

        if (down_scale)
        {
            SAIGA_ASSERT(crop_size(0) > 0);
            image_size_crop.push_back(crop_size);
        }
        else
        {
            image_size_crop.push_back({cam.w, cam.h});
        }
        uv_target.push_back(InitialUVImage(cam.h, cam.w));
    }
}


template <typename T>
void warpPerspective(ImageView<T> src, ImageView<T> dst, IntrinsicsPinholef dst_2_src, Matrix<float, 2, 2> crop_rot)
{
    for (auto y : dst.rowRange())
    {
        for (auto x : dst.colRange())
        {
            vec2 p(x, y);
            vec2 h = vec2(dst.cols / 2, dst.rows / 2);
            p -= h;
            p = crop_rot.inverse() * p;
            p += h;

            vec2 ip  = dst_2_src.normalizedToImage(p);
            float dx = ip(0);
            float dy = ip(1);
            if (src.inImage(dy, dx))
            {
                dst(y, x) = src.inter(dy, dx);
            }
        }
    }
}


template <typename T>
void warpPerspectiveNearest(ImageView<T> src, ImageView<T> dst, IntrinsicsPinholef dst_2_src)
{
    for (auto y : dst.rowRange())
    {
        for (auto x : dst.colRange())
        {
            vec2 p(x, y);
            vec2 ip = dst_2_src.normalizedToImage(p);
            int dx  = round(ip(0));
            int dy  = round(ip(1));
            if (src.inImage(dy, dx))
            {
                dst(y, x) = src(dy, dx);
            }
        }
    }
}

std::vector<NeuralTrainData> SceneDataTrainSampler::Get(int __index)
{
    static thread_local bool init = false;
    static std::atomic_int count  = 2346;
    if (!init)
    {
        int x = count.fetch_add(142);
        Saiga::Random::setSeed(98769264 * x);
    }

    long actual_index = indices[__index];
    const auto fd     = scene->Frame(actual_index);
    int camera_id     = fd.camera_index;


    // std::cout << scene->dataset_params.image_dir + "/" + fd.target_file << std::endl;
    SAIGA_ASSERT(std::filesystem::exists(scene->dataset_params.image_dir + "/" + fd.target_file));
    Saiga::TemplatedImage<ucvec3> img_gt_large(scene->dataset_params.image_dir + "/" + fd.target_file);

    Saiga::TemplatedImage<unsigned char> img_mask_large;
    if (use_image_mask)
    {
        auto f = scene->dataset_params.mask_dir + "/" + fd.mask_file;
        SAIGA_ASSERT(std::filesystem::exists(f));
        bool ret = img_mask_large.load(f);
        if (!ret)
        {
            SAIGA_EXIT_ERROR("could not load mask image " + f);
        }
    }

    // inner_sample_size = 1;
    // inner_batch_size = 20;

    std::vector<NeuralTrainData> result;
    result.reserve(inner_batch_size);

    auto crops =
        RandomImageCrop(inner_batch_size, inner_sample_size, image_size_input[camera_id], image_size_crop[camera_id],
                        prefere_border, random_translation, sample_gaussian, min_max_zoom);
    SAIGA_ASSERT(crops.size() == inner_batch_size);

    for (int ix = 0; ix < inner_batch_size; ++ix)
    {
        NeuralTrainData pd = std::make_shared<TorchFrameData>();

        pd->img.crop_transform = IntrinsicsPinholef();

        pd->img.h = image_size_crop[camera_id](1);
        pd->img.w = image_size_crop[camera_id](0);

        Matrix<float, 2, 2> crop_rot;
        crop_rot.setZero();
        crop_rot(0, 0) = 1;
        crop_rot(1, 1) = 1;
        if (crop_rotate)
        {
            float rand_angle = Random::sampleDouble(-M_PI, M_PI);
            crop_rot(0, 0)   = std::cos(rand_angle);
            crop_rot(0, 1)   = -std::sin(rand_angle);
            crop_rot(1, 0)   = std::sin(rand_angle);
            crop_rot(1, 1)   = std::cos(rand_angle);
        }
        pd->img.crop_rotation = crop_rot;
        float zoom            = 1;
        if (down_scale)
        {
            pd->img.crop_transform = crops[ix];
            zoom                   = pd->img.crop_transform.fx;

            Saiga::TemplatedImage<ucvec3> gt_crop(image_size_crop[camera_id].y(), image_size_crop[camera_id].x());
            Saiga::TemplatedImage<vec2> uv_crop(image_size_crop[camera_id].y(), image_size_crop[camera_id].x());

            gt_crop.makeZero();
            uv_crop.makeZero();

            auto dst_2_src = pd->img.crop_transform.inverse();

            warpPerspective(img_gt_large.getImageView(), gt_crop.getImageView(), dst_2_src, crop_rot);
            warpPerspective(uv_target[camera_id].getImageView(), uv_crop.getImageView(), dst_2_src, crop_rot);

            pd->target = ImageViewToTensor(gt_crop.getImageView());
            pd->uv     = ImageViewToTensor(uv_crop.getImageView());

            TemplatedImage<unsigned char> mask(pd->img.h, pd->img.w);
            for (int i : mask.rowRange())
            {
                for (int j : mask.colRange())
                {
                    if (uv_crop(i, j).isZero() && gt_crop(i, j).isZero())
                    {
                        mask(i, j) = 0;
                    }
                    else
                    {
                        mask(i, j) = 255;
                    }
                }
            }
            pd->target_mask = ImageViewToTensor(mask.getImageView());

            if (use_image_mask)
            {
                Saiga::TemplatedImage<unsigned char> mask_crop(image_size_crop[camera_id].y(),
                                                               image_size_crop[camera_id].x());
                warpPerspective(img_mask_large.getImageView(), mask_crop.getImageView(), dst_2_src, crop_rot);
                pd->target_mask = pd->target_mask * ImageViewToTensor(mask_crop.getImageView());
            }

            SAIGA_ASSERT(uv_crop.getImageView().isFinite());
        }
        else
        {
            pd->target = ImageViewToTensor(img_gt_large.getImageView());
            pd->uv     = ImageViewToTensor(uv_target[camera_id].getImageView());

            TemplatedImage<unsigned char> mask(pd->img.h, pd->img.w);
            mask.getImageView().set(255);
            pd->target_mask = ImageViewToTensor(mask.getImageView());

            if (use_image_mask)
            {
                pd->target_mask = pd->target_mask * ImageViewToTensor(img_mask_large.getImageView());
            }
        }

        float min_mask_value = 0.1;
        pd->target_mask      = pd->target_mask.clamp_min(min_mask_value);

        long camera_index         = actual_index;
        pd->img.camera_model_type = scene->dataset_params.camera_model;
        pd->img.image_index       = actual_index;
        pd->img.camera_index      = fd.camera_index;
        pd->camera_index = torch::from_blob(&camera_index, {1}, torch::TensorOptions().dtype(torch::kLong)).cuda();
        pd->scale        = torch::from_blob(&zoom, {1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).cuda();
        pd->timestep =
            torch::from_blob(&camera_index, {1}, torch::TensorOptions().dtype(torch::kLong)).cuda().to(torch::kFloat);
        pd->to(torch::kCUDA);
        result.push_back(std::move(pd));
    }


    // sample map debug image
    // Test if the random intrinsics provide a good cover of the large input image
#if 0
    std::cout << "create sample map debug image" << std::endl;

    TemplatedImage<ucvec3> input(img_gt_large);
    // input.makeZero();

    ucvec3 color(255, 0, 0);
    int i = 0;
    for (auto td : result)
    {
        std::cout << "crop transform: " << td->img.crop_transform << std::endl;
        std::cout << "crop rot: " << td->img.crop_rotation << std::endl;
        std::vector<vec2> corners = {vec2(0, 0), vec2(image_size_crop[camera_id](0) - 1, 0),
                                     vec2(image_size_crop[camera_id](0) - 1, image_size_crop[camera_id](1) - 1),
                                     vec2(0, image_size_crop[camera_id](1) - 1)};
        std::cout << "Image size      " << image_size_input[camera_id].transpose() << std::endl;
        std::cout << "Image size crop " << image_size_crop[camera_id].transpose() << std::endl;

        auto rot_p = [&](vec2 point, vec2 center)
        {
            point -= center;
            point = td->img.crop_rotation.inverse() * point;
            point += center;
            return point;
        };
        vec2 center = image_size_crop[camera_id].cast<float>() * 0.5f;
        vec2 cen    = td->img.crop_transform.inverse().normalizedToImage(center);
        cen         = cen.array().round();

        for (auto& c : corners)
        {
            auto o = c;
            c      = td->img.crop_transform.inverse().normalizedToImage(c);
            c      = c.array().round();
            c      = rot_p(c, cen);
            ImageDraw::drawCircle(input.getImageView(), c, 10, color);
            //          std::cout << "Corner " << c.transpose() << std::endl;
        }

        ImageDraw::drawCircle(input.getImageView(), cen, 5, color);
        //      std::cout << "Center " << c.transpose() << std::endl;

        ImageDraw::drawLineBresenham(input.getImageView(), corners[0], corners[1], color);
        ImageDraw::drawLineBresenham(input.getImageView(), corners[1], corners[2], color);
        ImageDraw::drawLineBresenham(input.getImageView(), corners[2], corners[3], color);
        ImageDraw::drawLineBresenham(input.getImageView(), corners[3], corners[0], color);

        TensorToImage<ucvec3>(td->target).save("debug/crop_" + std::to_string(i) + "_target.jpg");
        TensorToImage<unsigned char>(td->target_mask).save("debug/crop_" + std::to_string(i) + "_mask.png");
        // TensorToImage<unsigned char>(td->classes).save("debug/crop_" + std::to_string(i) + "_classes.png");

        i++;
    }


    input.save("debug/sample_map" + std::to_string(__index) + ".png");
    img_gt_large.save("debug/gt.jpg");
//    exit(0);
#endif
    return result;
}

torch::MultiDatasetSampler::MultiDatasetSampler(std::vector<uint64_t> _sizes, int batch_size, bool shuffle)
    : sizes(_sizes), batch_size(batch_size)
{
    std::vector<std::vector<size_t>> indices;
    for (int i = 0; i < sizes.size(); ++i)
    {
        auto& s = sizes[i];
        std::vector<size_t> ind(s);
        std::iota(ind.begin(), ind.end(), 0);

        if (shuffle)
        {
            std::shuffle(ind.begin(), ind.end(), Random::generator());
        }

        // s = Saiga::iAlignDown(s, batch_size);
        // ind.resize(s);
        // SAIGA_ASSERT(ind.size() % batch_size == 0);
        // total_batches += ind.size() / batch_size;

        for (int j = 0; j < s; j += batch_size)
        {
            auto combined_offset = j + combined_indices.size();
            batch_offset_size.push_back({combined_offset, std::min<int>(batch_size, s - j)});
        }

        for (auto j : ind)
        {
            combined_indices.emplace_back(i, j);
        }
    }

    if (shuffle)
    {
        std::shuffle(batch_offset_size.begin(), batch_offset_size.end(), Random::generator());
    }
}

torch::optional<std::vector<size_t>> torch::MultiDatasetSampler::next(size_t batch_size)
{
    SAIGA_ASSERT(this->batch_size == batch_size);

    if (current_index >= batch_offset_size.size()) return {};

    auto [bo, bs] = batch_offset_size[current_index];
    current_index++;

    std::vector<size_t> result;

    for (int i = 0; i < bs; ++i)
    {
        auto [scene, image] = combined_indices[i + bo];

        size_t comb = (size_t(scene) << 32) | size_t(image);
        result.push_back(comb);
    }


    return result;
}



TorchSingleSceneDataset::TorchSingleSceneDataset(std::vector<SceneDataTrainSampler> sampler) : sampler(sampler) {}

std::vector<NeuralTrainData> TorchSingleSceneDataset::get2(size_t index)
{
    int scene = index >> 32UL;
    int image = index & ((1UL << 32UL) - 1UL);

    auto data = sampler[scene].Get(image);
    for (auto& l : data) l->scene_id = scene;
    return data;
}
std::vector<NeuralTrainData> TorchSingleSceneDataset::get_batch(torch::ArrayRef<size_t> indices)
{
    std::vector<NeuralTrainData> batch;
    batch.reserve(indices.size());
    for (const auto i : indices)
    {
        auto e = get2(i);
        for (auto& l : e)
        {
            batch.push_back(std::move(l));
        }
    }
    return batch;
}
