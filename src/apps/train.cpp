/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/time/time.h"
#include "saiga/core/util/MemoryUsage.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/vision/cameraModel/OCam.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/LRScheduler.h"

#include "data/Dataset.h"
#include "models/Pipeline.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <csignal>
#include <torch/script.h>

#include "../../External/NeAT/src/utils/cimg_wrapper.h"
#include "git_sha1.h"
#include "neat-utils/NeAT_interop.h"
#include "neat-utils/image_utils.h"
#include "tensorboard_logger.h"

std::string full_experiment_dir;

std::shared_ptr<CombinedParams> params;

torch::Device device = torch::kCUDA;

inline std::string EncodeImageToString(const Image& img, std::string type = "png")
{
    auto data = img.saveToMemory(type);

    std::string result;
    result.resize(data.size());

    memcpy(result.data(), data.data(), data.size());
    return result;
}

template <typename T>
inline void LogImage(TensorBoardLogger* tblogger, const TemplatedImage<T>& img, std::string name, int step)
{
    auto img_str = EncodeImageToString(img, "png");
    tblogger->add_image(name, step, img_str, img.h, img.w, channels(img.type));
}

static torch::Tensor CropMask(int h, int w, int border)
{
    // Create a center crop mask so that the border is valued less during training.
    TemplatedImage<unsigned char> target_mask(h, w);
    target_mask.makeZero();

    int b     = border;
    auto crop = target_mask.getImageView().subImageView(b, b, h - b * 2, w - b * 2);
    crop.set(255);

    return ImageViewToTensor<unsigned char>(target_mask, true).unsqueeze(0);
}


class TrainScene
{
   public:
    TrainScene(std::vector<std::string> scene_dirs)
    {
        for (int i = 0; i < scene_dirs.size(); ++i)
        {
            auto scene = std::make_shared<SceneData>(params->train_params.scene_base_dir + scene_dirs[i]);

            if (params->points_adding_params.dont_use_initial_pointcloud)
            {
                scene->point_cloud.position.clear();
                scene->point_cloud.color.clear();
                scene->point_cloud.data.clear();
                scene->point_cloud.normal.clear();
            }
            original_pc_size = scene->point_cloud.position.size();

            if (params->points_adding_params.scene_add_initially_random_points > 0)
            {
                AABB custom_aabb = scene->dataset_params.aabb;

                std::cout << "Adding random uniform points, "
                          << params->points_adding_params.scene_add_initially_random_points << " in bounding box "
                          << custom_aabb << std::endl;
                scene->AddRandomPoints(params->points_adding_params.scene_add_initially_random_points, custom_aabb);
            }

            // 1.  Separate indices
            auto all_indices                   = scene->Indices();
            auto [train_indices, test_indices] = params->train_params.Split(all_indices);

            if (std::filesystem::exists(params->train_params.split_index_file_train))
            {
                train_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_train);
            }


            if (std::filesystem::exists(params->train_params.split_index_file_test))
            {
                test_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_test);
            }


            if (params->train_params.duplicate_train_factor > 1)
            {
                // this multiplies the epoch size
                // increases performance for small epoch sizes
                auto cp = train_indices;
                for (int i = 1; i < params->train_params.duplicate_train_factor; ++i)
                {
                    train_indices.insert(train_indices.end(), cp.begin(), cp.end());
                }
            }


            {
                std::ofstream strm(full_experiment_dir + "/train_indices_" + scene->scene_name + ".txt");
                for (auto i : train_indices)
                {
                    strm << i << "\n";
                }
                std::ofstream strm2(full_experiment_dir + "/test_indices_" + scene->scene_name + ".txt");
                for (auto i : test_indices)
                {
                    strm2 << i << "\n";
                }
            }

            std::cout << "Train(" << train_indices.size() << "): " << array_to_string(train_indices, ' ') << std::endl;
            std::cout << "Test(" << test_indices.size() << "): " << array_to_string(test_indices, ' ') << std::endl;

            PerSceneData scene_data;

            scene_data.not_training_indices = all_indices;
            for (auto i : train_indices)
            {
                auto it = std::find(scene_data.not_training_indices.begin(), scene_data.not_training_indices.end(), i);
                if (it != scene_data.not_training_indices.end())
                {
                    scene_data.not_training_indices.erase(it);
                }
            }
            train_cropped_samplers.push_back(CroppedSampler(scene, train_indices));
            test_cropped_samplers.push_back(CroppedSampler(scene, test_indices));
            eval_samplers.push_back(FullSampler(scene, train_indices));
            test_samplers.push_back(FullSampler(scene, test_indices));


            if (params->train_params.train_mask_border > 0)
            {
                int i = 0;
                for (auto dims : test_samplers.back().image_size_crop)
                {
                    int w              = dims.x();
                    int h              = dims.y();
                    torch::Tensor mask = CropMask(h, w, params->train_params.train_mask_border).to(device);
                    TensorToImage<unsigned char>(mask).save(full_experiment_dir + "/eval_mask_" + scene->scene_name +
                                                            "_" + std::to_string(i) + ".png");
                    scene_data.eval_crop_mask.push_back(mask);
                    ++i;
                }
            }

            scene->AddIntrinsicsNoise(params->train_params.noise_intr_k, params->train_params.noise_intr_d);

            auto ns = std::make_shared<NeuralScene>(scene, params);
            if (params->train_params.noise_pose_r > 0 || params->train_params.noise_pose_t > 0)
            {
                scene->AddPoseNoise(radians(params->train_params.noise_pose_r),
                                    params->train_params.noise_pose_t / 1000.);

                torch::NoGradGuard ngg;
                auto poses2 = PoseModule(scene);
                poses2->to(device);
                ns->poses->to(device);

                PrintTensorInfo(ns->poses->poses_se3);
                PrintTensorInfo(poses2->poses_se3);
                ns->poses->poses_se3.set_(poses2->poses_se3);
            }

            if (params->train_params.noise_point > 0)
            {
                torch::NoGradGuard ngg;
                auto noise =
                    torch::normal(0, params->train_params.noise_point, ns->point_cloud_cuda->t_position.sizes())
                        .to(device);
                noise.slice(1, 3, 4).zero_();

                ns->point_cloud_cuda->t_position += noise;
            }


            scene_data.scene = ns;

            data.push_back(scene_data);
        }
    }

    SceneDataTrainSampler CroppedSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices)
    {
        ivec2 crop(params->train_params.train_crop_size, params->train_params.train_crop_size);

        SceneDataTrainSampler sampler(scene, indices, params->train_params.train_use_crop, crop,
                                      params->train_params.inner_batch_size, params->train_params.use_image_masks,
                                      params->train_params.crop_rotation);
        sampler.min_max_zoom(0)   = params->train_params.min_zoom * scene->dataset_params.render_scale;
        sampler.min_max_zoom(1)   = params->train_params.max_zoom * scene->dataset_params.render_scale;
        sampler.prefere_border    = params->train_params.crop_prefere_border;
        sampler.inner_sample_size = params->train_params.inner_sample_size;
        sampler.sample_gaussian   = params->train_params.crop_gaussian_sample;
        std::cout << "cropped sampler " << sampler.image_size_input[0].x() << "x" << sampler.image_size_input[0].y()
                  << " to " << sampler.image_size_crop[0].x() << " x " << sampler.image_size_crop[0].y()
                  << " render scale " << scene->dataset_params.render_scale << std::endl;
        return sampler;
    }

    SceneDataTrainSampler FullSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices)
    {
        int w = scene->scene_cameras.front().w * scene->dataset_params.render_scale;
        int h = scene->scene_cameras.front().h * scene->dataset_params.render_scale;

        int max_eval_size = iAlignUp(params->train_params.max_eval_size, 32);

        std::cout << "full sampler " << w << "x" << h << " render scale " << scene->dataset_params.render_scale
                  << std::endl;

        int min_scene_size = std::min(w, h);
        if (min_scene_size > max_eval_size && params->train_params.train_use_crop)
        {
            w = std::min(w, max_eval_size);
            h = std::min(h, max_eval_size);

            SceneDataTrainSampler sdf(scene, indices, true, ivec2(w, h), 1, params->train_params.use_image_masks,
                                      false);
            sdf.random_zoom        = true;
            sdf.min_max_zoom(0)    = max_eval_size / double(min_scene_size);
            sdf.min_max_zoom(1)    = max_eval_size / double(min_scene_size);
            sdf.random_translation = false;
            return sdf;
        }
        else if (scene->dataset_params.render_scale != 1)
        {
            SceneDataTrainSampler sdf(scene, indices, true, ivec2(w, h), 1, params->train_params.use_image_masks,
                                      false);
            sdf.random_zoom        = true;
            sdf.min_max_zoom(0)    = scene->dataset_params.render_scale;
            sdf.min_max_zoom(1)    = scene->dataset_params.render_scale;
            sdf.random_translation = false;
            return sdf;
        }
        else
        {
            SceneDataTrainSampler sdf(scene, indices, false, ivec2(-1, -1), 1, params->train_params.use_image_masks,
                                      false);
            sdf.random_translation = false;
            return sdf;
        }
    }

    auto DataLoader(std::vector<SceneDataTrainSampler>& train_cropped_samplers, bool train)
    {
        std::vector<uint64_t> sizes;
        for (auto& t : train_cropped_samplers)
        {
            sizes.push_back(t.Size());
        }

        int batch_size  = train ? params->train_params.batch_size : 1;
        int num_workers = train ? params->train_params.num_workers_train : params->train_params.num_workers_eval;
        bool shuffle    = train ? params->train_params.shuffle_train_indices : false;
        auto options    = torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(false).workers(num_workers);

        auto sampler = torch::MultiDatasetSampler(sizes, options.batch_size(), shuffle);
        int n        = sampler.NumImages();
        return std::pair(
            torch::data::make_data_loader(TorchSingleSceneDataset(train_cropped_samplers), std::move(sampler), options),
            n);
    }

    void Load(torch::DeviceType device, int scene)
    {
        if (scene == current_scene) return;
        Unload();
        // std::cout << "Load " << scene << std::endl;
        current_scene = scene;
        data[current_scene].scene->to(device);
    }

    void Unload(bool always_unload = false)
    {
        if (params->train_params.keep_all_scenes_in_memory && !always_unload) return;
        std::cout << "Unload " << current_scene << std::endl;
        if (current_scene != -1)
        {
            data[current_scene].scene->to(torch::kCPU);
        }
        current_scene = -1;
    }

    void SetScene(int id) { current_scene = id; }

    void Train(int epoch, bool v)
    {
        SAIGA_ASSERT(current_scene != -1);
        data[current_scene].scene->Train(epoch, v);
    }

    void StartEpoch()
    {
        for (auto& sd : data)
        {
            sd.epoch_loss = {};
        }
    }

    int current_scene = -1;

    struct PerSceneData
    {
        std::shared_ptr<NeuralScene> scene;

        // for each camera one
        std::vector<torch::Tensor> eval_crop_mask;

        // all image indices that are not used during training.
        // -> we interpolate the metadata for them
        std::vector<int> not_training_indices;

        LossResult epoch_loss;
    };
    int original_pc_size = -1;

    std::vector<PerSceneData> data;
    std::vector<SceneDataTrainSampler> train_cropped_samplers, eval_samplers, test_samplers, test_cropped_samplers;
};

ImGui::IMConsole console_error;

class NeuralTrainer;

NeuralTrainer* instance;


class NeuralTrainer
{
   public:
    torch::DeviceType device = torch::kCUDA;
    std::shared_ptr<NeuralPipeline> pipeline;

    torch::Tensor train_crop_mask;
    std::shared_ptr<TrainScene> train_scenes;
    std::string ep_dir;
    LRSchedulerPlateau lr_scheduler;

    std::shared_ptr<TensorBoardLogger> tblogger;

    std::vector<int> point_adding_epochs;
    std::vector<int> point_removal_epochs;

    struct PointAddingStateSave
    {
        std::filesystem::path latest_neat_config_path;
        std::filesystem::path latest_ct_reco_path;
        float latest_scene_scale;
        AABB latest_scene_aabb;
        vec3 latest_scene_translation;
    };
    std::vector<PointAddingStateSave> latest_pa_state;

    IntrinsicsPinholef ocam_targetK;
    ivec2 size_of_ocam_target_image = ivec2(2560, 2560);

    CUDA::CudaTimerSystem timer_system;

    void signal_handler(int signal)
    {
        ep_dir.pop_back();
        ep_dir += "_interupted/";
        std::filesystem::create_directory(ep_dir);

        bool reduced_cp = params->train_params.reduced_check_point;
        // Save checkpoint
        console << "Saving checkpoint..." << std::endl;

        if (!reduced_cp)
        {
            pipeline->SaveCheckpoint(ep_dir);
        }

        // for (auto scene : scenes)
        for (auto& s : train_scenes->data)
        {
            s.scene->SaveCheckpoint(ep_dir, reduced_cp);
        }
        exit(0);
    }

    ~NeuralTrainer() {}

    NeuralTrainer()
    {
        instance = this;
        ocam_targetK =
            IntrinsicsPinholef(1000, 1000, size_of_ocam_target_image.x() / 2, size_of_ocam_target_image.y() / 2, 0);

        lr_scheduler =
            LRSchedulerPlateau(params->train_params.lr_decay_factor, params->train_params.lr_decay_patience, true);
        torch::set_num_threads(1);

        std::string experiment_name = Saiga::CurrentTimeString("%F_%H-%M-%S") + "_" + params->train_params.name;
        full_experiment_dir         = params->train_params.experiment_dir + "/" + experiment_name + "/";
        std::filesystem::create_directories(full_experiment_dir);

        console.setOutputFile(full_experiment_dir + "log.txt");
        SAIGA_ASSERT(console.rdbuf());
        std::cout.rdbuf(console.rdbuf());

        console_error.setOutputFile(full_experiment_dir + "error.txt");
        SAIGA_ASSERT(console_error.rdbuf());
        std::cerr.rdbuf(console_error.rdbuf());

        // create important epochs
        auto& prm = params->points_adding_params;
        if (params->pipeline_params.use_point_adding_and_removing_module)
        {
            for (int ep = 0; ep < params->train_params.num_epochs; ++ep)
            {
                // select render mode
                int render_mode_select  = 0;
                int diff_to_start_epoch = 1000000;
                for (int i = 0; i < params->pipeline_params.render_modes_start_epochs.size(); ++i)
                {
                    int start_ep_m = params->pipeline_params.render_modes_start_epochs[i];
                    if (ep >= start_ep_m)
                    {
                        if (ep - start_ep_m < diff_to_start_epoch)
                        {
                            diff_to_start_epoch = ep - start_ep_m;
                            render_mode_select  = i;
                        }
                    }
                }

                if (prm.start_adding_points_epoch >= 0)
                {
                    // remove points starting at epoch and every adding interval afterwards
                    if (ep == prm.start_adding_points_epoch) point_adding_epochs.push_back(ep);
                    if (ep > prm.start_adding_points_epoch)
                    {
                        if (((ep - prm.start_adding_points_epoch) % prm.point_adding_epoch_interval) == 0)
                        {
                            if (render_mode_select != PointRendererCache::RenderMode::FUZZY_DT)
                            {
                                point_adding_epochs.push_back(ep);
                            }
                        }
                    }
                }
                if (prm.start_removing_points_epoch >= 0)
                {
                    if (ep == prm.start_removing_points_epoch) point_removal_epochs.push_back(ep);
                    if (ep > prm.start_removing_points_epoch)
                        if (((ep - prm.start_removing_points_epoch) % prm.point_removal_epoch_interval) == 0)
                            point_removal_epochs.push_back(ep);
                }
            }

            if (point_adding_epochs.size() > 0)
            {
                std::cout << "Add points in epochs: ";
                for (auto e : point_adding_epochs) std::cout << e << ", ";
                std::cout << std::endl;
            }
            if (point_removal_epochs.size() > 0)
            {
                std::cout << "Remove points in epochs: ";
                for (auto e : point_removal_epochs) std::cout << e << ", ";
                std::cout << std::endl;
            }
        }
        std::cout << "Render Mode epochs: DT " << params->pipeline_params.render_modes_start_epochs[0]
                  << " - Fullblend " << params->pipeline_params.render_modes_start_epochs[1] << " - Fuzzyblend "
                  << params->pipeline_params.render_modes_start_epochs[2] << " - BilinearBlend "
                  << params->pipeline_params.render_modes_start_epochs[3] << std::endl;
        std::cout << (params->points_adding_params.neat_use_as_subprocess_ct_reco
                          ? "Use NeAT reco"
                          : ((params->points_adding_params.fixed_ct_reco_path != "")
                                 ? "use fixed reco path:" + params->points_adding_params.fixed_ct_reco_path
                                 : "use grid based reco"));
        // end create important epochs

        tblogger     = std::make_shared<TensorBoardLogger>((full_experiment_dir + "/tfevents.pb").c_str());
        train_scenes = std::make_shared<TrainScene>(params->train_params.scene_names);

        latest_pa_state = std::vector<PointAddingStateSave>(train_scenes->data.size());

        // Save all parameters into experiment output dir
        params->Save(full_experiment_dir + "/params.ini");

        {
            std::ofstream strm(full_experiment_dir + "/git.txt");
            strm << GIT_SHA1 << std::endl;
        }

        pipeline = std::make_shared<NeuralPipeline>(params);


        /*  if (params->pipeline_params.use_points_for_env_map)
          {
              for (int i = 0; i < train_scenes->data.size(); ++i)
              {
                  train_scenes->data[i].scene->AddNewRandomForEnvSphere(
                      params->pipeline_params.env_spheres, params->pipeline_params.env_inner_radius,
                      params->pipeline_params.env_radius_factor, params->pipeline_params.env_num_points, false);
              }
          }*/

        for (int epoch_id = 0; epoch_id <= params->train_params.num_epochs; ++epoch_id)
        {
            bool point_important_epoch_for_train = params->pipeline_params.use_point_adding_and_removing_module &&
                                                   (std::find(point_adding_epochs.begin(), point_adding_epochs.end(),
                                                              epoch_id) != point_adding_epochs.end() ||
                                                    std::find(point_removal_epochs.begin(), point_removal_epochs.end(),
                                                              epoch_id) != point_removal_epochs.end());

            // we want an eval step before point adding, for up-to-date grid
            bool point_important_epoch_for_eval = params->pipeline_params.use_point_adding_and_removing_module &&
                                                  (std::find(point_adding_epochs.begin(), point_adding_epochs.end(),
                                                             epoch_id + 1) != point_adding_epochs.end() ||
                                                   std::find(point_removal_epochs.begin(), point_removal_epochs.end(),
                                                             epoch_id + 1) != point_removal_epochs.end());
            std::cout << std::endl;
            std::cout << "=== Epoch " << epoch_id << " ===" << std::endl;
            std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);

            bool last_ep         = epoch_id == params->train_params.num_epochs;
            bool save_checkpoint = epoch_id % params->train_params.save_checkpoints_its == 0 || last_ep;

            // always save cp on add_remove
            save_checkpoint |= point_important_epoch_for_train;
            save_checkpoint |= point_important_epoch_for_eval;

            std::string point_add_remove              = "";
            std::vector<std::string> name_render_mode = {"DT", "FullBlend", "FuzzyBlend", "BilinearBlend"};
            for (int i = 0; i < params->pipeline_params.render_modes_start_epochs.size(); ++i)
            {
                int switch_epoch = params->pipeline_params.render_modes_start_epochs[i];
                if (switch_epoch > 0)
                {
                    save_checkpoint |= (switch_epoch == epoch_id || switch_epoch + 1 == epoch_id);

                    if (switch_epoch == epoch_id) point_add_remove += "_last" + name_render_mode[i];
                    if (switch_epoch + 1 == epoch_id) point_add_remove += "_first" + name_render_mode[i];
                }
            }
            if (std::find(point_adding_epochs.begin(), point_adding_epochs.end(), epoch_id + 1) !=
                point_adding_epochs.end())
                point_add_remove += "_beforeAdd";
            if (std::find(point_adding_epochs.begin(), point_adding_epochs.end(), epoch_id) !=
                point_adding_epochs.end())
                point_add_remove += "_afterAdd";
            if (std::find(point_removal_epochs.begin(), point_removal_epochs.end(), epoch_id + 1) !=
                point_removal_epochs.end())
                point_add_remove += "_beforeRem";
            if (std::find(point_removal_epochs.begin(), point_removal_epochs.end(), epoch_id) !=
                point_removal_epochs.end())
                point_add_remove += "_afterRem";

            ep_dir = full_experiment_dir + "ep" + ep_str + point_add_remove + "/";
            if (save_checkpoint)
            {
                std::filesystem::create_directory(ep_dir);
                std::filesystem::create_directory(ep_dir + "/test/");
            }
            {
                if (params->train_params.do_train && epoch_id > 0)
                {
                    AddAndRemovePoints(epoch_id);
                    auto epoch_loss = TrainEpoch(epoch_id, train_scenes->train_cropped_samplers, false, "Train");

                    for (auto& sd : train_scenes->data)
                    {
                        tblogger->add_scalar("LossTrain/" + sd.scene->scene->scene_name, epoch_id,
                                             sd.epoch_loss.Average().loss_float);
                        sd.epoch_loss.Average().AppendToFile(
                            full_experiment_dir + "loss_train_" + sd.scene->scene->scene_name + ".txt", epoch_id);
                    }

                    if (params->train_params.optimize_eval_camera)
                    {
                        TrainEpoch(epoch_id, train_scenes->test_cropped_samplers, true, "EvalRefine");
                        for (auto& sd : train_scenes->data)
                        {
                            tblogger->add_scalar("LossEvalRefine/" + sd.scene->scene->scene_name, epoch_id,
                                                 sd.epoch_loss.Average().loss_float);
                            sd.epoch_loss.Average().AppendToFile(
                                full_experiment_dir + "loss_eval_refine_" + sd.scene->scene->scene_name + ".txt",
                                epoch_id);
                        }
                    }

                    auto reduce_factor           = lr_scheduler.step(epoch_loss);
                    static double current_factor = 1;
                    tblogger->add_scalar("LR/factor", epoch_id, current_factor);
                    current_factor *= reduce_factor;

                    if (reduce_factor < 1)
                    {
                        std::cout << "Reducing LR by " << reduce_factor << ". Current Factor: " << current_factor
                                  << std::endl;
                    }

                    pipeline->UpdateLearningRate(reduce_factor);
                    for (auto& s : train_scenes->data)
                    {
                        s.scene->UpdateLearningRate(epoch_id, reduce_factor);
                    }
                }

                if (params->train_params.debug)
                {
                    std::cout << GetMemoryInfo() << std::endl;
                }
                pipeline->Log(full_experiment_dir);
                for (auto& s : train_scenes->data)
                {
                    s.scene->Log(full_experiment_dir);
                }
                bool want_eval =
                    params->train_params.do_eval && (!params->train_params.eval_only_on_checkpoint || save_checkpoint);

                // always eval on point_add_remove
                want_eval |= point_important_epoch_for_eval;
                // always eval after adding/removing points for clarity
                want_eval |= point_important_epoch_for_train;

                if (want_eval)
                {
                    EvalEpoch(epoch_id, save_checkpoint);

                    TestEpoch(epoch_id);
                    for (auto& sd : train_scenes->data)
                    {
                        auto avg = sd.epoch_loss.Average();
                        tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/vgg", epoch_id,
                                             avg.loss_vgg);
                        tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/lpips", epoch_id,
                                             avg.loss_lpips);
                        tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/psnr", epoch_id,
                                             avg.loss_psnr);
                        avg.AppendToFile(full_experiment_dir + "loss_eval_" + sd.scene->scene->scene_name + ".txt",
                                         epoch_id);
                    }
                }
            }

            if (save_checkpoint)
            {
                bool reduced_cp = params->train_params.reduced_check_point && !last_ep;
                // Save checkpoint
                console << "Saving checkpoint..." << std::endl;

                if (!reduced_cp)
                {
                    pipeline->SaveCheckpoint(ep_dir);
                }

                for (auto& s : train_scenes->data)
                {
                    s.scene->SaveCheckpoint(ep_dir, reduced_cp);
                }
            }
        }

        std::string finished_ep_dir = params->train_params.experiment_dir + "/" + "_f_" + experiment_name + "/";
        std::cout << "rename " << full_experiment_dir << " to " << finished_ep_dir << std::endl;
        std::filesystem::rename(full_experiment_dir, finished_ep_dir);
    }


    std::filesystem::path get_last_epoch(std::filesystem::path p, std::string epoch_prefix = "ep")
    {
        std::set<std::filesystem::path, std::greater<std::filesystem::path>> sorted_by_name;
        for (auto& entry : std::filesystem::directory_iterator(p))
            if (entry.is_directory()) sorted_by_name.insert(entry.path());

        for (auto& f : sorted_by_name)
        {
            std::string dir = f.filename();
            if (dir.rfind(epoch_prefix, 0) == 0) return f;
        }
        return p;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void AddAndRemovePoints(int epoch_id)
    {
        bool point_important_epoch =
            params->pipeline_params.use_point_adding_and_removing_module &&
            (std::find(point_adding_epochs.begin(), point_adding_epochs.end(), epoch_id) != point_adding_epochs.end() ||
             std::find(point_removal_epochs.begin(), point_removal_epochs.end(), epoch_id) !=
                 point_removal_epochs.end());

        // if (epoch_id == 2) point_important_epoch = true;
        // point_adding_epochs.push_back(2);

        if (point_important_epoch)
        {
            std::cout << "Important epoch: Add or remove points" << std::endl;
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                auto& tex = train_scenes->data[i].scene->texture;
                {
                    if (std::find(point_removal_epochs.begin(), point_removal_epochs.end(), epoch_id) !=
                        point_removal_epochs.end())
                    {
                        std::cout << "Remove Points" << std::endl;

                        auto indices_to_remove =
                            torch::where(tex->confidence_value_of_point.squeeze() <
                                             params->points_adding_params.removal_confidence_cutoff,
                                         1, 0)
                                .nonzero();

                        if (indices_to_remove.size(0) > 0)
                        {
                            train_scenes->data[i].scene->RemovePoints(indices_to_remove);
                            train_scenes->data[i].scene->OptimizerClear(epoch_id, false);
                        }
                        else
                        {
                            std::cout << "no removal" << std::endl;
                        }
                        std::cout << "Texture after removal: " << TensorInfo(tex->texture) << std::endl;
                    }
                    // add points
                    if (std::find(point_adding_epochs.begin(), point_adding_epochs.end(), epoch_id) !=
                        point_adding_epochs.end())
                    {
                        std::cout << "Add Points" << std::endl;

                        if (params->points_adding_params.only_use_point_growing)
                        {
                            train_scenes->data[i].scene->AddPointsViaPointGrowing(2, 0.5);
                        }
                        else if (params->points_adding_params.neat_use_as_subprocess_ct_reco)
                        {
                            std::cout << "Use NeAT dataset reco" << std::endl;
                            // SAIGA_ASSERT(!latest_neat_config_path.empty());
                            std::string path_to_neat_executable =
                                params->points_adding_params.full_path_to_neat_executable;
                            std::string subprocess_execute_call =
                                path_to_neat_executable + " " + latest_pa_state[i].latest_neat_config_path.string();
#if 1
                            // unload all scenes
                            int save_scene_id = train_scenes->current_scene;
                            train_scenes->Unload(true);
                            c10::cuda::CUDACachingAllocator::emptyCache();

                            std::cout << "Params: " << params->points_adding_params.add_points_amount_max_per_cell
                                      << "; " << latest_pa_state[i].latest_scene_scale << "; "
                                      << latest_pa_state[i].latest_scene_translation << "; "
                                      << latest_pa_state[i].latest_scene_aabb << std::endl;

                            std::cout << "System call NeAT: " << subprocess_execute_call << std::flush << std::endl;
                            auto ret_val = std::system(subprocess_execute_call.c_str());
                            std::cout << std::flush << "NeAT returned with return code: " << ret_val << std::endl;

                            train_scenes->Load(device, save_scene_id);


                            auto last_ep             = get_last_epoch(latest_pa_state[i].latest_ct_reco_path);
                            auto volume_neat_path    = get_last_epoch(last_ep, "volume");
                            std::string hdr_img_path = volume_neat_path.string() + "/volume.hdr";
                            std::cout << "Load from path: " << hdr_img_path << std::endl;
                            torch::Tensor volume_tensor = LoadHDRImageTensor(hdr_img_path);
                            std::cout << "Loaded Volume Tensor: " << TensorInfo(volume_tensor) << std::endl;

                            train_scenes->data[i].scene->AddNewRandomPointsFromCTHdr(
                                volume_tensor, params->points_adding_params.add_points_amount_max_per_cell,
                                latest_pa_state[i].latest_scene_scale, latest_pa_state[i].latest_scene_translation,
                                latest_pa_state[i].latest_scene_aabb);

#else
                            torch::Tensor volume_tensor = torch::rand({1, 5, 5, 5});
                            std::cout << "Loaded Volume Tensor: " << TensorInfo(volume_tensor) << std::endl;
                            train_scenes->data[i].scene->AddNewRandomPointsFromCTHdr(
                                volume_tensor, params->points_adding_params.add_points_amount_max_per_cell, 10,
                                vec3(15, 15, 15), AABB(vec3(10, 10, 10), vec3(20, 20, 20)));
#endif
                        }
                        else if (params->points_adding_params.fixed_ct_reco_path != "")
                        {
#if 0
                            latest_pa_state[i].latest_scene_scale       = 1.5;
                            latest_pa_state[i].latest_scene_translation = vec3(1, -0.45, 0.65);
                            latest_pa_state[i].latest_scene_aabb        = AABB(vec3(0, -2, 0), vec3(2, 1.1, 1.3));
#endif
                            std::string hdr_img_path = params->points_adding_params.fixed_ct_reco_path +
                                                       "/volume.hdr";  // ep_dir + "/ct_reco/";
                            torch::Tensor volume_tensor = LoadHDRImageTensor(hdr_img_path);
                            std::cout << "Loaded Volume Tensor: " << TensorInfo(volume_tensor) << std::endl;
                            train_scenes->data[i].scene->AddNewRandomPointsFromCTHdr(
                                volume_tensor, params->points_adding_params.add_points_amount_max_per_cell,
                                latest_pa_state[i].latest_scene_scale, latest_pa_state[i].latest_scene_translation,
                                latest_pa_state[i].latest_scene_aabb);


#ifdef USE_PNG_STACK
                            train_scenes->data[i].scene->AddNewRandomPointsFromCTStack(
                                params->points_adding_params.add_points_amount_max_per_cell,
                                params->points_adding_params.fixed_ct_reco_path);
#endif
                        }
                        else
                        {
                            train_scenes->data[i].scene->AddNewRandomPointsInValuefilledBB(
                                params->points_adding_params.add_points_amount_max_per_cell,
                                params->points_adding_params.add_only_in_top_x_factor_of_cells);
                        }
                        train_scenes->data[i].scene->OptimizerClear(epoch_id, false);
                        std::cout << "Texture after adding: " << TensorInfo(tex->texture) << std::endl;
                    }

                    static bool once_ep = true;
                    if (params->pipeline_params.use_points_for_env_map &&
                        !params->pipeline_params.use_env_points_at_start &&
                        !params->pipeline_params.enable_environment_map)
                    {
                        train_scenes->data[i].scene->AddNewRandomForEnvSphere(
                            params->pipeline_params.env_spheres, params->pipeline_params.env_inner_radius,
                            params->pipeline_params.env_radius_factor, params->pipeline_params.env_num_points, true);

                        once_ep = false;
                    }
                }
            }
        }
    }


    double TrainEpoch(int epoch_id, std::vector<SceneDataTrainSampler>& data, bool structure_only, std::string name)
    {
        train_scenes->StartEpoch();
        // Train
        float epoch_loss           = 0;
        int num_images             = 0;
        auto [loader, loader_size] = train_scenes->DataLoader(data, true);

        pipeline->Train(epoch_id);
        {
            Saiga::ProgressBar bar(std::cout, name + " " + std::to_string(epoch_id) + " |",
                                   loader_size * params->train_params.inner_batch_size, 30, false, 5000);
            for (std::vector<NeuralTrainData>& batch : *loader)
            {
                SAIGA_ASSERT(batch.size() <= params->train_params.batch_size * params->train_params.inner_batch_size);

                int scene_id_of_batch = batch.front()->scene_id;
                auto& scene_data      = train_scenes->data[scene_id_of_batch];

                train_scenes->Load(device, scene_id_of_batch);
                train_scenes->Train(epoch_id, true);
                if (!train_crop_mask.defined() && params->train_params.train_mask_border > 0)
                {
                    int h           = batch.front()->img.h;
                    int w           = batch.front()->img.w;
                    train_crop_mask = CropMask(h, w, params->train_params.train_mask_border).to(device);
                    TensorToImage<unsigned char>(train_crop_mask).save(full_experiment_dir + "/train_mask.png");
                }
                pipeline->timer_system = &timer_system;

                timer_system.BeginFrame();
                bool output_during_training = false;
                // std::cout << timer_system-> << " , " << std::endl;
                ForwardResult result;
                {
                    //  auto timer = timer_system.Measure("Forward");
                    SAIGA_OPTIONAL_TIME_MEASURE("Forward", pipeline->timer_system);

                    result = pipeline->Forward(*scene_data.scene, batch, train_crop_mask, false, epoch_id,
                                               output_during_training);
                    scene_data.epoch_loss += result.float_loss;
                    epoch_loss += result.float_loss.loss_float * batch.size();
                    num_images += batch.size();
                }


                if (output_during_training)
                {
                    std::cout << "IMAGES: " << result.image_ids.size() << std::endl;

                    for (int i = 0; i < result.image_ids.size(); ++i)
                    {
                        result.outputs[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) + "_" + std::to_string(i) +
                                               params->train_params.output_file_type);
                        result.targets[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) + "_" + std::to_string(i) +
                                               "_gt" + params->train_params.output_file_type);
                    }
                }
                {
                    auto timer = timer_system.Measure("Backwards");

                    // scene_data.epoch_loss += result.float_loss;
                    // epoch_loss += result.float_loss.loss_float;
                    // num_images += batch.size();
                    result.loss.backward();
                    // std::cout << "AFTER BACK: " << TensorInfo(scene_data.scene->poses->tangent_poses)
                    //           << TensorInfo(scene_data.scene->poses->tangent_poses.mutable_grad()) << std::endl;
                    if (params->points_adding_params.push_point_confidences_down != 0.f)
                    {
                        //   std::cout << TensorInfo(scene_data.scene->texture->confidence_raw.grad()) << std::endl;
                        scene_data.scene->texture->confidence_raw.mutable_grad() +=
                            params->points_adding_params.push_point_confidences_down;
                        // 0.0001 * (scene_data.scene->texture->confidence_raw.mutable_grad().max() -
                        //           scene_data.scene->texture->confidence_raw.mutable_grad().min());
                    }

                    if (!structure_only)
                    {
                        pipeline->OptimizerStep(epoch_id);
                    }
                    scene_data.scene->OptimizerStep(epoch_id, structure_only);
                    bar.addProgress(batch.size());
                    bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) + " " +
                                   std::to_string(num_images) + " Avg=" + std::to_string(epoch_loss / num_images));
                }
                timer_system.EndFrame();
            }
        }
        timer_system.PrintTable(std::cout);

        train_scenes->Unload();
        return epoch_loss / num_images;
    }


    void EvalEpoch(int epoch_id, bool save_checkpoint)
    {
        bool epoch_before_pointadd = false;
        if (std::find(point_adding_epochs.begin(), point_adding_epochs.end(), epoch_id + 1) !=
            point_adding_epochs.end())
            epoch_before_pointadd = true;

        train_scenes->StartEpoch();

        // Eval
        torch::NoGradGuard ngg;
        float epoch_loss           = 0;
        int num_images             = 0;
        auto [loader, loader_size] = train_scenes->DataLoader(train_scenes->eval_samplers, false);

        pipeline->Train(false);

        float best_loss  = 1000000;
        float worst_loss = 0;
        ForwardResult best_batch, worst_batch;

        bool write_test_images = save_checkpoint && params->train_params.write_test_images;

        if (params->points_adding_params.use_grid_loss)
        {
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                train_scenes->data[i].scene->point_cloud_cuda->ResetCellValues();
            }
        }
        if (epoch_before_pointadd)
        {
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                // std::filesystem::create_directories(ep_dir + "/l1_loss/");
                if (params->points_adding_params.neat_loss_folder_name == "l1_loss_grey")
                    std::filesystem::create_directories(ep_dir + "/l1_loss_grey_" + std::to_string(i) + "/");
                if (params->points_adding_params.neat_loss_folder_name == "ssim_map")
                    std::filesystem::create_directories(ep_dir + "/ssim_map_" + std::to_string(i) + "/");
                if (params->points_adding_params.neat_loss_folder_name == "l2_loss")
                    std::filesystem::create_directories(ep_dir + "/l2_loss_" + std::to_string(i) + "/");
            }
        }

        Saiga::ProgressBar bar(std::cout, "Eval  " + std::to_string(epoch_id) + " |", loader_size, 30, false, 5000);
        for (std::vector<NeuralTrainData>& batch : *loader)
        {
            SAIGA_ASSERT(batch.size() == 1);
            int scene_id_of_batch = batch.front()->scene_id;
            auto& scene_data      = train_scenes->data[scene_id_of_batch];
            train_scenes->Load(device, scene_id_of_batch);
            train_scenes->Train(epoch_id, false);
            int camera_id = batch.front()->img.camera_index;

            SAIGA_ASSERT(!torch::GradMode::is_enabled());

            auto result =
                pipeline->Forward(*scene_data.scene, batch, scene_data.eval_crop_mask[camera_id], true, epoch_id,
                                  write_test_images | params->train_params.write_images_at_checkpoint);

            if (params->train_params.write_images_at_checkpoint)
            {
                for (int i = 0; i < result.image_ids.size(); ++i)
                {
                    // In average only write 10 images
                    if (Random::sampleBool(std::min(1.0, 10.0 / loader_size)))
                    {
                        auto err = ImageTransformation::ErrorImage(result.outputs[i], result.targets[i]);
                        TemplatedImage<ucvec3> combined(err.h, err.w + result.outputs[i].w);
                        combined.getImageView().setSubImage(0, 0, result.outputs[i].getImageView());
                        combined.getImageView().setSubImage(0, result.outputs[i].w, err.getImageView());

                        // LogImage(
                        //     tblogger.get(), combined,
                        //     "Checkpoint" + leadingZeroString(epoch_id, 4) + "/" +
                        //     scene_data.scene->scene->scene_name, result.image_ids[i]);


                        result.outputs[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) +
                                               params->train_params.output_file_type);

                        result.targets[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) + "_gt" +
                                               params->train_params.output_file_type);
                    }
                }
            }
            if (write_test_images)
            {
                if (result.float_loss.loss_float < best_loss)
                {
                    best_loss  = result.float_loss.loss_float;
                    best_batch = result;
                }

                if (result.float_loss.loss_float > worst_loss)
                {
                    worst_loss  = result.float_loss.loss_float;
                    worst_batch = result;
                }
            }

            epoch_loss += result.float_loss.loss_float;
            scene_data.epoch_loss += result.float_loss;
            num_images += batch.size();

            std::vector<ReducedImageInfo> images_rii;
            for (int b_id = 0; b_id < batch.size(); ++b_id)
            {
                images_rii.push_back(batch[b_id]->img);
            }

            // before point adding: save error maps
            if (epoch_before_pointadd)
            {
                auto ssim_map = pipeline->loss_ssim->get_ssim_map(result.target, result.x);

                if (scene_data.scene->scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
                {
                    scene_data.scene->DownloadIntrinsics();
                }
                for (int b_id = 0; b_id < batch.size(); ++b_id)
                {
                    auto l1_l_tens =
                        torch::abs(result.target - result.x).slice(0, b_id, b_id + 1).squeeze(0).contiguous();

                    if (params->points_adding_params.use_grid_loss)
                        pipeline->render_module->cache->FillSceneGridWithLoss(b_id, scene_data.scene->point_cloud_cuda,
                                                                              images_rii, l1_l_tens);

                    // l1 greyscale
                    l1_l_tens   = process_l1_image(torch::sum(l1_l_tens, 0).unsqueeze(0));  // .expand({3,-1,-1})/3.0;
                    auto l1_img = Saiga::TensorToImage<float>(l1_l_tens);

                    // l2 loss
                    auto l2_tens = torch::abs(result.target.pow(2.f) - result.x.pow(2.f)).sqrt();
                    l2_tens      = l2_tens.slice(0, b_id, b_id + 1).squeeze(0).contiguous();
                    l2_tens      = process_l2_image(torch::sum(l2_tens, 0).unsqueeze(0));
                    auto l2_img  = Saiga::TensorToImage<float>(l2_tens);

                    // ssim_map
                    auto ssim_img = Saiga::TensorToImage<float>(process_ssim_image(
                        1 - ssim_map.slice(0, b_id, b_id + 1).squeeze(0).contiguous().mean(0).unsqueeze(0)));

                    if (scene_data.scene->scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
                    {
                        auto K = scene_data.scene->scene->scene_cameras[camera_id].K;
                        auto D = scene_data.scene->scene->scene_cameras[camera_id].distortion;

                        auto target_K  = K;
                        ivec2 size_img = ivec2(scene_data.scene->scene->scene_cameras[camera_id].w,
                                               scene_data.scene->scene->scene_cameras[camera_id].h);
                        // currently, only one camera is supported: thus undistort into that cams K
                        if (camera_id > 0)
                        {
                            target_K = scene_data.scene->scene->scene_cameras[0].K;
                            size_img = ivec2(scene_data.scene->scene->scene_cameras[0].w,
                                             scene_data.scene->scene->scene_cameras[0].h);
                        }
                        l1_img   = UndistortImage(l1_img, K, D, target_K, size_img);
                        l2_img   = UndistortImage(l2_img, K, D, target_K, size_img);
                        ssim_img = UndistortImage(ssim_img, K, D, target_K, size_img);
                    }
                    else if (scene_data.scene->scene->dataset_params.camera_model == CameraModel::OCAM)
                    {
                        auto K = scene_data.scene->scene->scene_cameras[camera_id].K;
                        auto O = scene_data.scene->scene->scene_cameras[camera_id].ocam;


                        l1_img   = UndistortOCAMImage(l1_img, K, ocam_targetK, O, size_of_ocam_target_image,
                                                      scene_data.scene->scene->dataset_params.render_scale);
                        l2_img   = UndistortOCAMImage(l2_img, K, ocam_targetK, O, size_of_ocam_target_image,
                                                      scene_data.scene->scene->dataset_params.render_scale);
                        ssim_img = UndistortOCAMImage(ssim_img, K, ocam_targetK, O, size_of_ocam_target_image,
                                                      scene_data.scene->scene->dataset_params.render_scale);
                    }

                    // save images
                    if (params->points_adding_params.neat_loss_folder_name == "l1_loss_grey")
                        write16bitImg(l1_img, ep_dir + "/l1_loss_grey_" + std::to_string(scene_id_of_batch) + "/" +
                                                  leadingZeroString(result.image_ids[b_id], 5) + ".png");
                    else if (params->points_adding_params.neat_loss_folder_name == "l2_loss")
                        write16bitImg(l2_img, ep_dir + "/l2_loss_" + std::to_string(scene_id_of_batch) + "/" +
                                                  leadingZeroString(result.image_ids[b_id], 5) + ".png");
                    else if (params->points_adding_params.neat_loss_folder_name == "ssim_map")
                        write16bitImg(ssim_img, ep_dir + "/ssim_map_" + std::to_string(scene_id_of_batch) + "/" +
                                                    leadingZeroString(result.image_ids[b_id], 5) + ".png");
                }
            }
            bar.addProgress(batch.size());
            bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) +
                           " Avg=" + std::to_string(epoch_loss / num_images));
        }
        if (params->points_adding_params.use_grid_loss)
        {
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                train_scenes->data[i].scene->point_cloud_cuda->NormalizeBBCellValue();
            }
        }
        bar.Quit();

        if (epoch_before_pointadd)
        {
            // SAIGA_ASSERT(train_scenes->data.size() == 1, "IF more scenes: rewrite state saving paths and scales");
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                auto& scene_data = train_scenes->data[i];

                float scene_scale      = params->points_adding_params.neat_scene_scale;
                vec3 scene_translation = vec3(0, 0, 0);
                AABB sceneAABB         = scene_data.scene->scene->dataset_params.aabb;
                if (sceneAABB.maxSize() < 0.001)
                {
                    std::cout << "Compute Bounding Box of scene: " << std::flush;

                    sceneAABB = train_scenes->data[i].scene->point_cloud_cuda->Mesh().BoundingBox();
                    std::cout << sceneAABB << std::endl;
                    sceneAABB.scale(vec3(params->points_adding_params.neat_scene_scale,
                                         params->points_adding_params.neat_scene_scale,
                                         params->points_adding_params.neat_scene_scale));
                    std::cout << "scale with " << params->points_adding_params.neat_scene_scale
                              << ",result: " << sceneAABB << std::endl;
                }
                else
                {
                    std::cout << "Use scene AABB: " << sceneAABB << std::endl;
                }
                scene_translation = 0.5f * (sceneAABB.max + sceneAABB.min);

                float max_len_box = sceneAABB.maxSize();
                // max_len_box *= 1.1f;
                scene_scale = max_len_box / 2.f;
                std::cout << "Scene scale: " << scene_scale << "; scene translation: (" << scene_translation.x() << ", "
                          << scene_translation.y() << ", " << scene_translation.z() << ") " << std::endl;

                latest_pa_state[i].latest_scene_aabb        = sceneAABB;
                latest_pa_state[i].latest_scene_scale       = scene_scale;
                latest_pa_state[i].latest_scene_translation = scene_translation;


                if (params->points_adding_params.neat_use_as_subprocess_ct_reco)
                {
                    std::string scene_name_ex = "neat_dataset_" + std::to_string(i) + "_" + std::to_string(epoch_id);
                    std::filesystem::path neat_dataset_internal_dir = std::filesystem::current_path() / ep_dir;
                    std::filesystem::path neat_dataset_path         = neat_dataset_internal_dir / scene_name_ex;
                    std::cout << "Create NeAT dataset: " << neat_dataset_path << std::endl;
                    std::filesystem::create_directories(neat_dataset_path);

                    // NeAT camera.ini
                    CameraBase cam;

                    if (train_scenes->data[i].scene->scene->dataset_params.camera_model ==
                        CameraModel::PINHOLE_DISTORTION)
                    {
                        //  SAIGA_ASSERT(train_scenes->eval_samplers[i].image_size_input.size() == 1);
                        std::cerr << "Multiple Cameras found, undistort using intrinsics of the first one" << std::endl;
                        std::vector<IntrinsicsPinholef> K_mats = train_scenes->data[i].scene->intrinsics->DownloadK();

                        auto K_n = K_mats[0];
                        cam.K.cx = K_n.cx;
                        cam.K.cy = K_n.cy;

                        cam.K.fx = K_n.fx;
                        cam.K.fy = K_n.fy;
                        cam.K.s  = 0;
                        cam.w    = train_scenes->eval_samplers[i].image_size_input[0].x();
                        cam.h    = train_scenes->eval_samplers[i].image_size_input[0].y();
                    }
                    else if (train_scenes->data[i].scene->scene->dataset_params.camera_model == CameraModel::OCAM)
                    {
                        auto K_n = ocam_targetK;
                        cam.K.cx = K_n.cx;
                        cam.K.cy = K_n.cy;

                        cam.K.fx = K_n.fx;
                        cam.K.fy = K_n.fy;
                        cam.K.s  = 0;

                        cam.w = size_of_ocam_target_image.x();
                        cam.h = size_of_ocam_target_image.y();
                    }

                    cam.Save(neat_dataset_path / "camera.ini");

                    // NeAT dataset.ini
                    DatasetParams out_params;
                    out_params.image_dir = std::filesystem::current_path().string() + "/" + ep_dir + "/" +
                                           params->points_adding_params.neat_loss_folder_name + "_" +
                                           std::to_string(i) + "/";
                    out_params.mask_dir = "";

                    // scene scale neat always 1!
                    out_params.scene_scale     = 1.f;
                    out_params.log_space_input = true;
                    out_params.xray_min        = 1;
                    out_params.z_min           = params->points_adding_params.neat_zmin;
                    out_params.Save(neat_dataset_path / "dataset.ini");

                    // images.txt
                    std::filesystem::copy(train_scenes->data[i].scene->scene->scene_path + "/images.txt",
                                          neat_dataset_path);  //, std::filesystem::copy_options::recursive);
                    {
                        std::ofstream strm(neat_dataset_path / "images.txt");
                        std::ifstream images_file(train_scenes->data[i].scene->scene->scene_path + "/images.txt");
                        std::string line;
                        int l_i = 0;
                        while (std::getline(images_file, line))
                        {
                            // remove "png" or "jpg" and adding png file ending
                            std::string img_name = leadingZeroString(l_i, 5) + ".png";
                            strm << img_name << "\n";
                            ++l_i;
                        }
                        strm.close();
                    }

                    // poses.txt and camera_indices.txt
                    {
                        std::vector<Sophus::SE3d> current_poses = train_scenes->data[i].scene->poses->Download();
                        std::ofstream strm(neat_dataset_path / "poses.txt");
                        std::ofstream strm2(neat_dataset_path / "camera_indices.txt");

                        for (auto pf : current_poses)
                        {
                            SE3 p  = pf.cast<double>().inverse();
                            auto q = p.unit_quaternion();
                            auto t = (p.translation() - scene_translation.cast<double>()) / scene_scale;

                            strm << std::setprecision(8) << std::scientific << q.x() << " " << q.y() << " " << q.z()
                                 << " " << q.w() << " " << t.x() << " " << t.y() << " " << t.z() << "\n";

                            strm2 << 0 << "\n";
                        }
                        strm.close();
                    }
                    // switch axis to align:
                    // scene_translation = vec3(scene_translation.y(),scene_translation.x(),scene_translation.z());

                    // train.txt and eval.txt
                    {
                        std::ofstream strm(neat_dataset_path / "train.txt");
                        std::ofstream strm2(neat_dataset_path / "eval.txt");
                        for (auto ind : train_scenes->eval_samplers[i].indices)
                        {
                            strm << ind << "\n";
                            strm2 << ind << "\n";
                        }
                        strm.close();
                        strm2.close();
                    }

                    // scene_name_config_file.ini
                    NeATCombinedParams neat_config_params;
                    neat_config_params.Load("src/lib/neat-utils/neat-base-config.ini");
                    neat_config_params.train_params.scene_dir = neat_dataset_internal_dir;
                    neat_config_params.train_params.scene_name.clear();
                    neat_config_params.train_params.scene_name.push_back(scene_name_ex);
                    neat_config_params.train_params.random_seed    = params->train_params.random_seed;
                    neat_config_params.train_params.batch_size     = 16000;
                    neat_config_params.train_params.temp_image_dir = params->train_params.temp_image_dir;
                    neat_config_params.train_params.loss_tv        = params->points_adding_params.neat_tv_loss;
                    neat_config_params.train_params.experiment_dir_override =
                        std::filesystem::current_path().string() + "/" + ep_dir;
                    neat_config_params.train_params.experiment_name_str = "ct_reco_" + std::to_string(i);

                    neat_config_params.Save(neat_dataset_path / "config.ini");
                    latest_pa_state[i].latest_neat_config_path = neat_dataset_path / "config.ini";
                    latest_pa_state[i].latest_ct_reco_path = neat_config_params.train_params.experiment_dir_override +
                                                             neat_config_params.train_params.experiment_name_str;
                }
            }
        }
        train_scenes->Unload();


        if (write_test_images)
        {
            console << "Best - Worst (Eval) [" << best_loss << ", " << worst_loss << "]" << std::endl;

            for (int i = 0; i < best_batch.targets.size(); ++i)
            {
                best_batch.targets[i].save(ep_dir + "/img_best_" + to_string(best_batch.image_ids[i]) + "_target" +
                                           params->train_params.output_file_type);
                best_batch.outputs[i].save(ep_dir + "/img_best_" + to_string(best_batch.image_ids[i]) + "_output" +
                                           params->train_params.output_file_type);
            }
            for (int i = 0; i < worst_batch.targets.size(); ++i)
            {
                worst_batch.targets[i].save(ep_dir + "/img_worst_" + to_string(worst_batch.image_ids[i]) + "_target" +
                                            params->train_params.output_file_type);
                worst_batch.outputs[i].save(ep_dir + "/img_worst_" + to_string(worst_batch.image_ids[i]) + "_output" +
                                            params->train_params.output_file_type);
            }
        }

        for (auto& sd : train_scenes->data)
        {
            console << "Loss: " << std::setw(20) << sd.scene->scene->scene_name << " ";
            sd.epoch_loss.Average().Print();
        }
    }


    void TestEpoch(int epoch_id)
    {
        train_scenes->StartEpoch();

        if (params->train_params.interpolate_eval_settings)
        {
            SAIGA_ASSERT(params->train_params.optimize_eval_camera == false);
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                auto indices = train_scenes->data[i].not_training_indices;
                train_scenes->data[i].scene->camera->InterpolateFromNeighbors(indices);
            }
        }

        // Eval
        torch::NoGradGuard ngg;
        float epoch_loss           = 0;
        int num_images             = 0;
        auto [loader, loader_size] = train_scenes->DataLoader(train_scenes->test_samplers, false);

        pipeline->Train(false);

        float best_loss  = 1000000;
        float worst_loss = 0;
        ForwardResult best_batch, worst_batch;

        bool write_test_images = params->train_params.write_test_images;

        Saiga::ProgressBar bar(std::cout, "Test  " + std::to_string(epoch_id) + " |", loader_size, 30, false, 5000);
        for (std::vector<NeuralTrainData>& batch : *loader)
        {
            CHECK_EQ(1, 1);
            SAIGA_ASSERT(batch.size() == 1);
            int scene_id_of_batch = batch.front()->scene_id;
            auto& scene_data      = train_scenes->data[scene_id_of_batch];
            train_scenes->Load(device, scene_id_of_batch);
            train_scenes->Train(epoch_id, false);
            int camera_id = batch.front()->img.camera_index;

            SAIGA_ASSERT(!torch::GradMode::is_enabled());

            auto result =
                pipeline->Forward(*scene_data.scene, batch, scene_data.eval_crop_mask[camera_id], true, epoch_id,
                                  write_test_images | params->train_params.write_images_at_checkpoint);

            if (params->train_params.write_images_at_checkpoint)
            {
                for (int i = 0; i < result.image_ids.size(); ++i)
                {
                    // In average only write 20 images
                    //  if (Random::sampleBool(std::min(1.0, 20.0 / loader_size)))
                    {
                        auto err = ImageTransformation::ErrorImage(result.outputs[i], result.targets[i]);
                        TemplatedImage<ucvec3> combined(err.h, err.w + result.outputs[i].w);
                        combined.getImageView().setSubImage(0, 0, result.outputs[i].getImageView());
                        combined.getImageView().setSubImage(0, result.outputs[i].w, err.getImageView());

                        // LogImage(
                        //     tblogger.get(), combined,
                        //     "Checkpoint" + leadingZeroString(epoch_id, 4) + "/" +
                        //     scene_data.scene->scene->scene_name, result.image_ids[i]);


                        result.outputs[i].save(ep_dir + "/test/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) +
                                               params->train_params.output_file_type);

                        result.targets[i].save(ep_dir + "/test/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) + "_gt" +
                                               params->train_params.output_file_type);
                    }
                }
            }
            if (write_test_images)
            {
                if (result.float_loss.loss_float < best_loss)
                {
                    best_loss  = result.float_loss.loss_float;
                    best_batch = result;
                }

                if (result.float_loss.loss_float > worst_loss)
                {
                    worst_loss  = result.float_loss.loss_float;
                    worst_batch = result;
                }
            }

            epoch_loss += result.float_loss.loss_float;
            scene_data.epoch_loss += result.float_loss;
            num_images += batch.size();

            std::vector<ReducedImageInfo> images_rii;
            for (int b_id = 0; b_id < batch.size(); ++b_id)
            {
                images_rii.push_back(batch[b_id]->img);
            }

            bar.addProgress(batch.size());
            bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) +
                           " Avg=" + std::to_string(epoch_loss / num_images));
        }
        train_scenes->Unload();

        bar.Quit();

        if (write_test_images)
        {
            console << "Best - Worst (Eval) [" << best_loss << ", " << worst_loss << "]" << std::endl;

            for (int i = 0; i < best_batch.targets.size(); ++i)
            {
                best_batch.targets[i].save(ep_dir + "/test/img_best_" + to_string(best_batch.image_ids[i]) + "_target" +
                                           params->train_params.output_file_type);
                best_batch.outputs[i].save(ep_dir + "/test/img_best_" + to_string(best_batch.image_ids[i]) + "_output" +
                                           params->train_params.output_file_type);
            }
            for (int i = 0; i < worst_batch.targets.size(); ++i)
            {
                worst_batch.targets[i].save(ep_dir + "/test/img_worst_" + to_string(worst_batch.image_ids[i]) +
                                            "_target" + params->train_params.output_file_type);
                worst_batch.outputs[i].save(ep_dir + "/test/img_worst_" + to_string(worst_batch.image_ids[i]) +
                                            "_output" + params->train_params.output_file_type);
            }
        }

        for (auto& sd : train_scenes->data)
        {
            console << "Loss: " << std::setw(20) << sd.scene->scene->scene_name << " ";
            sd.epoch_loss.Average().Print();
        }
    }
};


CombinedParams LoadParamsHybrid(int argc, const char** argv)
{
    CLI::App app{"Train on your Scenes", "train"};

    std::string config_file;
    app.add_option("--config", config_file)->required();
    // std::cout << config_file << std::endl;
    // SAIGA_ASSERT(std::filesystem::exists(config_file));
    auto params = CombinedParams();
    params.Load(app);


    try
    {
        app.parse(argc, argv);
    }
    catch (CLI::ParseError& error)
    {
        std::cout << "Parsing failed!" << std::endl;
        std::cout << error.what() << std::endl;
        CHECK(false);
    }

    std::cout << "Loading Config File " << config_file << std::endl;
    params.Load(config_file);
    app.parse(argc, argv);

    return params;
}

void call_handler(int signal)
{
    instance->signal_handler(signal);
    exit(0);
}

int main(int argc, const char* argv[])
{
    std::cout << "Git ref: " << GIT_SHA1 << std::endl;


    std::cout << "PyTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH
              << std::endl;


    long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
    std::cout << "The cuDNN version is " << cudnn_version << std::endl;
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "The CUDA runtime version is " << runtimeVersion << std::endl;
    int version;
    cudaDriverGetVersion(&version);
    std::cout << "The driver version is " << version << std::endl;
    //    std::string config_file;

    // at::globalContext().setAllowTF32CuBLAS(true);
    // at::globalContext().setAllowTF32CuDNN(true);

    // at::globalContext().setAllowBF16ReductionCuBLAS(true);
    // at::globalContext().setAllowFP16ReductionCuBLAS(true);
    params = std::make_shared<CombinedParams>(LoadParamsHybrid(argc, argv));


    //    console << "Train Config: " << config_file << std::endl;
    //    SAIGA_ASSERT(std::filesystem::exists(config_file));


    //    params = std::make_shared<CombinedParams>(config_file);
    if (params->train_params.random_seed == 0)
    {
        std::cout << "generating random seed..." << std::endl;
        params->train_params.random_seed = Random::generateTimeBasedSeed();
    }

    {
        std::cout << "Using Random Seed: " << params->train_params.random_seed << std::endl;
        Random::setSeed(params->train_params.random_seed);
        torch::manual_seed(params->train_params.random_seed * 937545);
    }


    params->Check();
    console << "torch::cuda::cudnn_is_available() " << torch::cuda::cudnn_is_available() << std::endl;
    std::filesystem::create_directories("experiments/");

    {
        // transfer image folder to local tmp folder (mostly for clusters)
        // images should be in scene folder
        if (!params->train_params.temp_image_dir.empty())
        {
            std::cout << "Copy scene to local temp folder" << std::endl;
            std::string job_id_sub_folder = "_x_/";
            if (std::getenv("SLURM_JOBID") != nullptr)
            {
                job_id_sub_folder = "/_" + std::string(std::getenv("SLURM_JOBID")) + "_/";
            }
            std::filesystem::create_directory(params->train_params.temp_image_dir + job_id_sub_folder);
            for (int i = 0; i < params->train_params.scene_names.size(); ++i)
            {
                std::string scene_name = params->train_params.scene_names[i];
                std::string path_to_sc = params->train_params.scene_base_dir + scene_name;

                std::string path_to_tmp = params->train_params.temp_image_dir + job_id_sub_folder + scene_name;

                std::cout << "Copy " << path_to_sc << " to " << path_to_tmp << std::endl;

                std::filesystem::remove_all(path_to_tmp);
                std::filesystem::copy(path_to_sc, path_to_tmp, std::filesystem::copy_options::recursive);

                {
                    std::string file_ini = path_to_tmp + "/dataset.ini";
                    SAIGA_ASSERT(std::filesystem::exists(file_ini));
                    auto dataset_params = SceneDatasetParams(file_ini);

                    if (!std::filesystem::exists(path_to_tmp + "/images/"))
                        std::filesystem::copy(dataset_params.image_dir, path_to_tmp + "/images/",
                                              std::filesystem::copy_options::recursive);
                    // std::filesystem::remove_all(path_to_tmp+ "/images/");
                    std::cout << "replace image dir with " << path_to_tmp + "/images/" << std::endl;
                    dataset_params.image_dir = path_to_tmp + "/images/";
                    if (params->train_params.use_image_masks)
                    {
                        if (!std::filesystem::exists(path_to_tmp + "/masks/"))
                            std::filesystem::copy(dataset_params.mask_dir, path_to_tmp + "/masks/",
                                                  std::filesystem::copy_options::recursive);
                        // std::filesystem::remove_all(path_to_tmp+ "/images/");
                        std::cout << "replace mask dir with " << path_to_tmp + "/masks/" << std::endl;
                        dataset_params.mask_dir = path_to_tmp + "/masks/";
                    }

                    std::filesystem::remove(file_ini);
                    dataset_params.Save(file_ini);
                }
            }
            params->train_params.scene_base_dir = params->train_params.temp_image_dir + job_id_sub_folder;
            std::cout << "Finished copying" << std::endl;
        }
    }

    {
        // signal handler for cluster
        std::signal(SIGTERM, call_handler);
        std::signal(SIGINT, call_handler);
        std::signal(SIGHUP, call_handler);
    }

    {
        NeuralTrainer trainer;
    }

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
