/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "adop_viewer.h"

#include "saiga/core/geometry/cameraAnimation.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/exif/TinyEXIF.h"

#include "git_sha1.h"


ADOPViewer::ADOPViewer(std::string scene_dir, std::unique_ptr<DeferredRenderer> renderer_,
                       std::unique_ptr<WindowType> window_)
    : StandaloneWindow<wm, DeferredRenderer>(std::move(renderer_), std::move(window_))
{
    main_menu.AddItem(
        "Saiga", "MODEL", [this]() { view_mode = ViewMode::MODEL; }, GLFW_KEY_F1, "F1");


    main_menu.AddItem(
        "Saiga", "NEURAL", [this]() { view_mode = ViewMode::NEURAL; }, GLFW_KEY_F2, "F2");


    main_menu.AddItem(
        "Saiga", "SPLIT_NEURAL", [this]() { view_mode = ViewMode::SPLIT_NEURAL; }, GLFW_KEY_F3, "F3");

    main_menu.AddItem(
        "Saiga", "DEBUG_ONLY", [this]() { view_mode = ViewMode::DEBUG_ONLY; }, GLFW_KEY_F4, "F4");

    main_menu.AddItem(
        "Saiga", "GT_ONLY", [this]() { view_mode = ViewMode::GT_ONLY; }, GLFW_KEY_F6, "F6");

    std::cout << "Program Initialized!" << std::endl;

    std::cout << "Loading Scene " << scene_dir << std::endl;
    LoadScene(scene_dir);
    LoadSceneImpl();

    std::filesystem::create_directories("videos/");
    recording_dir = "videos/" + scene->scene->scene_name + "/";
    view_mode     = ViewMode::SPLIT_NEURAL;
}


void ADOPViewer::LoadSceneImpl()
{
    if (renderer->tone_mapper.auto_exposure || renderer->tone_mapper.auto_white_balance)
    {
        renderer->tone_mapper.download_tmp_values = true;
    }

    renderer->lighting.pointLights.clear();
    renderer->lighting.directionalLights.clear();
    renderer->lighting.spotLights.clear();

    ::camera = &scene->scene_camera;
    window->setCamera(camera);

    if (scene->scene->point_cloud.NumVertices() > 15000000)
    {
        // by default don't render very large point clouds in the viewport
        render_points = false;
    }
    auto& f = scene->scene->frames.front();
    camera->setModelMatrix(f.OpenglModel());
    camera->updateFromModel();

    renderer->params.useSSAO = false;
    renderer->params.hdr     = false;
    sun                      = std::make_shared<DirectionalLight>();
    sun->ambientIntensity    = exp2(scene->scene->dataset_params.scene_exposure_value);
    sun->intensity           = 0;
    renderer->lighting.AddLight(sun);
    renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;
}

void ADOPViewer::render(RenderInfo render_info)
{
    // update states:
    if (neural_renderer)
    {
        neural_renderer->custom_discard_aabb = scene->use_custom_aabb_to_cut_out ? scene->custom_aabb : AABB();
        neural_renderer->render_env_map      = !scene->no_env_map;
    }


    if (renderObject && render_info.render_pass == RenderPass::Deferred)
    {
        if (!scene->model.mesh.empty())
        {
            if (renderTexture && scene->model.mesh.front().HasTC())
            {
                if (!object_tex)
                {
                    object_tex = std::make_shared<TexturedAsset>(scene->model);
                }
                object_tex->render(render_info.camera, mat4::Identity());
            }
            else
            {
                if (!object_col)
                {
                    scene->model.ComputeColor();
                    auto mesh = scene->model.CombinedMesh(VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR).first;
                    mesh.RemoveDoubles(0.001);
                    // mesh.SmoothVertexColors(10, 0);
                    object_col = std::make_shared<ColoredAsset>(mesh);
                }
                object_col->render(render_info.camera, mat4::Identity());
            }
        }
    }

    if (render_info.render_pass == RenderPass::Final)
    {
        if (view_mode == ViewMode::NEURAL || view_mode == ViewMode::DEBUG_ONLY || view_mode == ViewMode::GT_ONLY)
        {
            auto fd = scene->CurrentFrameData();

            fd.w              = fd.w * render_scale;
            fd.h              = fd.h * render_scale;
            fd.crop_transform = fd.crop_transform.scale(render_scale);


            fd.exposure_value = renderer->tone_mapper.params.exposure_value;
            fd.white_balance  = renderer->tone_mapper.params.white_point;

            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
            }
            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;
            if (view_mode == ViewMode::NEURAL)
            {
                neural_renderer->timer_system.BeginFrame();
                neural_renderer->Render(fd, debug_directionality ? debug_refl_dir : vec3(0, 0, 0));
                neural_renderer->timer_system.EndFrame();
                int rotate_result = rotate_result_90deg == 0 ? 0 : (rotate_result_90deg > 0 ? 1 : -1);
                if (neural_renderer->use_gl_tonemapping)
                {
                    display.render(neural_renderer->output_texture_ldr.get(), ivec2(0, 0), renderer->viewport_size,
                                   true, rotate_result);
                }
                else
                {
                    display.render(neural_renderer->output_texture.get(), ivec2(0, 0), renderer->viewport_size, true,
                                   rotate_result);
                }
            }
            else if (view_mode == ViewMode::DEBUG_ONLY)
            {
                neural_renderer->timer_system.BeginFrame();

                neural_renderer->ns->dynamic_refinement_t = torch::zeros({1}).cuda();
                neural_renderer->ns->texture->PrepareTexture(false);
                neural_renderer->ns->texture->PrepareConfidence(0.f);
                neural_renderer->RenderColor(fd, neural_renderer->color_flags);
                neural_renderer->timer_system.EndFrame();

                if (neural_renderer->use_gl_tonemapping)
                {
                    display.render(neural_renderer->output_color.get(), ivec2(0, 0), renderer->viewport_size, true);
                }
                else
                {
                    display.render(neural_renderer->output_color.get(), ivec2(0, 0), renderer->viewport_size, true);
                }
            }
            else
            {
                auto tex = neural_renderer->getClosestGTImage(fd);
                display.render(tex.get(), ivec2(0, 0), renderer->viewport_size, true);
            }
        }
    }

    // Debug view of point cloud + image frames
    if ((view_mode == ViewMode::SPLIT_NEURAL || view_mode == ViewMode::MODEL) &&
        render_info.render_pass == RenderPass::Forward)
    {
        if (render_debug)
        {
            scene->RenderDebug(render_info.camera);
        }


        if (spline_mesh)
        {
            glLineWidth(3);
            spline_mesh->renderForward(render_info.camera, mat4::Identity());
            glLineWidth(1);
        }

        if (render_points)
        {
            if (!scene->gl_points)
            {
                scene->gl_points = std::make_shared<NeuralPointCloudOpenGL>(scene->scene->point_cloud);
            }
            static std::shared_ptr<NeuralPipeline> last_pipeline;
            if (render_grid && (!grid_renderer || last_pipeline != neural_renderer->pipeline))
            {
                std::cout << "create grid mesh" << std::endl;
                grid_renderer = std::make_shared<GridGLRenderer>(neural_renderer->ns->point_cloud_cuda);
            }
            // Create a frame around the current gl-camera
            FrameData fd;
            fd.w              = renderer->viewport_size.x();
            fd.h              = renderer->viewport_size.y();
            fd.pose           = Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>();
            fd.K              = GLProjectionMatrix2CVCamera(camera->proj, fd.w, fd.h);
            fd.exposure_value = scene->scene->dataset_params.scene_exposure_value;
            if (scene->scene->scene_cameras.front().camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                fd.distortion = scene->scene->scene_cameras.front().distortion;
            }
            else
            {
                fd.ocam.poly_cam2world.clear();
                for (auto d : scene->scene->scene_cameras.front().ocam.poly_cam2world)
                {
                    fd.ocam.poly_cam2world.push_back(d);
                }
                fd.ocam.poly_world2cam.clear();
                for (auto d : scene->scene->scene_cameras.front().ocam.poly_world2cam)
                {
                    fd.ocam.poly_world2cam.push_back(d);
                }
                fd.ocam.c  = scene->scene->scene_cameras.front().ocam.c;
                fd.ocam.d  = scene->scene->scene_cameras.front().ocam.d;
                fd.ocam.e  = scene->scene->scene_cameras.front().ocam.e;
                fd.ocam.cx = scene->scene->scene_cameras.front().ocam.cx;
                fd.ocam.cy = scene->scene->scene_cameras.front().ocam.cy;
            }

            // fd.camera_model_type = scene->scene->dataset_params.camera_model;
            {
                auto tim = renderer->timer->Measure("GL_POINTS render");
                if (render_grid)
                    grid_renderer->render(fd, grid_cutoff_val, grid_mode, grid_cutoff_as_percent);
                else
                    scene->gl_points->render(fd, 0);
            }
            if (render_coordinate_system)
            {
                static std::shared_ptr<ColoredAsset> coordinate_mesh;
                if (!coordinate_mesh || last_pipeline != neural_renderer->pipeline)
                    coordinate_mesh = std::make_shared<ColoredAsset>(CoordinateSystemMesh(0.2, true));

                coordinate_mesh->renderForward(camera, Saiga::mat4::Identity());
            }

            if (render_scene_aabb)
            {
                /*{
                    static std::shared_ptr<LineVertexColoredAsset> bounding_box_mesh;
                    if (!bounding_box_mesh || last_pipeline != neural_renderer->pipeline)
                    {
                        auto aabb = scene->scene->point_cloud.BoundingBox();
                        std::cout << aabb << std::endl;
                        UnifiedMesh bb_scene_unified_mesh = GridBoxLineMesh(aabb);
                        if (neural_renderer)
                        {
                            if (neural_renderer->experiments.size() > 0)
                            {
                                auto pap = neural_renderer->params->points_adding_params;
                                if (pap.custom_aabb_max != vec3(0, 0, 0) && pap.custom_aabb_min != vec3(0, 0, 0))
                                {
                                    bb_scene_unified_mesh =
                                        GridBoxLineMesh(AABB(pap.custom_aabb_min, pap.custom_aabb_max));
                                }
                            }
                        }
                        bb_scene_unified_mesh.SetVertexColor(vec4(1, 1, 1, 1));
                        bounding_box_mesh = std::make_shared<LineVertexColoredAsset>(bb_scene_unified_mesh);
                    }
                    if (bounding_box_mesh)
                    {
                        bounding_box_mesh->renderForward(camera, Saiga::mat4::Identity());
                    }
                }*/
                {
                    static std::shared_ptr<LineVertexColoredAsset> bounding_box_mesh;
                    if (!bounding_box_mesh || last_pipeline != neural_renderer->pipeline)
                    {
                        auto aabb = scene->scene->dataset_params.aabb;
                        std::cout << aabb << std::endl;
                        UnifiedMesh bb_scene_unified_mesh = GridBoxLineMesh(aabb);
                        bb_scene_unified_mesh.SetVertexColor(vec4(1, 1, 1, 1));
                        bounding_box_mesh = std::make_shared<LineVertexColoredAsset>(bb_scene_unified_mesh);
                    }
                    if (bounding_box_mesh)
                    {
                        bounding_box_mesh->renderForward(camera, Saiga::mat4::Identity());
                    }
                }
            }


            /*
                        {
                            static std::shared_ptr<LineVertexColoredAsset> bounding_box_mesh;
                            if (!bounding_box_mesh)
                            {
                                auto aabb = scene->scene->point_cloud.BoundingBox();
                                //     std::cout << aabb << std::endl;
                                UnifiedMesh bb_scene_unified_mesh = GridBoxLineMesh(aabb);
                                if (neural_renderer)
                                {
                                    if (neural_renderer->experiments.size() > 0)
                                    {
                                        auto pap = neural_renderer->params->points_adding_params;
                                        if (pap.custom_aabb_max != vec3(0, 0, 0) && pap.custom_aabb_min != vec3(0, 0,
               0))
                                        {
                                            bb_scene_unified_mesh = GridBoxLineMesh(AABB(pap.custom_aabb_min,
               pap.custom_aabb_max));
                                        }
                                    }
                                }
                                bb_scene_unified_mesh.SetVertexColor(vec4(1, 1, 1, 1));
                                bounding_box_mesh = std::make_shared<LineVertexColoredAsset>(bb_scene_unified_mesh);
                            }
                            if (bounding_box_mesh)
                            {
                                glLineWidth(2);
                                bounding_box_mesh->renderForward(camera, Saiga::mat4::Identity());
                                glLineWidth(1);
                            }
                        }*/

            if (neural_renderer && scene->render_experiment_sphere)
            {
                if (neural_renderer->experiments.size() > 0)
                {
                    auto pp = neural_renderer->params->pipeline_params;
                    scene->RenderMultiscalarSpheres(render_info.camera, last_pipeline != neural_renderer->pipeline,
                                                    pp.env_spheres, pp.env_inner_radius, pp.env_radius_factor);
                }
            }
            if (scene->render_custom_sphere)
            {
                scene->RenderMultiscalarSpheres(render_info.camera, scene->custom_sphere_dirty_flag,
                                                scene->custom_num_spheres, scene->custom_inner_radius,
                                                scene->custom_radius_factor);
                scene->custom_sphere_dirty_flag = false;
            }

            if (last_pipeline != neural_renderer->pipeline) last_pipeline = neural_renderer->pipeline;
        }



        if (renderWireframe)
        {
            glEnable(GL_POLYGON_OFFSET_LINE);
            //        glLineWidth(1);
            glPolygonOffset(0, -500);

            // object.renderWireframe(cam);
            glDisable(GL_POLYGON_OFFSET_LINE);
        }
    }

    if (render_info.render_pass == RenderPass::GUI)
    {
        if (ImGui::Begin("Video Recording"))
        {
        }
        ImGui::End();
        if (ImGui::Begin("Extra"))
        {
        }
        ImGui::End();

        ViewerBase::imgui();

        ImGui::Begin("Model Viewer");

        ImGui::SliderFloat("render_scale", &render_scale, 0.1, 2);
        ImGui::Checkbox("use debug directionality instead", &debug_directionality);

        ImGui::Direction("debug_refl_dir", debug_refl_dir);
        debug_refl_dir.normalize();


        if (neural_renderer)
        {
            if (ImGui::Button("Set to closest frame"))
            {
                auto& f = scene->scene->frames[neural_renderer->current_best_gt];
                ::camera->setModelMatrix(f.OpenglModel());
                ::camera->updateFromModel();
                renderer->tone_mapper.params.exposure_value = f.exposure_value;
                renderer->tone_mapper.params_dirty          = true;
            }
        }
        ImGui::End();

        auto fd = scene->CurrentFrameData();
        /*
            if (scene->scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
            {
                fd.w = fd.w * render_scale * scene->scene->dataset_params.render_scale;
                fd.h = fd.h * render_scale * scene->scene->dataset_params.render_scale;
                // Linus: I think this is not used as expected
                fd.K = fd.K.scale(scene->scene->dataset_params.render_scale);
            }
            else if (scene->scene->dataset_params.camera_model == CameraModel::OCAM)
            {
                fd.w              = fd.w * render_scale * scene->scene->dataset_params.render_scale;
                fd.h              = fd.h * render_scale * scene->scene->dataset_params.render_scale;
                fd.crop_transform = fd.crop_transform.scale(render_scale *
                scene->scene->dataset_params.render_scale);
                }
        */
        fd.w = fd.w * render_scale;
        fd.h = fd.h * render_scale;
        // fd.K = fd.K.scale(render_scale);
        fd.crop_transform = fd.crop_transform.scale(render_scale);

        fd.exposure_value = renderer->tone_mapper.params.exposure_value;
        fd.white_balance  = renderer->tone_mapper.params.white_point;

        if (view_mode == ViewMode::SPLIT_NEURAL)
        {
            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
            }
            mouse_in_gt                              = neural_renderer->mouse_on_view;
            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;

            // neural_renderer->tone_mapper.params_dirty |= renderer->tone_mapper.params_dirty;
            neural_renderer->Forward(fd, rotate_result_90deg, debug_directionality ? debug_refl_dir : vec3(0, 0, 0));
        }

        if (ImGui::Begin("Video Recording"))
        {
            Recording(fd);
        }
        ImGui::End();

        if (ImGui::Begin("Extra"))
        {
            ImGui::Checkbox("render Grid", &render_grid);
            ImGui::SliderFloat("cutoff val", &grid_cutoff_val, 0.f, 2.f);
            ImGui::Checkbox("cutoff as percent", &grid_cutoff_as_percent);


            ImGui::SliderInt("grid color mode", &grid_mode, 0, 2);


            ImGui::SliderInt("rotate results 90 degrees", &rotate_result_90deg, -1, 1);

            if (ImGui::Button("rotate 90 deg"))
            {
                camera->rotateGlobal(camera->getDirection().head<3>(), 90.f);
            }
            static bool keep_up_vector_fixed = false;
            ImGui::Checkbox("fix up vector", &keep_up_vector_fixed);
            if (keep_up_vector_fixed)
            {
                vec3 pos    = camera->getPosition().head<3>();
                vec3 center = (pos - vec3(1, 0, 0)).eval();
                vec3 up     = vec3(0, 0, 1);
                camera->setView(pos, center, up);
            }
        }
        ImGui::End();
    }
}

void ADOPViewer::Recording(ImageInfo& fd)
{
    std::string out_dir = recording_dir;

    static bool is_recording = false;
    static bool downscale_gt = false;

    static bool interpolate_exposure = false;

    static bool record_debug  = false;
    static bool record_gt     = false;
    static bool record_neural = false;
    ImGui::Checkbox("record_debug", &record_debug);
    ImGui::Checkbox("record_gt", &record_gt);
    ImGui::Checkbox("record_neural", &record_neural);
    ImGui::Checkbox("downscale_gt", &downscale_gt);


    static std::vector<SplineKeyframe> traj;


    auto insert = [this](int id, int pos = -1)
    {
        SplineKeyframe kf;
        kf.user_index = id;
        kf.pose       = scene->scene->frames[id].pose;
        if (pos >= 0)
            camera_spline.Insert(kf, pos);
        else
            camera_spline.Insert(kf);
    };

    bool update_curve         = false;
    static bool hdr_video_gen = false;

    static int current_frame = 0;

    if (is_recording)
    {
        std::stringstream ssf;
        ssf << std::setfill('0') << std::setw(5) << std::to_string(current_frame) << ".png";
        std::string frame_name = ssf.str();  // std::to_string(current_frame) + ".png";

        if (record_neural)
        {
            SAIGA_ASSERT(neural_renderer);
            auto frame = neural_renderer->DownloadRender();
            frame.save(out_dir + "/neural/" + frame_name);
        }
        if (record_debug)
        {
            SAIGA_ASSERT(neural_renderer);
            auto frame = neural_renderer->DownloadColor();
            frame.save(out_dir + "/debug/" + frame_name);
        }

        if (record_gt)
        {
            SAIGA_ASSERT(neural_renderer);
            TemplatedImage<ucvec4> frame = neural_renderer->DownloadGt();

            if (downscale_gt)
            {
                TemplatedImage<ucvec4> gt_small(frame.h / 2, frame.w / 2);
                frame.getImageView().copyScaleDownPow2(gt_small.getImageView(), 2);
                frame = gt_small;
            }
            frame.save(out_dir + "/gt/" + frame_name);
        }

        current_frame++;
    }

    if (ImGui::Button("fly through all"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 30;
        std::vector<int> kfs(scene->scene->frames.size());
        std::iota(kfs.begin(), kfs.end(), 0);
        kfs.insert(kfs.begin(), 0);
        kfs.insert(kfs.end(), 0);
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = scene->scene->frames[kfs[0]].exposure_value;
        renderer->tone_mapper.params_dirty          = true;

        update_curve = true;
    }

    auto add_frames = [&](std::vector<int> kfs, std::vector<Sophus::SE3d> custom_poses = {}, int framerate = 60)
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = framerate;

        int custom_poses_idx = 0;
        for (auto i : kfs)
        {
            if (i >= 0)
                insert(i);
            else
            {
                SplineKeyframe kf;
                kf.user_index = i;
                kf.pose       = custom_poses[custom_poses_idx];
                camera_spline.Insert(kf);
                ++custom_poses_idx;
            }
        }
        renderer->tone_mapper.params.exposure_value = 0;
        renderer->tone_mapper.params_dirty          = true;

        update_curve = true;
    };

    // {220, 8, 29, 51, 66, 81, 95, 106, 124, 127, 146, 155, 173, 182, 198, 211, 220, 220}
    if (ImGui::Button("preset owngarden4"))
        add_frames({0,   -1,  -1,  -1, -1, -1, -1, 17, 5,   2,   -1,  -1,  -1,
                    254, 263, 275, -1, -1, -1, -1, -1, 428, 281, 282, 283, 284},
                   {
                       {{0.962682, 0.162321, 0.171413, 0.132334}, {0.380364, 3.590672, -3.250923}},
                       {{0.777176, 0.095162, 0.516101, 0.347248}, {-0.433170, 2.730655, -2.624574}},
                       {{0.623156, 0.057073, 0.643658, 0.440595}, {-0.653258, 1.972846, -1.981532}},
                       {{0.233948, -0.002828, 0.806650, 0.542748}, {0.511080, 1.188083, -0.999644}},
                       {{-0.160876, -0.085971, 0.803120, 0.567210}, {1.971140, 1.208635, -0.917601}},
                       {{0.645914, 0.123225, -0.634515, -0.406202}, {3.394852, 2.178853, -1.953768}},
                       {{0.941449, 0.267970, -0.195602, -0.060048}, {1.083020, 3.646496, -3.205715}},
                       {{0.954096, 0.285966, -0.085804, -0.023697}, {-0.369368, 2.671801, -2.205998}},
                       {{0.975064, 0.211805, -0.061409, 0.024835}, {-1.391991, 0.893331, -2.277236}},
                       {{0.276099, -0.145801, 0.696566, 0.645993}, {-1.705122, -3.446581, 3.637299}},
                       {{-0.318042, -0.125955, 0.662801, 0.666093}, {-0.990073, -3.415712, 3.581855}},
                       {{0.651899, -0.184908, -0.415515, -0.606781}, {-1.173580, -3.185102, 3.101508}},
                       {{0.934022, -0.218890, -0.215242, -0.182650}, {-1.198260, -2.806252, 2.994166}},
                       {{0.985221, -0.141413, 0.043185, 0.086469}, {-1.548519, -2.703350, 2.793318}},
                   });



    if (ImGui::Button("preset m360_garden")) add_frames({0, 35, 38, 41, 78, 111, 113, 115, 117, 120, 91, 63, 0});
    if (ImGui::Button("preset m360_stump")) add_frames({0, 40, 42, 64, 66, 48, 32, 35, 0});
    if (ImGui::Button("preset m360_counter"))
        add_frames({0, 213, 196, 171, 158, 145, 130, 121, 108, 93, 81, 66, 60, 43, 36, 22, 0, 0});


    if (ImGui::Button("preset cherry")) add_frames({0, 7, 230, 195, 175, 136, 90, 41, 0});

    if (ImGui::Button("preset bicycle")) add_frames({0, 0, 44, 40, 38, 36, 34, 30, 76, 70, 67, 62, 60, 54, 50, 0, 0});

    if (ImGui::Button("preset lighthouse")) add_frames({108, 108, 121, 138, 6, 43, 71, 90, 108, 108});

    if (ImGui::Button("preset spheres"))
        add_frames({0, 0, 143, 266, 402, 517, 635, 698, 880, 1004, 1148, 1241, 1285, 1332, 0, 0});

    if (ImGui::Button("preset m60")) add_frames({0, 0, 148, 27, 190, 67, 84, 90, 96, 277, 290, 303, 0, 0});

    if (ImGui::Button("preset face")) add_frames({0, 0, 38, 79, 96, 133, 169, 209, 251, 291, 335, 881, 891, 0, 0});

    if (ImGui::Button("preset chairs")) add_frames({0, 530, 891, 1232, 1376, 1002, 578, 213, 0});

    if (ImGui::Button("preset opera"))
        add_frames({0, 0, 18, 31, 43, 62, 74, 87, 98, 113, 130, 146, 158, 169, 177, 187, 193, 0, 0});

    //   add_frames({0, 12, 111, 120, 134, 144, 152, 163, 178, 4, 22, 49, 71, 86, 98, 105, 106});

    if (ImGui::Button("preset onetrade"))
        add_frames({421, 439, 446, 449, 109, 459, 467, 480, 491, 503, 519, 533, 534, 537, 544,
                    545, 551, 565, 573, 579, 588, 601, 631, 644, 655, 3,   51,  56,  59,  69,
                    73,  78,  80,  108, 458, 456, 448, 447, 135, 153, 394, 400, 408, 420, 421});

    if (ImGui::Button("preset motorcycle")) add_frames({0, 265, 315, 83, 342, 108, 394, 187, 196, 460, 0});

    if (ImGui::Button("preset playground")) add_frames({0, 0, 23, 52, 73, 88, 98, 136, 157, 170, 0, 0});

    if (ImGui::Button("preset cars"))
        add_frames({0, 48, 88, 92, 96, 100, 104, 116, 120, 128, 140, 156, 176, 192, 204, 208, 216, 0});


    if (ImGui::Button("preset office"))
        add_frames({57, 69, 99, 203, 11, 317, 298, 365, 390, 455, 454, 493, 658, 501, 517});

    if (ImGui::Button("preset train"))
        add_frames({0, 0, 13, 20, 30, 40, 50, 207, 95, 100, 110, 120, 130, 140, 150, 160, 0, 0});

    if (ImGui::Button("preset kemenate"))
    {
        ::camera->position = vec4(7.79858, 1.07699, -0.849739, 1);
        ::camera->rot      = quat(0.731483, -0.00347084, 0.68185, 0.00113975);
    }


    if (ImGui::Button("preset boat"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 60;

        std::vector<int> kfs = {// outer circle
                                568, 568, 568, 590, 679, 556, 68, 101, 146, 190, 224, 290, 346, 446, 490, 590, 194,
                                // inner
                                442, 447, 462, 474, 475, 477, 372,
                                // close up sign
                                543, 548, 548, 541, 398,
                                // end
                                396, 664, 663, 663, 663};
        for (auto i : kfs)
        {
            insert(i);
        }

        renderer->tone_mapper.params.exposure_value = 14.1;
        renderer->tone_mapper.params_dirty          = true;
        renderer->tone_mapper.tm_operator           = 4;

        neural_renderer->use_gl_tonemapping = true;

        camera_spline.time_in_seconds = 30;
        downscale_gt                  = true;

        update_curve = true;
    }

    if (ImGui::Button("preset boat hdr stuff"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 30;

        std::vector<int> kfs = {663, 663, 663, 663};
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = 14.1;
        renderer->tone_mapper.params_dirty          = true;
        renderer->tone_mapper.tm_operator           = 4;


        auto exp_ref    = scene->scene->frames[kfs[0]].exposure_value;
        auto exp_target = 13;

        std::vector<double> x;
        x.push_back(1);
        for (auto& k : camera_spline.keyframes)
        {
            k.user_data = x;
        }


        camera_spline.keyframes[0].user_data[0] = exp_ref;
        camera_spline.keyframes[1].user_data[0] = exp_ref;
        camera_spline.keyframes[2].user_data[0] = exp_target;
        camera_spline.keyframes[3].user_data[0] = exp_target;

        hdr_video_gen = true;

        camera_spline.time_in_seconds = 15;

        update_curve = true;
    }

    if (ImGui::Button("playground extrapolation1"))
    {
        ::camera->position = vec4(2.97414, -2.65628, 4.64867, 1);
        ::camera->rot      = quat(0.0477469, 0.270376, -0.280752, 0.919671);
        ::camera->calculateModel();
        ::camera->updateFromModel();

        std::cout << Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>() << std::endl;
        std::cout << Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>().inverse() << std::endl;
    }

    if (ImGui::Button("playground extrapolation2"))
    {
        camera->position = vec4(-3.72784, -2.68887, -5.74922, 1);
        camera->rot      = quat(-0.190701, -0.93891, -0.0748287, 0.276543);
        ::camera->calculateModel();
        ::camera->updateFromModel();
        std::cout << Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>() << std::endl;
        std::cout << Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>().inverse() << std::endl;
    }



    update_curve |= camera_spline.imgui();
    static int sel_kf = 0;

    if (camera_spline.selectedKeyframe != sel_kf)
    {
        scene->selected_capture = camera_spline.keyframes[camera_spline.selectedKeyframe].user_index;
        if (scene->selected_capture < 0)
        {
            auto OpenglModel = [](Sophus::SE3d pose) { return pose.matrix().cast<float>() * CV2GLView(); };
            camera->setModelMatrix(OpenglModel(camera_spline.keyframes[camera_spline.selectedKeyframe].pose));
        }
        else
            camera->setModelMatrix(scene->scene->frames[scene->selected_capture].OpenglModel());
        camera->updateFromModel();
        sel_kf = camera_spline.selectedKeyframe;
    }
    static int insert_at = -1;
    ImGui::InputInt("insert at (-1==end)", &insert_at);
    ImGui::SameLine();
    if (ImGui::ArrowButton("##Left", ImGuiDir_Left))
    {
        --insert_at;
        if (insert_at < -1) insert_at = -1;
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##Right", ImGuiDir_Right))
    {
        ++insert_at;
        if (insert_at > camera_spline.keyframes.size()) insert_at = camera_spline.keyframes.size();
    }

    if (ImGui::Button("add yellow frame"))
    {
        if (scene->selected_capture >= 0)
        {
            insert(scene->selected_capture, insert_at);
            update_curve = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("remove yellow frame"))
    {
        int id = -1;
        for (int i = 0; i < camera_spline.keyframes.size(); ++i)
        {
            if (camera_spline.keyframes[i].user_index == scene->selected_capture)
            {
                id = i;
            }
        }
        std::cout << "remove id " << id << " selected " << scene->selected_capture << std::endl;
        if (id != -1)
        {
            camera_spline.keyframes.erase(camera_spline.keyframes.begin() + id);
            update_curve = true;
        }
    }

    if (ImGui::Button("add gl camera"))
    {
        SplineKeyframe kf;
        kf.user_index = -1;
        kf.pose       = Sophus::SE3f::fitToSE3(scene->scene_camera.model * GL2CVView()).cast<double>();
        camera_spline.Insert(kf, insert_at);
        update_curve = true;
    }

    if (update_curve)
    {
        for (auto& f : scene->scene->frames)
        {
            f.display_color = vec4(1, 0, 0, 1);
        }
        for (auto& kf : camera_spline.keyframes)
        {
            scene->scene->frames[kf.user_index].display_color = vec4(0, 1, 0, 1);
        }
        camera_spline.updateCurve();

        auto mesh = camera_spline.ProxyMesh();
        if (mesh.NumVertices() > 0)
        {
            spline_mesh = std::make_shared<LineVertexColoredAsset>(
                mesh.SetVertexColor(exp2(scene->scene->dataset_params.scene_exposure_value) * vec4(0, 1, 0, 1)));
        }
    }

    if (!is_recording && ImGui::Button("start recording"))
    {
        SAIGA_ASSERT(neural_renderer);
        neural_renderer->current_best_gt = -1;
        is_recording                     = true;
        traj                             = camera_spline.Trajectory();
        current_frame                    = 0;
        std::filesystem::create_directories(out_dir);
        if (record_debug)
        {
            std::filesystem::remove_all(out_dir + "/debug");
            std::filesystem::create_directories(out_dir + "/debug");
        }
        if (record_gt)
        {
            std::filesystem::remove_all(out_dir + "/gt");
            std::filesystem::create_directories(out_dir + "/gt");
        }
        if (record_neural)
        {
            std::filesystem::remove_all(out_dir + "/neural");
            std::filesystem::create_directories(out_dir + "/neural");
        }
    }

    if (is_recording)
    {
        if (current_frame == traj.size() || ImGui::Button("stop recording"))
        {
            is_recording = false;
        }
    }

    if (is_recording && !traj.empty())
    {
        auto frame = traj[current_frame];

        mat4 model = frame.pose.matrix().cast<float>() * CV2GLView();

        if (interpolate_exposure)
        {
            float alpha  = 0.001;
            auto new_exp = scene->scene->frames[neural_renderer->current_best_gt].exposure_value;
            renderer->tone_mapper.params.exposure_value =
                (1 - alpha) * renderer->tone_mapper.params.exposure_value + alpha * new_exp;
            renderer->tone_mapper.params_dirty = true;
        }


        if (hdr_video_gen)
        {
            auto new_exp                                = frame.user_data[0];
            renderer->tone_mapper.params.exposure_value = new_exp;
            renderer->tone_mapper.params_dirty          = true;
        }

        ::camera->setModelMatrix(model);
        ::camera->updateFromModel();
    }
}
static void set_imgui_dark_theme()
{
    ImGui::StyleColorsDark();

    auto& colors              = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg] = ImVec4{0.1f, 0.105f, 0.11f, 1.0f};

    // Headers
    colors[ImGuiCol_Header]        = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_HeaderHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_HeaderActive]  = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

    // Buttons
    colors[ImGuiCol_Button]        = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_ButtonHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_ButtonActive]  = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

    // Frame BG
    colors[ImGuiCol_FrameBg]        = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_FrameBgHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_FrameBgActive]  = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

    // Tabs
    colors[ImGuiCol_Tab]                = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TabHovered]         = ImVec4{0.38f, 0.3805f, 0.381f, 1.0f};
    colors[ImGuiCol_TabActive]          = ImVec4{0.28f, 0.2805f, 0.281f, 1.0f};
    colors[ImGuiCol_TabUnfocused]       = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};

    // Title
    colors[ImGuiCol_TitleBg]          = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TitleBgActive]    = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};


    ImGuiStyle& style     = ImGui::GetStyle();
    style.FrameBorderSize = 1.f;
    style.FramePadding    = ImVec2(5.f, 2.f);

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding        = 0.f;
        colors[ImGuiCol_WindowBg].w = 1.f;
    }
}
int main(int argc, char* argv[])
{
    std::cout << "Git ref: " << GIT_SHA1 << std::endl;

#ifdef _WIN32
    std::filesystem::current_path(EXECUTABLE_DIR);
#endif
    float render_scale = 1.0f;
    std::string scene_dir;
    CLI::App app{"Viewer for Scenes", "viewer"};
    app.add_option("--scene_dir", scene_dir)->required();
    app.add_option("--render_scale", render_scale);
    CLI11_PARSE(app, argc, argv);

    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    WindowParameters windowParameters;
    OpenGLParameters openglParameters;
    DeferredRenderingParameters rendererParameters;
    windowParameters.fromConfigFile("config.ini");
    rendererParameters.hdr = true;

    auto window   = std::make_unique<WindowType>(windowParameters, openglParameters);
    auto renderer = std::make_unique<DeferredRenderer>(*window, rendererParameters);


    MainLoopParameters mlp;
    set_imgui_dark_theme();
    ADOPViewer viewer(scene_dir, std::move(renderer), std::move(window));
    viewer.render_scale = render_scale;
    viewer.run(mlp);
}
