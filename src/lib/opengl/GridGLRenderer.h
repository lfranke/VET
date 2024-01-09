/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
 #pragma once

#include "NeuralPointCloudOpenGL.h"
#include "models/Pipeline.h"

template <typename T>
std::vector<T> tensor_to_vector(torch::Tensor t)
{
    auto t_x = t.contiguous().cpu();
    SAIGA_ASSERT(t.sizes().size() == 2);
    SAIGA_ASSERT(t.dtype() == torch::kFloat);
    uint floats_per_elem = sizeof(T) / sizeof(float);
    std::vector<T> vec(t_x.numel() / floats_per_elem);
    // std::cout << t.sizes() << t.numel()<< std::endl;
    std::memcpy((float*)vec.data(), t_x.data_ptr<float>(), t_x.numel() * sizeof(float));
    //   std::cout << t_x.numel() << std::endl;

    // std::vector<T> vec = std::vector<T>(t_x.data_ptr<float>(), t_x.data_ptr<float>()+t_x.numel());
    return vec;
}

class GridGLRenderer : public Saiga::Object3D
{
   public:
    GridGLRenderer(NeuralPointCloudCuda neural_pc);

    void render(const FrameData& fd, float cutoff_val, int mode, bool cutoff_as_percent = false);

   private:
    std::vector<float> sorted_grid_values;

    UniformBuffer ocam_model_ssbo;

    std::shared_ptr<Saiga::Shader> shader;

    Saiga::VertexBuffer<PositionIndex> gl_grid;

    Saiga::TemplatedBuffer<vec4> gl_normal = {GL_ARRAY_BUFFER};
    Saiga::TemplatedBuffer<vec4> gl_color  = {GL_ARRAY_BUFFER};
    Saiga::TemplatedBuffer<vec4> gl_data   = {GL_ARRAY_BUFFER};
};
