/**
 * Copyright (c) 2023 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/math/Types.h"

#include "config.h"
using namespace Saiga;

inline HD vec4 compute_blending_fac(vec2 uv_pos, Matrix<float, 4, 2>* J_uv = nullptr)
{
    // derivative is 1 in relevant area
    // vec2 subpixel_pos = uv_pos - ivec2(__float2int_rd(uv_pos(0)), __float2int_rd(uv_pos(1)));
    vec2 subpixel_pos = uv_pos - uv_pos.array().floor();

    vec4 blend_vec;
    blend_vec.setZero();
    blend_vec(0) = (1 - subpixel_pos.x()) * (1 - subpixel_pos.y());
    blend_vec(1) = subpixel_pos.x() * (1 - subpixel_pos.y());
    blend_vec(2) = (1 - subpixel_pos.x()) * subpixel_pos.y();
    blend_vec(3) = subpixel_pos.x() * subpixel_pos.y();

    if (J_uv)
    {
        auto& J = *J_uv;
        J.setZero();
        J(0, 0) = subpixel_pos.y() - 1;
        J(0, 1) = subpixel_pos.x() - 1;

        J(1, 0) = 1 - subpixel_pos.y();
        J(1, 1) = -subpixel_pos.x();


        J(2, 0) = -subpixel_pos.y();
        J(2, 1) = 1 - subpixel_pos.x();

        J(3, 0) = subpixel_pos.y();
        J(3, 1) = subpixel_pos.x();
    }

    return blend_vec;
}


// #define CHANNELS 1
template <typename desc_vec, int size_of_desc_vec>
inline HD desc_vec compute_blend_vec(float alpha_dest, float alpha_s, desc_vec color, desc_vec color_dest,
                                     Saiga::Matrix<double, size_of_desc_vec, 1>* J_alphasource              = nullptr,
                                     Saiga::Matrix<double, size_of_desc_vec, size_of_desc_vec>* J_color     = nullptr,
                                     Saiga::Matrix<double, size_of_desc_vec, 1>* J_alphadest                = nullptr,
                                     Saiga::Matrix<double, size_of_desc_vec, size_of_desc_vec>* J_colordest = nullptr)
{
    desc_vec blended_col = alpha_dest * alpha_s * color + color_dest;

    if (J_alphadest)
    {
        auto& J = *J_alphadest;
        for (int i = 0; i < size_of_desc_vec; ++i)
        {
            J(i, 0) = alpha_s * color[i];
        }
    }
    if (J_alphasource)
    {
        auto& J = *J_alphasource;
        for (int i = 0; i < size_of_desc_vec; ++i)
        {
            J(i, 0) = alpha_dest * color[i];
        }
    }
    if (J_color)
    {
        auto& J = *J_color;
        for (int i = 0; i < size_of_desc_vec; ++i)
        {
            J(i, i) = alpha_dest * alpha_s;
        }
    }
    if (J_colordest)
    {
        auto& J = *J_colordest;
        for (int i = 0; i < size_of_desc_vec; ++i) J(i, i) = 1;
    }

    return blended_col;
}

inline HD float compute_blend(float alpha_dest, float alpha_s, float color, float color_dest,
                              float* J_alphasource = nullptr, float* J_color = nullptr, float* J_alphadest = nullptr,
                              float* J_colordest = nullptr)
{
    float blended_col = alpha_dest * alpha_s * color + color_dest;

    if (J_alphadest)
    {
        *J_alphadest = alpha_s * color;
    }
    if (J_alphasource)
    {
        *J_alphasource = alpha_dest * color;
    }
    if (J_color)
    {
        *J_color = alpha_dest * alpha_s;
    }
    if (J_colordest)
    {
        *J_colordest = 1;
    }

    return blended_col;
}

inline HD float compute_new_alphadest(float alphadest_old, float alpha_s, float* J_alphasource = nullptr,
                                      float* J_alphadest_old = nullptr)
{
    float new_alphadest = (1 - alpha_s) * alphadest_old;
    if (J_alphadest_old)
    {
        *J_alphadest_old = (1 - alpha_s);
    }
    if (J_alphasource)
    {
        *J_alphasource = -alphadest_old;
    }
    return new_alphadest;
}



#define CHANNELS 1
inline HD float compute_blend_d(float alpha_dest, float alpha_s, float color, float color_dest,
                                Saiga::Matrix<double, 1, CHANNELS>* J_alphasource = nullptr,
                                Saiga::Matrix<double, 1, CHANNELS>* J_color       = nullptr,
                                Saiga::Matrix<double, 1, CHANNELS>* J_alphadest   = nullptr,
                                Saiga::Matrix<double, 1, CHANNELS>* J_colordest   = nullptr)
{
    float blended_col = alpha_dest * alpha_s * color + color_dest;

    if (J_alphadest)
    {
        auto& J = *J_alphadest;
        J(0, 0) = alpha_s * color;
    }
    if (J_alphasource)
    {
        auto& J = *J_alphasource;
        J(0, 0) = alpha_dest * color;
    }
    if (J_color)
    {
        auto& J = *J_color;
        J(0, 0) = alpha_dest * alpha_s;
    }
    if (J_colordest)
    {
        auto& J = *J_colordest;
        J(0, 0) = 1;
    }

    return blended_col;
}

inline HD float compute_new_alphadest_d(float alphadest_old, float alpha_s,
                                        Saiga::Matrix<double, 1, 1>* J_alphasource   = nullptr,
                                        Saiga::Matrix<double, 1, 1>* J_alphadest_old = nullptr)
{
    float new_alphadest = (1 - alpha_s) * alphadest_old;
    if (J_alphadest_old)
    {
        auto& J = *J_alphadest_old;
        J(0, 0) = (1 - alpha_s);
    }
    if (J_alphasource)
    {
        auto& J = *J_alphasource;
        J(0, 0) = -alphadest_old;
    }
    return new_alphadest;
}


template <typename desc_vec, int size_of_desc_vec>
inline HD float normalize_by_alphadest(float color, float alphadest, Saiga::Matrix<double, 1, 1>* J_alphadest = nullptr,
                                       Saiga::Matrix<double, size_of_desc_vec, 1>* J_color = nullptr)
{
    SAIGA_ASSERT(false);  // not implemented
    float result = color / (1 - alphadest);
    if (J_alphadest)
    {
        auto& J = *J_alphadest;
        J(0, 0) = color / ((1 - alphadest) * (1 - alphadest));
    }
    if (J_color)
    {
        auto& J = *J_color;
        J(0, 0) = 1.0 / (1 - alphadest);
    }
    return result;
}