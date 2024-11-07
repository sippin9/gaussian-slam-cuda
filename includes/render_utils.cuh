// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "camera.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "rasterizer.cuh"
#include "sh_utils.cuh"
#include <cmath>
#include <torch/torch.h>

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(GaussianModel& gaussianModel,
                                                                                     torch::Tensor& bg_color,
                                                                                     float scaling_modifier = 1.0,
                                                                                     torch::Tensor override_color = torch::empty({})) {
    // Ensure background tensor (bg_color) is on GPU!
    bg_color = bg_color.to(torch::kCUDA);

    int w = 640;
    int h = 480;
    float fx = 517.3;
    float fy = 516.5;
    float cx = 318.6;
    float cy = 255.3;
    float near = 0.01;
    float far = 100;

    // Should get param from model for SLAM
    auto w2c = torch::eye(4);
    auto w2c_cuda = w2c.to(torch::kCUDA).to(torch::kFloat);
    auto cam_center = torch::inverse(w2c_cuda).slice(/*dim=*/0, /*start=*/3, /*end=*/4);
    auto viewmatrix = w2c_cuda.transpose(0, 1);
    auto opengl_proj = torch::tensor({
        {2 * fx / w, 0.0f, -(w - 2 * cx) / w, 0.0f},
        {0.0f, 2 * fy / h, -(h - 2 * cy) / h, 0.0f},
        {0.0f, 0.0f, far / (far - near), -(far * near) / (far - near)},
        {0.0f, 0.0f, 1.0f, 0.0f}
    }, torch::dtype(torch::kFloat32).device(torch::kCUDA)).transpose(0, 1);
    auto full_proj_matrix = viewmatrix.unsqueeze(0).bmm(opengl_proj.unsqueeze(0)).squeeze(0);

    // Set up rasterization configuration
    GaussianRasterizationSettings raster_settings = {
        .image_height = w,
        .image_width = h,
        .tanfovx = w / (2 * fx),
        .tanfovy = h / (2 * fy),
        .bg = bg_color,
        .scale_modifier = scaling_modifier,
        .viewmatrix = viewmatrix,
        .projmatrix = full_proj_matrix,
        .sh_degree = gaussianModel.Get_active_sh_degree(), //0 in SLAM?
        .camera_center = cam_center,
        .prefiltered = false};

    GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(true);
    means2D.retain_grad();
    auto opacity = gaussianModel.Get_opacity();

    auto scales = torch::Tensor();
    auto rotations = torch::Tensor();
    auto cov3D_precomp = torch::Tensor();

    scales = gaussianModel.Get_scaling();
    rotations = gaussianModel.Get_rotation();

    auto shs = torch::Tensor();
    torch::Tensor colors_precomp = torch::Tensor();
    // This is nonsense. Background color not used? See orginal file colors_precomp=None line 70
    shs = gaussianModel.Get_features();

    torch::cuda::synchronize();

    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    auto [rendererd_image, radii, rendererd_depth, rendererd_alpha] = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp);

    // Apply visibility filter to remove occluded Gaussians.
    // TODO: I think there is no real use for means2D, isn't it?
    // render, viewspace_points, visibility_filter, radii
    return {rendererd_image, means2D, radii > 0, radii, rendererd_depth, rendererd_alpha};
}
