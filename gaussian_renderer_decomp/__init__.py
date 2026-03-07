import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model_decompose import GaussianModel
from utils.encodings import STE_binary, STE_multistep

from gsplat2d.project_gaussians import project_gaussians
from gsplat2d.rasterize import rasterize_gaussians


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]  # [N_vis, 10, 1]

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    Q_feat = 1
    Q_scaling = 0.001 * 1000 * ((2**(pc.cur_scale)) / (2**(pc.multi_scale_level-1)))
    Q_offsets = 0.2 * ((2**(pc.cur_scale)) / (2**(pc.multi_scale_level-1)))

    if is_training:
        if pc.multi_scale_level == 4:
            if (step > 300 and step <= 1500) or (step > 1800 and step <= 4500) or (step > 5100 and step <= 9000) or (step > 9600 and step <= 12000):
                # quantization
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets
        elif pc.multi_scale_level == 3:
            if (step > 300 and step <= 3000) or (step>3600 and step<=9000) or (step>9600 and step <=12000):
                # quantization
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        elif pc.multi_scale_level == 2:
            if (step > 300 and step <= 9000) or (step > 9600 and step <= 12000):
                # quantization
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets
        elif pc.multi_scale_level == 1:
            if step > 1500 and step <= 12000:
                # quantization
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets
        else:
            pass

        if step == 1000 or step==2000 or step == 5000 or step == 12000:
            pc.update_anchor_bound()

        if step > 12000:
            # for rendering
            feat_context_orig = pc.calc_interp_feat(anchor)
            feat_context = pc.get_grid_mlp(feat_context_orig)
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim,pc.feat_dim, 4, 4, 2*pc.n_offsets, 2*pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            # for entropy
            choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.05
            anchor_chosen = pc.get_anchor[choose_idx]
            feat_chosen = pc._anchor_feat[choose_idx]
            grid_offsets_chosen = pc._offset[choose_idx]
            grid_scaling_chosen = pc.get_scaling[choose_idx]
            binary_grid_masks_chosen = pc.get_mask[choose_idx]  # [N_vis, 10, 1]
            mask_anchor_chosen = pc.get_mask_anchor[choose_idx]  # [N_vis, 1]

            feat_context_orig = pc.calc_interp_feat(anchor_chosen)
            feat_context = pc.get_grid_mlp(feat_context_orig)
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 4, 4, 2*pc.n_offsets, 2*pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = 1
            Q_scaling = 0.001 * 1000 * ((2**(pc.cur_scale)) / (2**(pc.multi_scale_level-1)))
            Q_offsets = 0.2 * ((2**(pc.cur_scale)) / (2**(pc.multi_scale_level-1)))

            if step % 1000 == 0:
                print(pc.cur_scale)
                print("Q_feat_adj", Q_feat, Q_scaling, Q_offsets)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 2)
            feat_chosen = feat_chosen + torch.empty_like(feat_chosen).uniform_(-0.5, 0.5) * Q_feat


            grid_scaling_chosen = grid_scaling_chosen + torch.empty_like(grid_scaling_chosen).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets_chosen = grid_offsets_chosen + torch.empty_like(grid_offsets_chosen).uniform_(-0.5, 0.5) * Q_offsets
            grid_offsets_chosen = grid_offsets_chosen.view(-1, 2 * pc.n_offsets)

            binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 2).view(-1, 2*pc.n_offsets)

            # check if mean has nan
            if torch.isnan(mean).any() or torch.isnan(scale).any() or torch.isnan(prob).any() or step == 15000 or step == 30000:
                print(feat_chosen.max(), feat_chosen.min(), feat_chosen.mean())
                print(grid_scaling_chosen.max(), grid_scaling_chosen.min(), grid_scaling_chosen.mean())
                print(grid_offsets_chosen.max(), grid_offsets_chosen.min(), grid_offsets_chosen.mean())
                print(mean.max(), mean.min(), mean.mean())
                print(scale.max(), scale.min(), scale.mean())

            bit_feat = pc.entropy_gaussian.forward(feat_chosen,mean,scale,Q=Q_feat, x_mean=pc._anchor_feat.mean())

            bit_feat = bit_feat * mask_anchor_chosen
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_scaling = bit_scaling * mask_anchor_chosen
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets.view(-1, 2*pc.n_offsets), pc._offset.mean())
            # bit_offsets = bit_offsets * mask_anchor_chosen * binary_grid_masks_chosen
            bit_offsets = bit_offsets * mask_anchor_chosen

            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

    elif not pc.decoded_version:
        print("not decoded version")
        torch.cuda.synchronize(); t1 = time.time()
        feat_context = pc.calc_interp_feat(anchor)
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 4, 4, 2*pc.n_offsets, 2*pc.n_offsets, 1, 1, 1], dim=-1)

        Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
        Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
        Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 2)  # [N_visible_anchor, 10, 2]
        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()
        torch.cuda.synchronize(); time_sub = time.time() - t1

    else:
        pass

    cat_local_view = feat # [N_visible_anchor, 32]

    # print(cat_local_view.shape)
    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]

    # select opacity
    opacity = neural_opacity  # [N_opacity_pos_gaussian, 1]

    mask = torch.ones_like(opacity, dtype=torch.bool, device=opacity.device).view(-1)  # [N_opacity_pos_gaussian]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    offsets = grid_offsets.view([-1, 2])  # [N_visible_anchor*K, 2]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 5+2]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 5+2]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (5+2)+3+3+2]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (5+2)+3+3+2]

    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([4, 2, 3, 3, 2], dim=-1)

    scaling = scaling_repeat[:, 2:] * torch.sigmoid(scale_rot[:, :2])
    rot = scale_rot[:, 2:]  # [N_opacity_pos_gaussian, 1]

    neural_opacity = scaling_repeat[:, 2:]

    if step % 1000 == 0:
        print(
            f"scaling:        prod {scaling.prod(dim=1).mean():.6f}, max {scaling.max():.6f}, mean {scaling.mean():.6f}")
        print(
            f"scaling_repeat: prod {scaling_repeat[:, 2:].prod(dim=1).mean():.6f}, max {scaling_repeat[:, 2:].max():.6f}, mean {scaling_repeat[:, 2:].mean():.6f}")
        print(
            f"offset_repeat:  prod {scaling_repeat[:, :2].prod(dim=1).mean():.6f}, max {scaling_repeat[:, :2].max():.6f}, mean {scaling_repeat[:, :2].mean():.6f}")
        print(
            f"scale_rot(sig): prod {torch.sigmoid(scale_rot[:, :2]).prod(dim=1).mean():.6f}, max {torch.sigmoid(scale_rot[:, :2]).max():.6f}, mean {torch.sigmoid(scale_rot[:, :2]).mean():.6f}")

    offsets = offsets * scaling_repeat[:, :2] # [N_opacity_pos_gaussian, 2]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 2]

    binary_grid_masks_pergaussian = binary_grid_masks.view(-1, 1)

    if is_training:
        pass
    else:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def compute_sigma(rot, scaling):
    """
    rot: [n, 1]  -> theta
    scaling: [n, 2] -> s1, s2
    return: [n, 3] -> [x11, x12, x22]
    """
    theta = rot.view(-1)         # [n]
    s1 = scaling[:, 0]           # [n]
    s2 = scaling[:, 1]           # [n]

    cos_t = torch.cos(theta)     # [n]
    sin_t = torch.sin(theta)     # [n]

    # Elements of RS
    rs_11 = cos_t * s1
    rs_12 = -sin_t * s2
    rs_21 = sin_t * s1
    rs_22 = cos_t * s2

    # Compute covariance matrix Σ = (RS)(RS)^T
    # Σ_11 = rs_11^2 + rs_12^2
    # Σ_12 = rs_11 * rs_21 + rs_12 * rs_22
    # Σ_22 = rs_21^2 + rs_22^2
    sigma_11 = rs_11**2 + rs_12**2
    sigma_12 = rs_11 * rs_21 + rs_12 * rs_22
    sigma_22 = rs_21**2 + rs_22**2

    sigma = torch.stack([sigma_11, sigma_12, sigma_22], dim=1)  # [n, 3]
    return sigma


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            xyz.retain_grad()
        except:
            pass

    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    cov2d = compute_sigma(rot, scaling)  # [N_opacity_pos_gaussian, 3]

    (
        xys,
        radii,
        conics,
        num_tiles_hit,
    ) = project_gaussians(
        cov2d,
        xyz,
        H,
        W,
        16, # self.B_SIZE
    )

    out_img = rasterize_gaussians(
        xys,
        radii,
        conics,
        num_tiles_hit,
        color,
        H,
        W,
        16,
    )[..., :3]
    rendered_image = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": xyz,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": xyz,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    xyz = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True

    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    (
        xys,
        radii_pure,
        conics,
        num_tiles_hit,
    ) = project_gaussians(
        scales[:, 1:],
        xyz,
        H,
        W,
        16,
    )

    # make it all positive
    radii_pure = torch.abs(radii_pure) + 1

    return radii_pure > 0
