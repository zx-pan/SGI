import copy
import csv
import json
import math
import os
import pathlib
import re
import shutil
import time
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import lpips
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer_decomp import prefilter_voxel, render, network_gui
from utils.encodings import get_binary_vxl_size
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from scene import GaussianModel, Scene
from utils.encodings_cuda import (
    set_chunk_size_cuda, get_chunk_size_cuda,
    encoder_gaussian_chunk, decoder_gaussian_chunk,
    encoder, decoder
)


lpips_fn = None
bit2MB_scale = 8 * 1024 * 1024
run_codec = True


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = [".git", ".gitignore"]
    ignorePatterns = set()
    ROOT = "."
    with open(os.path.join(ROOT, ".gitignore")) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith("#"):
                if line.endswith("\n"):
                    line = line[:-1]
                if line.endswith("/"):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print("Backup Finished!")


def list_image_paths(image_dir: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = []
    for name in os.listdir(image_dir):
        suffix = Path(name).suffix.lower()
        if suffix in exts:
            image_paths.append(os.path.join(image_dir, name))
    return sorted(image_paths)


def create_single_image_dataset(image_path: str, dataset_root: str, fov_deg: float):
    images_dir = os.path.join(dataset_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    dst_path = os.path.join(images_dir, image_name)
    if not os.path.exists(dst_path):
        try:
            os.symlink(image_path, dst_path)
        except OSError:
            shutil.copy2(image_path, dst_path)

    fov_rad = math.radians(fov_deg)
    transform = np.eye(4).tolist()
    frame = {"file_path": f"images/{image_name}", "transform_matrix": transform}
    transforms = {"camera_angle_x": fov_rad, "frames": [frame]}

    with open(os.path.join(dataset_root, "transforms_train.json"), "w") as f:
        json.dump(transforms, f, indent=2)
    with open(os.path.join(dataset_root, "transforms_test.json"), "w") as f:
        json.dump(transforms, f, indent=2)


def init_lpips_if_needed(disable_lpips: bool):
    global lpips_fn
    if disable_lpips:
        return None
    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
    return lpips_fn


def try_compute_lpips(image, gt_image, logger=None):
    if lpips_fn is None:
        return None
    try:
        return lpips_fn(image, gt_image).mean().double()
    except RuntimeError as exc:
        if logger is not None:
            logger.warning(f"Skipping LPIPS due to runtime error: {exc}")
        torch.cuda.empty_cache()
        return None


def training(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(
        args_param.multi_scale_level,
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        is_synthetic_nerf=os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")),
    )

    scales = args_param.multi_scale_level
    scene = Scene(
        dataset,
        gaussians,
        resolution_scales=[2**i for i in reversed(range(scales))]
    )
    gaussians.update_anchor_bound()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize()
    t_start = time.time()
    log_time_sub = 0

    for scale in reversed(range(scales)):
        if scales == 1:
            print("fitting without multi-scale")
        elif scales == 2:
            if scale == 1:
                opt.iterations = 9000
            elif scale == 0:
                opt.iterations = 15000
                first_iter = 9001
        elif scales == 3:
            if scale == 2:
                opt.iterations = 3000
            elif scale == 1:
                opt.iterations = 9000
                first_iter = 3001
            elif scale == 0:
                opt.iterations = 15000
                first_iter = 9001
        elif scales == 4:
            if scale == 3:
                opt.iterations = 1500
            elif scale == 2:
                opt.iterations = 4500
                first_iter = 1501
            elif scale == 1:
                opt.iterations = 9000
                first_iter = 4501
            elif scale == 0:
                opt.iterations = 15000
                first_iter = 9001
        else:
            first_iter = 1

        if scale != scales - 1:
            gaussians.update_attr_scale()
            print("update the scaling_repeat; update the anchor position")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(2**scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        for iteration in range(first_iter, opt.iterations + 1):
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            if (iteration - 1) == debug_from:
                pipe.debug = True

            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            bit_per_param = render_pkg["bit_per_param"]
            bit_per_feat_param = render_pkg["bit_per_feat_param"]
            bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
            bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

            if iteration % 1000 == 0 and bit_per_param is not None:
                ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
                ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
                ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
                ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

                logger.info("\n----------------------------------------------------------------------------------------")
                logger.info("\n-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
                logger.info("\n-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
                logger.info("\n-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
                logger.info("\n-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))
                with torch.no_grad():
                    binary_grid_masks_anchor = gaussians.get_mask_anchor.float()
                    mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks_anchor + 0.0)
                logger.info("\n-----[ITER {}] bits info: 1_rate_mask={}, mask_numel={}, mask_size_MB={}-----".format(iteration, mask_1_rate, mask_numel, mask_size_MB))

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            PSNR = psnr(image, gt_image).mean().double()

            scaling_reg = scaling.prod(dim=1).mean()
            opacity_reg = opacity.prod(dim=1).mean()
            loss = Ll1

            if bit_per_param is not None:
                _, bit_hash_grid, _, _ = get_binary_vxl_size((gaussians.get_encoding_params() + 1) / 2)
                denom = gaussians._anchor.shape[0] * (gaussians.feat_dim + 5 + 2 * gaussians.n_offsets)
                loss = loss + args_param.lmbda * (bit_per_param + bit_hash_grid / denom)

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                if iteration % 100 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{PSNR:.{7}f}"})
                    progress_bar.update(100)
                if scale == 0 and iteration == opt.iterations:
                    progress_bar.close()

                if scale == 0 and iteration == opt.iterations:
                    torch.cuda.synchronize()
                    t_start_log = time.time()
                    if (iteration in saving_iterations):
                        logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                    training_report(args_param, tb_writer, dataset_name, iteration, Ll1, loss, l1_loss,
                                    iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                                    (pipe, background), wandb, logger, args_param.model_path, scale)

                    torch.cuda.synchronize()
                    t_end_log = time.time()
                    t_log = t_end_log - t_start_log
                    log_time_sub += t_log

                if iteration < opt.update_until and iteration > opt.start_stat:
                    pass
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()

                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if (iteration in checkpoint_iterations):
                    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    torch.cuda.synchronize()
    t_end = time.time()
    total_training_time = t_end - t_start - log_time_sub
    logger.info("\n Total Training time: {}".format(t_end - t_start - log_time_sub))

    return scene, total_training_time


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(args, tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, wandb=None, logger=None, pre_path_name="", scale=1):
    render_path = os.path.join(args.model_path, dataset_name, "ours_{}".format(iteration), "renders-decoded-{}".format(scale))
    error_path = os.path.join(args.model_path, dataset_name, "ours_{}".format(iteration), "errors-decoded-{}".format(scale))
    gts_path = os.path.join(args.model_path, dataset_name, "ours_{}".format(iteration), "gt-decoded-{}".format(scale))

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(error_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    if tb_writer:
        tb_writer.add_scalar(f"{dataset_name}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{dataset_name}/train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar(f"{dataset_name}/iter_time", elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, "train_total_loss": loss, })

    if iteration in testing_iterations:
        scene.gaussians.eval()
        if iteration == testing_iterations[-1]:
            with torch.no_grad():
                log_info = scene.gaussians.estimate_final_bits()
                logger.info(log_info)
            if run_codec:
                with torch.no_grad():
                    bit_stream_path = os.path.join(pre_path_name, "bitstreams")
                    os.makedirs(bit_stream_path, exist_ok=True)
                    log_info = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
                    logger.info(log_info)
                    print("Encoding finished!")
                    log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path)
                    logger.info(log_info)
                    print("Decoding finished!")
        torch.cuda.empty_cache()
        validation_configs = ({"name": "test", "cameras": scene.getTestCameras(2**scale)},)

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                lpips_count = 0
                t_list = []

                for idx, viewpoint in enumerate(config["cameras"]):
                    torch.cuda.synchronize()
                    t_start = time.time()
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    torch.cuda.synchronize()
                    t_end = time.time()
                    t_list.append(t_end - t_start)

                    gt = viewpoint.original_image[0:3, :, :]
                    gt_image = torch.clamp(gt, 0.0, 1.0)
                    if tb_writer and (idx < 1):
                        tb_writer.add_images(f"{dataset_name}/" + config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f"{dataset_name}/" + config["name"] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None] - image[None]).abs(), global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{dataset_name}/" + config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_value = try_compute_lpips(image, gt_image, logger=logger)
                    if lpips_value is not None:
                        lpips_test += lpips_value
                        lpips_count += 1

                    errormap = (image - gt).abs()

                    torchvision.utils.save_image(image, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
                    torchvision.utils.save_image(errormap, os.path.join(error_path, "{0:05d}".format(idx) + ".png"))
                    torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))

                psnr_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                if lpips_count > 0:
                    lpips_test /= lpips_count
                    lpips_log_str = f"{lpips_test}"
                else:
                    lpips_test = None
                    lpips_log_str = "N/A (disabled or failed)"
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config["name"], l1_test, psnr_test, ssim_test, lpips_log_str))
                test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
                logger.info(f"Test FPS: {test_fps.item():.5f}")
                if tb_writer:
                    tb_writer.add_scalar(f"{dataset_name}/test_FPS", test_fps.item(), 0)
                if wandb is not None:
                    wandb.log({"test_fps": test_fps, })

                if tb_writer:
                    tb_writer.add_scalar(f"{dataset_name}/" + config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{dataset_name}/" + config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(f"{dataset_name}/" + config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
                    if lpips_test is not None:
                        tb_writer.add_scalar(f"{dataset_name}/" + config["name"] + "/loss_viewpoint - lpips", lpips_test, iteration)
                if wandb is not None:
                    log_payload = {
                        f"{config['name']}_loss_viewpoint_l1_loss": l1_test,
                        f"{config['name']}_PSNR": psnr_test,
                        f"{config['name']}_ssim": ssim_test,
                    }
                    if lpips_test is not None:
                        log_payload[f"{config['name']}_lpips"] = lpips_test
                    wandb.log(log_payload)

        if tb_writer:
            tb_writer.add_scalar(f"{dataset_name}/" + "total_points", scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()


def parse_encoding_total_mb(log_info: str):
    if not log_info:
        return None
    match = re.search(r"Encoded sizes in MB:.*?Total\s+([0-9.]+)", log_info, re.DOTALL)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_decoding_time_s(log_info: str):
    if not log_info:
        return None
    match = re.search(r"DecTime\s*([0-9.]+)", log_info)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_encoding_time_s(log_info: str):
    if not log_info:
        return None
    match = re.search(r"EncTime\s*([0-9.]+)", log_info)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def render_single_image_metrics(scene: Scene, pipe, white_background: bool, model_path: str):
    scene.gaussians.eval()
    bitstream_path = os.path.join(model_path, "bitstreams")
    os.makedirs(bitstream_path, exist_ok=True)
    enc_log = scene.gaussians.conduct_encoding(pre_path_name=bitstream_path)
    dec_log = scene.gaussians.conduct_decoding(pre_path_name=bitstream_path)
    enc_total_mb = parse_encoding_total_mb(enc_log)
    dec_time_s = parse_decoding_time_s(dec_log)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    view = scene.getTestCameras(1.0)[0]
    voxel_visible_mask = prefilter_voxel(view, scene.gaussians, pipe, background)
    time_start = time.time()
    render_output = render(view, scene.gaussians, pipe, background, visible_mask=voxel_visible_mask)
    image = torch.clamp(render_output["render"], 0.0, 1.0)
    time_end = time.time()
    render_time_s = time_end - time_start
    gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
    psnr_val = psnr(image, gt).mean().double().item()
    ssim_val = ssim(image, gt).mean().double().item()
    lpips_tensor = try_compute_lpips(image, gt)
    lpips_val = lpips_tensor.item() if lpips_tensor is not None else None
    return psnr_val, ssim_val, enc_total_mb, dec_time_s, enc_log, dec_log, render_time_s, lpips_val


def test_bitstream_times(args_param, dataset, pipe, bitstream_path: str, mlp_checkpoint_path: str = None):
    gaussians = GaussianModel(
        args_param.multi_scale_level,
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        is_synthetic_nerf=os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")),
    )

    scales = args_param.multi_scale_level
    scene = Scene(
        dataset,
        gaussians,
        resolution_scales=[2**i for i in reversed(range(scales))]
    )

    gaussians.update_anchor_bound()

    set_chunk_size_cuda(500)
    
    # Match the scale level of the final trained bitstream
    gaussians.cur_scale = args_param.multi_scale_level - 1
    # Ensure SH degree is correctly set
    gaussians.active_sh_degree = 3.
    
    scene.gaussians.eval()

    if not os.path.isdir(bitstream_path):
        raise ValueError(f"bitstream_path not found: {bitstream_path}")

    # Load MLP checkpoint (required for correct rendering)
    if mlp_checkpoint_path is None:
        # Try to find checkpoint relative to bitstream_path
        model_dir = os.path.dirname(bitstream_path)
        mlp_checkpoint_path = os.path.join(model_dir, "point_cloud", "iteration_15000", "checkpoint.pth")
    if os.path.exists(mlp_checkpoint_path):
        print(f"Loading MLP checkpoint from: {mlp_checkpoint_path}")
        gaussians.load_mlp_checkpoints(mlp_checkpoint_path)
    else:
        print(f"Warning: MLP checkpoint not found at {mlp_checkpoint_path}, PSNR will be low!")

    # Test both sequential and parallel decoding
    print("\n=== Testing Sequential Decoding ===")
    dec_log_seq = gaussians.conduct_decoding(pre_path_name=bitstream_path, use_parallel=False)
    dec_time_seq = parse_decoding_time_s(dec_log_seq)
    print(f"Sequential decode time: {dec_time_seq}s")
    
    print("\n=== Testing Parallel Decoding ===")
    dec_log = gaussians.conduct_decoding(pre_path_name=bitstream_path, use_parallel=True)
    dec_time_s = parse_decoding_time_s(dec_log)
    print(f"Parallel decode time: {dec_time_s}s")
    
    if dec_time_seq and dec_time_s:
        speedup = dec_time_seq / dec_time_s if dec_time_s > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    view = scene.getTestCameras(1.0)[0]
    voxel_visible_mask = prefilter_voxel(view, scene.gaussians, pipe, background)
    
    # Warmup render (first call compiles CUDA kernels)
    _ = render(view, scene.gaussians, pipe, background, visible_mask=voxel_visible_mask)
    torch.cuda.synchronize()
    
    # Actual timed render
    time_start = time.time()
    render_output = render(view, scene.gaussians, pipe, background, visible_mask=voxel_visible_mask)
    torch.cuda.synchronize()  # Ensure GPU work is done
    image = torch.clamp(render_output["render"], 0.0, 1.0)
    time_end = time.time()
    render_time_s = time_end - time_start
    print(f'render_time_s: {render_time_s}')
    gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
    psnr_val = psnr(image, gt).mean().double().item()
    print(f'psnr_val: {psnr_val}')

    return dec_time_s, dec_log


def test_chunk_size_variations(
    args_param, 
    dataset, 
    pipe, 
    original_bitstream_path: str,
    test_chunk_sizes: list = None,
    output_base_dir: str = None,
    mlp_checkpoint_path: str = None
):
    """
    Test different chunk sizes for arithmetic encoding/decoding.
    
    This function:
    1. Decodes the original bitstream with its original chunk size
    2. Re-encodes with different chunk sizes to new directories
    3. Decodes with new chunk sizes and measures performance
    4. Compares PSNR and decoding time
    
    Args:
        args_param: Model arguments
        dataset: Dataset configuration
        pipe: Pipeline configuration
        original_bitstream_path: Path to original encoded bitstream
        test_chunk_sizes: List of chunk sizes to test (default: [50000, 100000, 200000, 500000])
        output_base_dir: Base directory for re-encoded bitstreams (default: same as original with _chunk_test suffix)
        mlp_checkpoint_path: Path to MLP checkpoint
    
    Returns:
        dict: Results with chunk_size -> {dec_time_s, psnr, bitstream_path}
    """
    if test_chunk_sizes is None:
        test_chunk_sizes = [50000, 100000, 200000, 500000, 1000000]
    
    if output_base_dir is None:
        output_base_dir = original_bitstream_path.rstrip('/') + '_chunk_test'
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Store original chunk size
    original_chunk_size = get_chunk_size_cuda()
    print(f"Original chunk_size_cuda: {original_chunk_size}")
    
    # Initialize Gaussian model
    gaussians = GaussianModel(
        args_param.multi_scale_level,
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        is_synthetic_nerf=os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")),
    )
    
    scales = args_param.multi_scale_level
    scene = Scene(
        dataset,
        gaussians,
        resolution_scales=[2**i for i in reversed(range(scales))]
    )
    gaussians.update_anchor_bound()
    gaussians.cur_scale = args_param.multi_scale_level - 1
    gaussians.active_sh_degree = 3.
    gaussians.eval()
    
    # Load MLP checkpoint
    if mlp_checkpoint_path is None:
        model_dir = os.path.dirname(original_bitstream_path)
        mlp_checkpoint_path = os.path.join(model_dir, "point_cloud", "iteration_15000", "checkpoint.pth")
    if os.path.exists(mlp_checkpoint_path):
        print(f"Loading MLP checkpoint from: {mlp_checkpoint_path}")
        gaussians.load_mlp_checkpoints(mlp_checkpoint_path)
    else:
        print(f"Warning: MLP checkpoint not found at {mlp_checkpoint_path}")
    
    # Setup background and view for rendering
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    view = scene.getTestCameras(1.0)[0]
    gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
    
    results = {}
    
    # Step 1: Decode original bitstream (this loads data into gaussians)
    print(f"\n{'='*60}")
    print(f"Step 1: Decoding original bitstream with chunk_size={original_chunk_size}")
    print(f"{'='*60}")
    
    set_chunk_size_cuda(original_chunk_size)
    dec_log_original = gaussians.conduct_decoding(pre_path_name=original_bitstream_path)
    dec_time_original = parse_decoding_time_s(dec_log_original)
    
    # Render and get PSNR for original
    voxel_visible_mask = prefilter_voxel(view, scene.gaussians, pipe, background)
    render_output = render(view, scene.gaussians, pipe, background, visible_mask=voxel_visible_mask)
    image_original = torch.clamp(render_output["render"], 0.0, 1.0)
    psnr_original = psnr(image_original, gt).mean().double().item()
    
    results['original'] = {
        'chunk_size': original_chunk_size,
        'dec_time_s': dec_time_original,
        'psnr': psnr_original,
        'bitstream_path': original_bitstream_path
    }
    print(f"Original: chunk_size={original_chunk_size}, dec_time={dec_time_original:.4f}s, PSNR={psnr_original:.2f}dB")
    
    # Save the decoded state for re-encoding
    # We'll store the decoded parameters before they get modified
    decoded_anchor = gaussians._anchor.data.clone()
    decoded_anchor_feat = gaussians._anchor_feat.data.clone()
    decoded_offset = gaussians._offset.data.clone()
    decoded_scaling = gaussians._scaling.data.clone()
    decoded_mask = gaussians._mask.data.clone()
    
    # Step 2: Re-encode with different chunk sizes and test decoding
    for chunk_size in test_chunk_sizes:
        print(f"\n{'='*60}")
        print(f"Testing chunk_size={chunk_size}")
        print(f"{'='*60}")
        
        # Create output directory for this chunk size
        chunk_output_dir = os.path.join(output_base_dir, f"chunk_{chunk_size}")
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        # Set the new chunk size
        set_chunk_size_cuda(chunk_size)
        print(f"Set chunk_size_cuda to: {get_chunk_size_cuda()}")
        
        # Restore the decoded parameters
        gaussians._anchor = torch.nn.Parameter(decoded_anchor.clone())
        gaussians._anchor_feat = torch.nn.Parameter(decoded_anchor_feat.clone())
        gaussians._offset = torch.nn.Parameter(decoded_offset.clone())
        gaussians._scaling = torch.nn.Parameter(decoded_scaling.clone())
        gaussians._mask = torch.nn.Parameter(decoded_mask.clone())
        
        # Re-encode with new chunk size
        print(f"Re-encoding to: {chunk_output_dir}")
        try:
            enc_log = gaussians.conduct_encoding(pre_path_name=chunk_output_dir)
            enc_time = parse_encoding_time_s(enc_log)
            print(f"Encoding time: {enc_time}s")
        except Exception as e:
            print(f"Encoding failed with chunk_size={chunk_size}: {e}")
            results[chunk_size] = {'error': str(e)}
            continue
        
        # Decode with new chunk size
        print(f"Decoding with chunk_size={chunk_size}")
        try:
            time_start = time.time()
            dec_log = gaussians.conduct_decoding(pre_path_name=chunk_output_dir)
            dec_time = parse_decoding_time_s(dec_log)
            time_end = time.time()
            wall_dec_time = time_end - time_start
        except Exception as e:
            print(f"Decoding failed with chunk_size={chunk_size}: {e}")
            results[chunk_size] = {'error': str(e)}
            continue
        
        # Render and get PSNR
        voxel_visible_mask = prefilter_voxel(view, scene.gaussians, pipe, background)
        render_output = render(view, scene.gaussians, pipe, background, visible_mask=voxel_visible_mask)
        image = torch.clamp(render_output["render"], 0.0, 1.0)
        psnr_val = psnr(image, gt).mean().double().item()
        
        # Calculate bitstream size
        bitstream_size_bytes = sum(
            os.path.getsize(os.path.join(chunk_output_dir, f))
            for f in os.listdir(chunk_output_dir)
            if os.path.isfile(os.path.join(chunk_output_dir, f))
        )
        bitstream_size_mb = bitstream_size_bytes / (1024 * 1024)
        
        results[chunk_size] = {
            'chunk_size': chunk_size,
            'dec_time_s': dec_time,
            'wall_dec_time_s': wall_dec_time,
            'enc_time_s': enc_time,
            'psnr': psnr_val,
            'bitstream_path': chunk_output_dir,
            'bitstream_size_mb': bitstream_size_mb,
            'psnr_diff': psnr_val - psnr_original
        }
        
        print(f"chunk_size={chunk_size}: dec_time={dec_time:.4f}s, wall_time={wall_dec_time:.4f}s, "
              f"PSNR={psnr_val:.2f}dB (diff={psnr_val - psnr_original:+.4f}dB), "
              f"size={bitstream_size_mb:.3f}MB")
        
        torch.cuda.empty_cache()
    
    # Restore original chunk size
    set_chunk_size_cuda(original_chunk_size)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Chunk Size':<15} {'Dec Time (s)':<15} {'PSNR (dB)':<12} {'PSNR Diff':<12} {'Size (MB)':<12}")
    print("-" * 66)
    
    for key, val in results.items():
        if 'error' in val:
            print(f"{key:<15} ERROR: {val['error']}")
        else:
            psnr_diff = val.get('psnr_diff', 0)
            size_mb = val.get('bitstream_size_mb', 'N/A')
            print(f"{val['chunk_size']:<15} {val['dec_time_s']:<15.4f} {val['psnr']:<12.2f} {psnr_diff:+<12.4f} {size_mb if isinstance(size_mb, str) else f'{size_mb:.3f}':<12}")
    
    # Save results to JSON
    results_json_path = os.path.join(output_base_dir, "chunk_size_test_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_json_path}")
    
    return results


def write_metrics(output_root: str, metrics_rows):
    os.makedirs(output_root, exist_ok=True)
    json_path = os.path.join(output_root, "metrics.json")
    csv_path = os.path.join(output_root, "metrics.csv")

    with open(json_path, "w") as f:
        json.dump({"images": metrics_rows}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "psnr", "ssim", "lpips", "encoding_total_mb", "decoding_time_s", "rendering_time_s", "model_path", "total_training_time (minutes)"],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)


def average_metric(metrics, key):
    values = [metric[key] for metric in metrics if metric.get(key) is not None]
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-scale image folder training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--image_dir", type=str, default=None, help="Folder containing images to process")
    parser.add_argument("--output_root", type=str, default=None, help="Base output folder for all images")
    parser.add_argument("--fov_deg", type=float, default=60.0, help="FOV in degrees for single-image cameras")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--lmbda", type=float, default=0.001)
    parser.add_argument("--multi_scale_level", type=int, default=3)
    parser.add_argument("--anchor_number", type=int, default=5000)
    parser.add_argument("--n_offsets_num", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=50)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--test_chunk_sizes", action="store_true", default=False,
                        help="Test different chunk sizes for encoding/decoding")
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=[50000, 100000, 200000, 500000, 1000000],
                        help="List of chunk sizes to test (default: 50000 100000 200000 500000 1000000)")
    parser.add_argument("--bitstream_path", type=str, default=None,
                        help="Path to existing bitstream for chunk size testing")
    parser.add_argument("--chunk_test_output", type=str, default=None,
                        help="Output directory for chunk size test results")
    parser.add_argument("--disable_lpips", action="store_true", default=False,
                        help="Disable LPIPS metric computation (recommended for very large images)")
    parser.add_argument("--wandb_project", type=str, default="Scaffold-GS-Image",
                        help="Weights & Biases project name")

    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")

    image_dir = args.image_dir or args.source_path
    if image_dir is None:
        raise ValueError("image_dir or source_path must be provided")
    image_dir = os.path.abspath(image_dir)
    image_paths = list_image_paths(image_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    output_root = args.output_root or args.model_path
    if not output_root:
        output_root = os.path.join("./outputs", "image_runs")
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            name=Path(output_root).name,
            settings=wandb.Settings(start_method="fork"),
            config=vars(args),
        )
    else:
        wandb = None

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    init_lpips_if_needed(args.disable_lpips)

    metrics_rows = []

    # image_paths = image_paths[:2] # for testing

    for image_path in image_paths:
        image_name = Path(image_path).name
        image_stem = Path(image_path).stem
        per_image_args = copy.deepcopy(args)
        per_image_args.model_path = os.path.join(output_root, image_stem)
        dataset_root = os.path.join(output_root, "_datasets", image_stem)
        os.makedirs(dataset_root, exist_ok=True)
        create_single_image_dataset(image_path, dataset_root, per_image_args.fov_deg)
        per_image_args.source_path = dataset_root
        os.makedirs(per_image_args.model_path, exist_ok=True)

        logger = get_logger(per_image_args.model_path)
        logger.info(f"args: {per_image_args}")
        logger.info(f"processing image: {image_name}")

        dataset = lp.extract(per_image_args)
        pipe = pp.extract(per_image_args)
        opt = op.extract(per_image_args)

        per_image_args.port = np.random.randint(10000, 20000)
        
        # test if the bitstream can successfully decode
        if args.test_only:
            bitstream_path = args.bitstream_path
            if bitstream_path is None:
                raise ValueError("--bitstream_path is required when --test_only is enabled")
            dec_time_s, dec_log = test_bitstream_times(args, dataset, pipe, bitstream_path)
            print(dec_time_s)
            exit()

        # test the best chunk size to balance the decoding time and the size of the bitstream
        if args.test_chunk_sizes:
            bitstream_path = args.bitstream_path
            if bitstream_path is None:
                raise ValueError("--bitstream_path is required when --test_chunk_sizes is enabled")
            output_dir = args.chunk_test_output or (bitstream_path.rstrip('/') + '_chunk_test')
            print(f"\n{'#'*70}")
            print(f"Testing chunk sizes: {args.chunk_sizes}")
            print(f"Original bitstream: {bitstream_path}")
            print(f"Output directory: {output_dir}")
            print(f"{'#'*70}\n")
            results = test_chunk_size_variations(
                args, dataset, pipe, 
                original_bitstream_path=bitstream_path,
                test_chunk_sizes=args.chunk_sizes,
                output_base_dir=output_dir,
            )
            print("\nChunk size testing complete!")
            exit()

        scene, total_training_time = training(
            per_image_args,
            dataset,
            opt,
            pipe,
            image_stem,
            per_image_args.test_iterations,
            per_image_args.save_iterations,
            per_image_args.checkpoint_iterations,
            per_image_args.start_checkpoint,
            per_image_args.debug_from,
            wandb=wandb,
            logger=logger,
        )

        psnr_val, ssim_val, enc_total_mb, dec_time_s, enc_log, dec_log, render_time_s, lpips_val = render_single_image_metrics(
            scene, pipe, dataset.white_background, per_image_args.model_path
        )
        logger.info(enc_log)
        logger.info(dec_log)
        metrics_rows.append({
            "image": image_name,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "encoding_total_mb": enc_total_mb,
            "decoding_time_s": dec_time_s,
            "rendering_time_s": render_time_s,
            "model_path": per_image_args.model_path,
            "total_training_time (minutes)": total_training_time / 60.0,
        })
        write_metrics(output_root, metrics_rows)

        torch.cuda.empty_cache()

    if wandb is not None:
        wandb.finish()

    # the average statistics of the metrics.json
    with open(os.path.join(output_root, "metrics.json"), "r") as f:
        metrics_data = json.load(f)
    metrics = metrics_data.get("images", [])
    if len(metrics) == 0:
        raise ValueError("metrics.json has no images to aggregate")
    average_psnr = average_metric(metrics, "psnr")
    average_ssim = average_metric(metrics, "ssim")
    average_lpips = average_metric(metrics, "lpips")
    average_encoding_total_mb = average_metric(metrics, "encoding_total_mb")
    average_decoding_time_s = average_metric(metrics, "decoding_time_s")
    average_rendering_time_s = average_metric(metrics, "rendering_time_s")
    average_total_training_time = average_metric(metrics, "total_training_time (minutes)")
    print(f"Average PSNR: {average_psnr}")
    print(f"Average SSIM: {average_ssim}")
    if average_lpips is None:
        print("Average LPIPS: N/A (disabled or failed on all images)")
    else:
        print(f"Average LPIPS: {average_lpips}")
    print(f"Average Encoding Total MB: {average_encoding_total_mb}")
    print(f"Average Decoding Time (seconds): {average_decoding_time_s}")
    print(f"Average Rendering Time (seconds): {average_rendering_time_s}")
    print(f"Average Total Training Time (minutes): {average_total_training_time}")
