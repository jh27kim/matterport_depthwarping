import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from einops import rearrange
import torch.nn.functional as F
from typing import Tuple, Optional
from typing import Sequence, Union
import open3d as o3d

def pil_to_torch(pil_img):
    _np_img = np.array(pil_img).astype(np.float32) / 255.0
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img

def torch_to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    tensor = tensor.permute(1, 2, 0)
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = (np_tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(np_tensor)
    return pil_tensor

def cam2world(depth, pose, K1, device=0):
    """
    depth: torch.tensor() B H W 
        Source depth map
        
    pose: torch.tensor() B 4 4
        Source extrinsic camera parameter. 
        
    K1: torch.tensor() B 3 3
        Source intrinsic camera parameter. 
        
    device: str/int 
    """
    
    assert depth.dim() == 3
    batch_size, height, width = depth.shape
    assert batch_size == 1

    y_2d, x_2d = torch.meshgrid(torch.arange(height, dtype=depth.dtype), 
                                torch.arange(width, dtype=depth.dtype))
    
    x_2d = x_2d.to(device)
    y_2d = y_2d.to(device)
    z_2d = torch.ones_like(x_2d)
    depth = depth.to(device)
    pose = pose.to(device)
    permutation_matrix = PERMUTATION_MATRIX[None, ...].repeat([batch_size, 1, 1]).to(device)

    homo_x2d = torch.stack((x_2d, y_2d, z_2d), dim=-1).repeat([batch_size, 1, 1, 1]) # B H W 3
    # print("homo_x2d", homo_x2d.shape, homo_x2d[0][3][5])

    inv_K1 = torch.linalg.inv(K1).to(device) # B 3 3
    homo_x2d = rearrange(homo_x2d, "b h w c -> b c (h w)")
    uncalib_x3d = inv_K1 @ homo_x2d
    uncalib_x3d = rearrange(uncalib_x3d,  "b c (h w) -> b h w c", w=width, h=height) # B H W 3
    # print("uncalib_x3d", uncalib_x3d.shape, uncalib_x3d[0][3][5])
    uncalib_x3d = rearrange(uncalib_x3d, "b h w c -> b c (h w)")
    
    unproj_3d = permutation_matrix @ uncalib_x3d
    unproj_3d = rearrange(unproj_3d, "b c (h w) -> b h w c", w=width, h=height) # B H W 3
    # print("unproj_3d", unproj_3d.shape, unproj_3d[0][3][5])

    depth_unproj_3d = unproj_3d * (-depth[..., None])
    # print("depth_proj_3d", depth_unproj_3d.shape, depth_unproj_3d[0][3][5]) # B H W 3

    world_3d_homo_ones = torch.ones(batch_size, height, width, 1).to(device)
    world_3d_homo = torch.cat([depth_unproj_3d, world_3d_homo_ones], dim=3) # B H W 4
    # print("world_3d_homo", world_3d_homo.shape, world_3d_homo[0][3][5])
    world_3d_homo = rearrange(world_3d_homo, "b h w c -> b c (h w)")

    world_3d_pts = pose @ world_3d_homo
    world_3d_pts = rearrange(world_3d_pts, "b c (h w) -> b h w c", w=width, h=height)
    world_3d_pts = world_3d_pts[..., :3] # B H W 3
    # print("world_3d_pts", world_3d_pts.shape, world_3d_pts[0][3][5], "\n")

    return world_3d_pts
    
def world2cam(x_3d, pose, K1, device=0, depth=None, return_warp_depth=False):
    """
    x_3d: torch.tensor() B H W 3 
        3D world points. 
        
    pose: torch.tensor() B 4 4
        Target extrinsic camera parameter. 
        
    K1: torch.tensor() B 3 3
        Target intrinsic camera parameter. 
        
    device: str/int 
    """
    batch_size, height, width, _ = x_3d.shape
    pose_inv = torch.linalg.inv(pose).to(device)
    K1 = K1.to(device)
    permutation_matrix = torch.linalg.inv(PERMUTATION_MATRIX)[None, ...].repeat([batch_size, 1, 1]).to(device)
    # print("permutation_matrix", permutation_matrix)
    x_3d_homo_ones = torch.ones(batch_size, height, width, 1).to(device)
    x_3d_homo = torch.cat([x_3d, x_3d_homo_ones], axis=3) # B H W 4
    # print("x_3d_homo", x_3d_homo.shape, x_3d_homo[0][3][5]) 
    x_3d_homo = rearrange(x_3d_homo, "b h w c -> b c (h w)") 

    x_3d_trans = pose_inv @ x_3d_homo
    x_3d_trans = rearrange(x_3d_trans, "b c (h w) -> b h w c", w=width, h=height) # B H W 4
    # print("x_3d_trans", x_3d_trans.shape, x_3d_trans[0][3][5])

    # REMOVE ABSOLUTE
    ########################################################################
    x_2d_depth_norm = x_3d_trans[..., :3] / (x_3d_trans[..., 2:3] + 1e-9) # B H W 3
    # print("x_2d_depth_norm", x_2d_depth_norm.shape, x_2d_depth_norm[0][3][5])
    x_2d_proj = rearrange(x_2d_depth_norm, "b h w c -> b c (h w)")
    x_2d_proj = permutation_matrix @ x_2d_proj
    x_2d_proj = rearrange(x_2d_proj, "b c (h w) -> b h w c", w=width, h=height)
    # print("x_2d_proj", x_2d_proj.shape, x_2d_proj[0][3][5])
    x_2d_proj = rearrange(x_2d_proj, "b h w c -> b c (h w)")
    
    x_2d_calib = K1 @ x_2d_proj
    x_2d_calib = rearrange(x_2d_calib, "b c (h w) -> b h w c", w=width, h=height) # B H W 3
    # print("x_2d_calib", x_2d_calib.shape, x_2d_calib[0][3][5])
    x_2d = x_2d_calib[..., :2]
    # print("x_2d", x_2d.shape, x_2d[0][3][5], "\n") # B H W 2

    if depth != None:
        mask = depth == 0
        x_2d[mask] = -1e9

    if return_warp_depth:
        return x_2d, (-x_3d_trans[..., 2])
        
    return x_2d

def bilinear_splatting(frame1: torch.Tensor, 
                       mask1: Optional[torch.Tensor], 
                       depth1: torch.Tensor,
                       trans_pos: torch.Tensor, 
                       flow12_mask: Optional[torch.Tensor], 
                       is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        
        b, c, h, w = frame1.shape
        
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
            
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(frame1)
        
        # Compute relative weight to the closest grid 
        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)
        
        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)
        
        # Attenuate weight by depth value
        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)
        
        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)

        warped_frame.index_put_((batch_indices,
                                 trans_pos_floor[:, 1], 
                                 trans_pos_floor[:, 0]),
                                 frame1_cl * weight_nw, 
                                 accumulate=True)
        
        warped_frame.index_put_((batch_indices, 
                                 trans_pos_ceil[:, 1], 
                                 trans_pos_floor[:, 0]),
                                 frame1_cl * weight_sw, 
                                 accumulate=True)
        
        warped_frame.index_put_((batch_indices, 
                                 trans_pos_floor[:, 1], 
                                 trans_pos_ceil[:, 0]),
                                 frame1_cl * weight_ne, 
                                 accumulate=True)
        
        warped_frame.index_put_((batch_indices, 
                                 trans_pos_ceil[:, 1], 
                                 trans_pos_ceil[:, 0]),
                                 frame1_cl * weight_se, 
                                 accumulate=True)

        warped_weights.index_put_((batch_indices, 
                                   trans_pos_floor[:, 1], 
                                   trans_pos_floor[:, 0]),
                                   weight_nw, 
                                   accumulate=True)
        
        warped_weights.index_put_((batch_indices, 
                                   trans_pos_ceil[:, 1], 
                                   trans_pos_floor[:, 0]),
                                   weight_sw, 
                                   accumulate=True)
        
        warped_weights.index_put_((batch_indices, 
                                   trans_pos_floor[:, 1], 
                                   trans_pos_ceil[:, 0]),
                                   weight_ne, 
                                   accumulate=True)
        
        warped_weights.index_put_((batch_indices, 
                                   trans_pos_ceil[:, 1], 
                                   trans_pos_ceil[:, 0]),
                                   weight_se,
                                   accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -0.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=0, max=1)
            
        return warped_frame2, mask2


def upsample(img):
    assert torch.is_tensor(img)
    squeeze = False
    if img.dim() == 3:
        squeeze = True
        img = img.unsqueeze(0)

    upsampled_img = F.interpolate(img, (INTERPOLATE_HEIGHT, INTERPOLATE_WIDTH), mode='nearest')
    if squeeze:
        upsampled_img = upsampled_img.squeeze(0)
        
    return upsampled_img


INTERPOLATE_HEIGHT, INTERPOLATE_WIDTH = 512, 512
# Original Matterport3D 
# PERMUTATION_MATRIX = torch.tensor([
#     [1, 0, 0],
#     [0, -1, 0],
#     [0, 0, 1],
# ], dtype=torch.float32) # 3 3

# SparsePlane Matterport3D 
PERMUTATION_MATRIX = torch.tensor([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
], dtype=torch.float32) # 3 3


def main():
    json_path = "./input/metadata.json"
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)

    print(f"Total data: {len(json_dict['data'])}")

    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    intrinsic = torch.tensor([[focal_length, 0, offset_x], 
                            [0, focal_length, offset_y],
                            [0, 0, 1]], dtype=torch.float32).unsqueeze(0)


    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')

    save_img = True
    save_pc = True
    use_abs_depth = True

    for data_index in range(len(json_dict["data"])):
        print(f"{data_index} / {len(json_dict['data'])}")
        source_path = json_dict["data"][data_index]["source"]
        target_path = json_dict["data"][data_index]["target"]
        
        if use_abs_depth:
            print("USING GT DEPTH")
            source_rel_depth_path = json_dict["data"][data_index]["source_rel_depth"]
            source_abs_depth_path = json_dict["data"][data_index]["source_abs_depth"]
            target_rel_depth_path = json_dict["data"][data_index]["target_rel_depth"]
            target_abs_depth_path = json_dict["data"][data_index]["target_abs_depth"]
        else:
            print("USING PRED DEPTH")
            source_rel_depth_path = json_dict["data"][data_index]["pred_source_rel_depth"]
            source_abs_depth_path = json_dict["data"][data_index]["pred_source_abs_depth"]
            target_rel_depth_path = json_dict["data"][data_index]["pred_target_rel_depth"]
            target_abs_depth_path = json_dict["data"][data_index]["pred_target_abs_depth"]
        
        pil_source = Image.open(source_path).convert("RGB")
        pil_source_rel_depth = Image.open(source_rel_depth_path)
        np_source_rel_depth = np.array(pil_source_rel_depth)
        np_source_abs_depth = np.load(source_abs_depth_path)
        
        pil_target = Image.open(target_path).convert("RGB")
        pil_target_rel_depth = Image.open(target_rel_depth_path)
        np_target_rel_depth = np.array(pil_target_rel_depth)
        np_target_abs_depth = np.load(target_abs_depth_path)

        torch_target_abs_depth = torch.from_numpy(np_target_abs_depth)
        torch_source_abs_depth = torch.from_numpy(np_source_abs_depth)
        if torch_source_abs_depth.dim() == 2:
            torch_source_abs_depth = torch_source_abs_depth.unsqueeze(-1)
        if torch_target_abs_depth.dim() == 2:
            torch_target_abs_depth = torch_target_abs_depth.unsqueeze(-1)

        torch_source_rel_depth = torch.from_numpy(np_source_rel_depth)
        torch_target_rel_depth = torch.from_numpy(np_target_rel_depth)
        if torch_source_rel_depth.dim() == 2:
            torch_source_rel_depth = torch_source_rel_depth.unsqueeze(-1)
            _c = torch_source_rel_depth.shape[-1]
            if _c == 1:
                torch_source_rel_depth = torch.cat([torch_source_rel_depth]*3, axis=-1)
        if torch_target_rel_depth.dim() == 2:
            torch_target_rel_depth = torch_target_rel_depth.unsqueeze(-1)
            _c = torch_target_rel_depth.shape[-1]
            if _c == 1:
                torch_target_rel_depth = torch.cat([torch_target_rel_depth]*3, axis=-1)

        torch_target_abs_depth = torch_target_abs_depth.permute(2, 0, 1)
        torch_source_abs_depth = torch_source_abs_depth.permute(2, 0, 1)
        torch_target_rel_depth = torch_target_rel_depth.permute(2, 0, 1)
        torch_source_rel_depth = torch_source_rel_depth.permute(2, 0, 1)
        
        torch_source = pil_to_torch(pil_source)
        torch_target = pil_to_torch(pil_target)

        # torch_source = upsample(torch_source)
        # torch_source_abs_depth = upsample(torch_source_abs_depth)
        # torch_source_rel_depth = upsample(torch_source_rel_depth)
        # torch_target = upsample(torch_target)
        # torch_target_abs_depth = upsample(torch_target_abs_depth)
        # torch_target_rel_depth = upsample(torch_target_rel_depth)
        
        R_0, t_0 = json_dict["data"][data_index]["R_0"], json_dict["data"][data_index]["t_0"]
        R_1, t_1 = json_dict["data"][data_index]["R_1"], json_dict["data"][data_index]["t_1"]
        
        R_0 = torch.tensor(R_0, dtype=torch.float32)
        t_0 = torch.tensor(t_0, dtype=torch.float32)
        R_1 = torch.tensor(R_1, dtype=torch.float32)
        t_1 = torch.tensor(t_1, dtype=torch.float32)

        # Needed ?
        rot_matrix = torch.tensor(
            np.linalg.inv(
                np.array([[1, 0, 0, 0], 
                        [0, 0.9816272, -0.1908090, 0], 
                        [0, 0.1908090, 0.9816272, 0],
                        [0, 0, 0, 1]
                        ])
            )
        ).type(torch_source.dtype)
        
        P_0 = torch.eye(4).unsqueeze(0)
        P_0[:, :3, :3] = R_0
        P_0[:, :3, -1] = t_0
        # P_0 = P_0 @ rot_matrix
        
        P_1 = torch.eye(4).unsqueeze(0)
        P_1[:, :3, :3] = R_1
        P_1[:, :3, -1] = t_1
        # P_1 = P_1 @ rot_matrix

        ########################################
        ## Unproject 2D to 3D ##
        ########################################
        src_world_pts = cam2world(depth=torch_source_abs_depth, 
                                pose=P_0,
                                K1=intrinsic, 
                                device=device)
        
        tgt_world_pts = cam2world(depth=torch_target_abs_depth, 
                                pose=P_1,
                                K1=intrinsic, 
                                device=device)
        
        world_pts = [src_world_pts, tgt_world_pts]

        ########################################
        ## Reproject 3D to 2D ## 
        ########################################
        src_to_tgt_corr, src_to_tgt_warped_depth = world2cam(src_world_pts, P_1, intrinsic, device, depth=torch_source_abs_depth, return_warp_depth=True)
        tgt_to_src_corr, tgt_to_src_warped_depth = world2cam(tgt_world_pts, P_0, intrinsic, device, depth=torch_target_abs_depth, return_warp_depth=True)
        
        corr_map = [src_to_tgt_corr, tgt_to_src_corr]

        # index_put method -> does not require target image 
        rgbs = [torch_source, torch_target]
        
        # Depth at the novel view
        depths = [src_to_tgt_warped_depth, tgt_to_src_warped_depth]
        original_depths = [torch_source_abs_depth, torch_target_abs_depth]
        result = []
        batch_size, channel, height, width = torch_source.shape
        _i = 0
        for img, corr, depth_map in zip(rgbs, corr_map, depths):
            # corr 1 256 256 2
            # img 1 3 256 256
            # depth 1 256 256
            depth_map = depth_map.squeeze(-1).to(device)
            corr = corr.permute(0, 3, 1, 2).to(device)
            img = img.to(device)
            mask1 = (original_depths[_i].to(device) != 0) * (depth_map.to(device) > 0)
            novel_view_image, novel_view_mask = bilinear_splatting(frame1=img,
                                                                depth1=depth_map,
                                                                trans_pos=corr,
                                                                mask1=mask1,
                                                                flow12_mask=None,
                                                                is_image=True)
            _i += 1
            result.append(novel_view_image)

        # Sequence is now changed. 
        # Source is used to make Target.
        # Target is used to make source image.
        source_output = torch.cat([torch_source.detach().cpu().squeeze(), 
                            torch_source_rel_depth.detach().cpu().squeeze(0), 
                            result[1].detach().cpu().squeeze(0)], axis=-1)
        
        target_output = torch.cat([torch_target.detach().cpu().squeeze(), 
                            torch_target_rel_depth.detach().cpu().squeeze(0), 
                            result[0].detach().cpu().squeeze(0)], axis=-1)
        
        final_output = torch.cat([source_output, target_output], axis=1)
        if save_img:
            print("Saving images")
            fn = json_dict["data"][data_index]["source"].split("/")[-1].split(".")[0]
            torch_to_pil(final_output).save(f"./output/{fn}.png")
        
        if save_pc:
            print("Saving point cloud")
            pc_color = [torch_source, torch_target]
            _fn = [json_dict["data"][data_index]["source"].split("/")[-1].split(".")[0], \
                json_dict["data"][data_index]["target"].split("/")[-1].split(".")[0]]
            
            for _i, pts in enumerate(world_pts):
                pts = pts.reshape(-1, 3).detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(pc_color[_i].squeeze(0).permute(1, 2, 0).reshape(-1, 3))
                o3d.io.write_point_cloud(f"./output/{_fn[_i]}_point_cloud.ply", pcd, write_ascii=True)

if __name__ == "__main__":
    main()