import json
import os
import numpy as np
from PIL import Image
import open3d as o3d

def get_pcd_from_depth(depth, h=480, w=640, focal_length=517.97):
    """
    Convert depth map to 3D point cloud.
    
    Args:
    - depth (numpy.array): Depth map of shape (h, w).
    - h (int): Image height.
    - w (int): Image width.
    - focal_length (float): Focal length of the camera.
    
    Returns:
    - numpy.array: 3D point cloud of shape (h, w, 3).
    """
    
    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize pixel coordinates
    normalized_x = (x - w / 2.0) / focal_length
    normalized_y = (y - h / 2.0) / focal_length

    # Compute 3D coordinates
    Z = depth
    X = normalized_x * Z
    Y = normalized_y * Z
    
    # Stack X, Y, Z to create point cloud
    pcd = np.stack((X, Y, Z), axis=-1)
    
    return pcd

def save_pcd(pts, color, output):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(color / 255.)
    o3d.io.write_point_cloud(output, pcd, write_ascii=True)

def project_cam_to_world(pts_cam, camera_pose):
    pts_world = (np.array(camera_pose['rotation']) @ (pts_cam * np.array([1, -1, -1])).T).T + np.array(camera_pose['position'])
    return pts_world


def main():
    json_path = "./input/metadata.json"
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)

    print(f"Total data: {len(json_dict['data'])}")

    focal_length = 517.97
    offset_x = 320
    offset_y = 240

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
        
        R_0, t_0 = json_dict["data"][data_index]["R_0"], json_dict["data"][data_index]["t_0"]
        R_1, t_1 = json_dict["data"][data_index]["R_1"], json_dict["data"][data_index]["t_1"]

        """
        begin editing
        """
        # unproject depth to pcd in camera frame.
        pcd_source_cam = get_pcd_from_depth(np_source_abs_depth, h=480, w=640, focal_length=517.97)
        pcd_target_cam = get_pcd_from_depth(np_target_abs_depth, h=480, w=640, focal_length=517.97)
        save_pcd(pcd_source_cam.reshape(-1, 3), np.array(pil_source).reshape(-1, 3), f'debug/{data_index}_source_cam.ply')
        save_pcd(pcd_target_cam.reshape(-1, 3), np.array(pil_target).reshape(-1, 3), f'debug/{data_index}_target_cam.ply')
        # transform camera frame to world frame.
        pcd_source_world = project_cam_to_world(pcd_source_cam.reshape(-1, 3), {'rotation': R_0, 'position': t_0})
        pcd_target_world = project_cam_to_world(pcd_target_cam.reshape(-1, 3), {'rotation': R_1, 'position': t_1})
        save_pcd(pcd_source_world.reshape(-1, 3), np.array(pil_source).reshape(-1, 3), f'debug/{data_index}_source_world.ply')
        save_pcd(pcd_target_world.reshape(-1, 3), np.array(pil_target).reshape(-1, 3), f'debug/{data_index}_target_world.ply')

        """
        end editing
        """

if __name__ == "__main__":
    main()