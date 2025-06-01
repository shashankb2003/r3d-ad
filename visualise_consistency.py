import argparse
import os
import sys
import torch
import numpy as np
import open3d as o3d
import time

from utils.misc import *
from models.autoencoder import *

def load_point_cloud(file_path, num_points=2048):
    """
    Load a point cloud from file and preprocess it
    
    Args:
        file_path: Path to point cloud file (.pcd, .ply, .npy, .txt)
        num_points: Number of points to sample
    
    Returns:
        torch.Tensor: Preprocessed point cloud [1, N, 3]
        dict: Scaling information for converting back
    """
    
    # Load based on file extension
    if file_path.endswith('.pcd') or file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.array(pcd.points, dtype=np.float32)
    elif file_path.endswith('.npy'):
        points = np.load(file_path).astype(np.float32)
    elif file_path.endswith('.txt'):
        points = np.loadtxt(file_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded point cloud with {len(points)} points")
    
    # Sample to desired number of points
    if len(points) > num_points:
        choice = np.random.choice(len(points), num_points, replace=False)
        points = points[choice]
    elif len(points) < num_points:
        # Upsample by repeating points
        choice = np.random.choice(len(points), num_points, replace=True)
        points = points[choice]
    
    print(f"Resampled to {len(points)} points")
    
    # Center and scale (similar to dataset preprocessing)
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    points = points / scale
    
    # Convert to torch tensor and add batch dimension
    points_tensor = torch.from_numpy(points).unsqueeze(0)  # [1, N, 3]
    
    # Store scaling info for converting back to original coordinates
    scale_info = {
        'center': center,
        'scale': scale,
        'shift': torch.from_numpy(center).float(),
        'scale_tensor': torch.tensor(scale).float()
    }
    
    return points_tensor, scale_info

def visualize_diffusion_trajectory(trajectory, original_pc, scale_info, step_interval=10, auto_play=False, play_speed=0.5):
    """
    Visualize the diffusion trajectory step by step
    
    Args:
        trajectory: Dict containing point clouds at each timestep {T: pc_T, T-1: pc_T-1, ..., 0: pc_0}
        original_pc: Original input point cloud for comparison
        scale_info: Dictionary with scaling information
        step_interval: Show every N steps (to avoid too many windows)
        auto_play: If True, automatically advance through steps
        play_speed: Seconds to wait between steps in auto mode
    """
    
    # Convert original back to original coordinates
    original_np = original_pc[0].numpy() * scale_info['scale'] + scale_info['center']
    
    # Get timesteps in descending order (T to 0)
    timesteps = sorted(trajectory.keys(), reverse=True)
    
    # Filter timesteps based on interval
    display_timesteps = timesteps[::step_interval] + [0]  # Always include final step
    display_timesteps = sorted(list(set(display_timesteps)), reverse=True)
    
    print(f"Total timesteps: {len(timesteps)}")
    print(f"Displaying timesteps: {display_timesteps}")
    print(f"Original point cloud shape: {original_np.shape}")
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    for i, t in enumerate(display_timesteps):
        # Get point cloud at timestep t
        pc_t = trajectory[t]
        if torch.is_tensor(pc_t):
            pc_t = pc_t.cpu().numpy()
        
        # Convert back to original coordinates
        pc_t_scaled = pc_t[0] * scale_info['scale'] + scale_info['center']
        
        # Create point clouds for visualization
        # Current reconstruction step (blue to red gradient based on timestep)
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(pc_t_scaled)
        
        # Color based on timestep (blue=early/noisy, red=final/clean)
        color_intensity = 1.0 - (t / len(timesteps))  # 0 at T, 1 at 0
        current_pcd.paint_uniform_color([color_intensity, 0.2, 1.0 - color_intensity])
        
        # Original point cloud (orange)
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_np)
        original_pcd.paint_uniform_color([1, 0.706, 0])
        
        # Create visualization
        geometries = [current_pcd, original_pcd, coord_frame]
        
        window_name = f"Diffusion Step t={t} ({i+1}/{len(display_timesteps)})"
        print(f"\n{window_name}")
        print(f"Point cloud shape: {pc_t_scaled.shape}")
        
        if t == len(timesteps) - 1:  # First step (pure noise)
            print("Initial noise")
        elif t == 0:  # Final step
            print("Final reconstruction")
        else:
            print(f"Intermediate denoising step")
        
                # Save current step's point cloud
        save_dir = "saved_pcds"
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"step_{t:04d}.pcd"
        file_path = os.path.join(save_dir, file_name)
        o3d.io.write_point_cloud(file_path, current_pcd)
        print(f"Saved point cloud to {file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--model_type', type=str, required=True, help='diffusion or consistency')
    parser.add_argument('--pointcloud', type=str, required=True, help='Path to point cloud file (.pcd, .ply, .npy, .txt)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points to sample from input')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--step_interval', type=int, default=1, help='Show every N diffusion steps')
    parser.add_argument('--auto_play', action='store_true', help='Automatically advance through steps')
    parser.add_argument('--play_speed', type=float, default=0.5, help='Seconds between steps in auto mode')
    args = parser.parse_args()

    # Load checkpoint and model
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu',weights_only=False)
    ckpt_args = ckpt['args']
    ckpt_model_type = getattr(ckpt['args'], 'model_type', 'consistency')
    if ckpt_model_type == 'consistency':
        model = ConsistencyAutoEncoder(ckpt['args']).to(args.device)
    else:
        model = getattr(sys.modules[__name__], args.model)(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    print(f"Model has {ckpt_args.num_steps} diffusion steps")
    
    # Load point cloud
    print(f"Loading point cloud from {args.pointcloud}")
    points, scale_info = load_point_cloud(args.pointcloud, args.num_points)
    points = points.to(args.device)
    
    print(f"Point cloud shape: {points.shape}")
    
    with torch.no_grad():
        print("Encoding point cloud...")
        code = model.encode(points)
        print(f"Encoded to latent vector of size: {code.shape}")
        
        print("Generating diffusion trajectory...")
        # Get the full trajectory from T to 0
        trajectory = model.decode(code, points.size(1), flexibility=args.flexibility, ret_traj=True)
        
        print(f"Generated trajectory with {len(trajectory)} steps")
        
        # Handle relative reconstruction if needed
        if hasattr(ckpt_args, 'rel') and ckpt_args.rel:
            print("Applying relative reconstruction offset...")
            for t in trajectory:
                # Ensure both tensors are on the same device
                traj_tensor = trajectory[t]
                if torch.is_tensor(traj_tensor):
                    # Move both to CPU for consistency
                    trajectory[t] = traj_tensor.cpu() + points.cpu()

        
        print("\nStarting visualization...")
        print("Legend:")
        print("- Orange: Original input point cloud")
        print("- Blue→Red: Reconstruction process (blue=noisy start, red=clean end)")
        
        # Visualize the trajectory
        visualize_diffusion_trajectory(
            trajectory, points.cpu(), scale_info,
            step_interval=args.step_interval,
            auto_play=args.auto_play,
            play_speed=args.play_speed
        )

if __name__ == "__main__":
    main()
