
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterator
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def load_episode(episode_dir: Path) -> Dict[str, Any]:

    episode_dir = Path(episode_dir)

    data = np.load(episode_dir, allow_pickle=True).item()
    for key, value in data.items():
        print(f"{str(key)}: shape={value.shape}, dtype={value.dtype}")

    observations = _reconstruct_observations(data)
    actions = data.get('actions')
    
    episode_data = {
        'actions': actions,
        'observations': observations,
        'episode_dir': str(episode_dir),
    }
    
    return episode_data


def _reconstruct_observations(data: Dict[str, np.ndarray]):
    observations = {}
    for key, value in data.items():
        if key.startswith('obs_'):
            observations[key] = value
    return observations

def view_episode_images(
    episode: Dict[str, Any],
    camera_key: str = 'external_cam',
    num_frames: int = 9,
    save_path: Optional[Path] = None
):
    
    images = None
    obs = episode['observations']
    images = obs[camera_key]
    if len(images.shape) == 5 and images.shape[1] == 1:
        images = images[:, 0, :, :, :]

    # Select frames to display
    total_frames = len(images)
    num_frames = min(num_frames, total_frames)
    
    # Uniformly sample frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Compute grid size
    n_cols = min(3, num_frames)
    n_rows = (num_frames + n_cols - 1) // n_cols
    
    # Create image grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    if num_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Display images
    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        img = images[frame_idx]

        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]
        
        # If image is float and in [0, 1] range, convert to uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.set_title(f'Frame {frame_idx}/{total_frames-1}', fontsize=12)
        ax.axis('off')
    
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved visualization to {save_path}")
    
    plt.show()
    print(f"{'='*70}\n")

def create_video_from_episode(
    episode: Dict[str, Any],
    camera_key: str = 'external_cam',
    output_path: str = 'episode_video.mp4',
    fps: int = 10
):
    obs = episode['observations']
    images = obs[camera_key]
    
    if len(images.shape) == 5 and images.shape[1] == 1:
        images = images[:, 0, :, :, :]
 
    first_frame = images[0]
    
    if len(first_frame.shape) == 4 and first_frame.shape[0] == 1:
        first_frame = first_frame[0]
    
    height, width = first_frame.shape[:2]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Creating video from {len(images)} frames...")
    print(f"{'='*70}")
    print(f"  Frame shape: {first_frame.shape}")
    print(f"  Video resolution: {width}x{height}")
    print(f"  Output: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for i, img in enumerate(images):
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        out.write(img_bgr)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(images)} frames")
    
    out.release()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n Video saved successfully!")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Total frames: {len(images)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(images)/fps:.2f} seconds")
    print(f"{'='*70}\n")
    
    return output_path

def create_gif_from_episode(
    episode: Dict[str, Any],
    camera_key: str = 'external_cam',
    output_path: str = 'episode_video.gif',
    fps: int = 10,
    max_frames: Optional[int] = None,
    resize_factor: float = 1.0,
    optimize: bool = True
):
    """
    Create GIF from episode images
    
    Args:
        episode: Episode data dictionary
        camera_key: Camera key in observations
        output_path: Output GIF path
        fps: Frames per second
        max_frames: Maximum number of frames (None for all frames)
        resize_factor: Resize factor for images (0.5 = half size)
        optimize: Enable GIF optimization (smaller file size)
    """
    obs = episode['observations']
    images = obs[camera_key]
    
    # Remove extra dimensions
    if len(images.shape) == 5 and images.shape[1] == 1:
        images = images[:, 0, :, :, :]
    
    # Limit number of frames if specified
    if max_frames is not None and len(images) > max_frames:
        # Uniformly sample frames
        indices = np.linspace(0, len(images) - 1, max_frames, dtype=int)
        images = images[indices]
    
    first_frame = images[0]
    if len(first_frame.shape) == 4 and first_frame.shape[0] == 1:
        first_frame = first_frame[0]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Creating GIF from {len(images)} frames...")
    print(f"{'='*70}")
    print(f"  Frame shape: {first_frame.shape}")
    print(f"  Resize factor: {resize_factor}")
    print(f"  Output: {output_path}")
    
    # Convert frames to PIL Images
    pil_images = []
    for i, img in enumerate(images):
        # Remove extra dimension
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]
        
        # Convert to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Resize if specified
        if resize_factor != 1.0:
            new_width = int(pil_img.width * resize_factor)
            new_height = int(pil_img.height * resize_factor)
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        pil_images.append(pil_img)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(images)} frames")
    
    # Save as GIF
    duration = int(1000 / fps)  # Duration per frame in milliseconds
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,  # 0 = infinite loop
        optimize=optimize
    )
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n GIF saved successfully!")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Total frames: {len(pil_images)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(pil_images)/fps:.2f} seconds")
    print(f"  Resolution: {pil_images[0].width}x{pil_images[0].height}")
    print(f"{'='*70}\n")
    
    return output_path


def create_video_and_gif_from_episode(
    episode: Dict[str, Any],
    camera_key: str = 'external_cam',
    output_dir: str = 'output',
    base_name: str = 'episode',
    fps: int = 10,
    gif_resize_factor: float = 0.5,
    gif_max_frames: Optional[int] = None
):
    """
    Create both MP4 video and GIF from episode
    
    Args:
        episode: Episode data dictionary
        camera_key: Camera key in observations
        output_dir: Output directory
        base_name: Base filename (without extension)
        fps: Frames per second
        gif_resize_factor: Resize factor for GIF (to reduce file size)
        gif_max_frames: Maximum frames for GIF (None = all frames)
    
    Returns:
        Tuple of (video_path, gif_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video
    video_path = output_dir / f"{base_name}.mp4"
    create_video_from_episode(
        episode,
        camera_key=camera_key,
        output_path=str(video_path),
        fps=fps
    )
    
    # Create GIF
    gif_path = output_dir / f"{base_name}.gif"
    create_gif_from_episode(
        episode,
        camera_key=camera_key,
        output_path=str(gif_path),
        fps=fps,
        max_frames=gif_max_frames,
        resize_factor=gif_resize_factor
    )
    
    return video_path, gif_path

if __name__ == "__main__":
    from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

    episode_path = f"{REAL2SIM_ROOT_DIR}/scripts/real2sim-v0/dataset/pick up cube and place in bowl/20251114_151543/episode_0001"
    if Path(episode_path).exists():
        episode = load_episode(episode_path)
        print(f" Loaded episode from {episode_path}")
        print(f"  Actions shape: {episode['actions'].shape}")
        print(f"  Observations: {list(episode['observations'].keys())}")

        camera_keys = ['obs_policy_external_cam','obs_policy_wrist_cam','obs_policy_external_cam_front']
        for camera_key in camera_keys:
            output_dir = Path(f'{REAL2SIM_ROOT_DIR}/scripts/real2sim-v0/dataset/pictures/pick up cube/20251121_205523/{camera_key}')
            view_episode_images(
                episode,
                camera_key=camera_key,
                num_frames=1,
                save_path=output_dir / f"{camera_key}.png"
            )

        
        print("="*70)
        print("Creating Video")
        print("="*70)
        camera_keys = ['obs_policy_external_cam','obs_policy_wrist_cam','obs_policy_external_cam_front']
        for camera_key in camera_keys:
            output_dir = f'{REAL2SIM_ROOT_DIR}/scripts/real2sim-v0/dataset/videos/pick up cube and place in bowl/20251114_151543/{camera_key}'
            video_path, gif_path = create_video_and_gif_from_episode(
                episode,
                camera_key=camera_key,
                output_dir=output_dir,
                base_name='episode',
                fps=10,
                gif_resize_factor=0.5,  # GIF scaled down to 50%
                gif_max_frames=100       # GIF limited to 100 frames
            )
    else:
        print(f"Episode not found: {episode_path}")