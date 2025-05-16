"""
This file is provided as-is and does not require modification.
If you want to add custom data augmentation during training, feel free to extend this file.

Design pattern of the transforms:
1. Take in dictionary of sample data
2. Look for specific inputs in the sample
3. Process the inputs
4. Add new data to the sample
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as tv_transforms

from .road_utils import Track, homogeneous
import torch
from PIL import Image as PILImage
import random
from PIL import ImageEnhance
import torchvision.transforms.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import ColorJitter

def project(points, view, proj, h, w):
    points_uv_raw = points @ view @ proj
    points_uv = points_uv_raw / points_uv_raw[:, -1:]

    # convert from uv to pixel coordinates, [0, W] and [0, H]
    points_img = points_uv[:, :2]
    points_img[:, 0] = (points_img[:, 0] + 1) * w / 2
    points_img[:, 1] = (1 - points_img[:, 1]) * h / 2

    mask = (
        (points_uv_raw[:, -1] > 1)  # must be in front of camera
        & (points_uv_raw[:, -1] < 15)  # don't render too far
        & (points_img[:, 0] >= 0)  # projected in valid img width
        & (points_img[:, 0] < w)
        & (points_img[:, 1] >= 0)  # projected in valid img height
        & (points_img[:, 1] < h)
    )

    return points_img[mask], mask


def rasterize_lines(
    points: np.ndarray,
    canvas: np.ndarray,
    color: int,
    thickness: int = 4,
):
    for i in range(len(points) - 1):
        start = points[i].astype(int)
        end = points[i + 1].astype(int)

        cv2.line(canvas, tuple(start), tuple(end), color, thickness)


def pad(points: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pads/truncates the points to a set length

    Args:
        points (np.ndarray): sequence of points with shape (n, d)

    Returns:
        tuple[np.ndarray, np.ndarray]: padded points (max_length, d) and mask (max_length,)
    """
    truncated_points = points[:max_length]

    # create a mask denoting which points are valid
    mask = np.ones(max_length, dtype=bool)
    mask[len(truncated_points) :] = False

    required_padding = max_length - len(truncated_points)

    if required_padding > 0:
        # pad with the last element
        if len(truncated_points) == 0:
            padding = np.zeros((required_padding, points.shape[1]), dtype=np.float32)
        else:
            padding = np.repeat(truncated_points[-1:], required_padding, axis=0)
        padded_points = np.concatenate([truncated_points, padding])
    else:
        padded_points = truncated_points

    return padded_points, mask


def create_pose_matrix(
    location: np.ndarray,
    front: np.ndarray,
    up: np.ndarray = [0, 1, 0],
    eps: float = 1e-5,
):
    """
    Args:
        location: cart position
        front: Point the camera is looking at
        up: up vector, default is Y-up [0, 1, 0]

    Returns:
        4x4 matrix
    """
    forward = front - location
    forward = forward / (np.linalg.norm(forward) + eps)

    # calculate right vector (x-axis)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + eps)

    # recalculate up vector (y-axis) to ensure orthogonality
    up = np.cross(right, forward)

    # create matrix representations and compose
    R = np.eye(4)
    R[:3, :3] = np.vstack((-right, up, forward))
    T = np.eye(4)
    T[:3, 3] = -location
    pose_matrix = R @ T

    return pose_matrix


class Compose(tv_transforms.Compose):
    def __call__(self, sample: dict):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ImageLoader:
    def __init__(self, episode_path: str):
        self.episode_path = Path(episode_path)

    def __call__(self, sample: dict):
        image_path = self.episode_path / f"{sample['_idx']:05d}_im.jpg"
        image = np.uint8(Image.open(image_path)) / 255.0
        image = image.transpose(2, 0, 1)

        sample["image"] = image.astype(np.float32)

        return sample


class DepthLoader(ImageLoader):
    def __call__(self, sample: dict):
        depth_path = self.episode_path / f"{sample['_idx']:05d}_depth.png"
        depth = np.uint16(Image.open(depth_path)) / 65535.0

        sample["depth"] = depth.astype(np.float32)

        return sample


class RandomHorizontalFlip(tv_transforms.RandomHorizontalFlip):
    def __call__(self, sample: dict):
        if np.random.rand() < self.p:
            sample["image"] = np.flip(sample["image"], axis=2)
            sample["track"] = np.flip(sample["track"], axis=1)

        return sample


class TrackProcessor:
    """
    Provides segmentation labels for left and right track
    """
    def __init__(self, track: Track):
        self.track = track

    def __call__(self, sample: dict):
        idx = sample["_idx"]
        frames = sample["_frames"]
        image = sample["image"]
        distance_down_track = frames["distance_down_track"][idx]
        proj = frames["P"][idx].copy()
        view = frames["V"][idx].copy()
        view[-1, :3] += -1.0 * view[1, :3]

        track_left, track_right = self.track.get_boundaries(distance_down_track)

        # project to image plane
        h, w = image.shape[1:]
        track_left, _ = project(track_left, view, proj, h, w)
        track_right, _ = project(track_right, view, proj, h, w)

        # draw line segments onto a blank canvas
        track = np.zeros((h, w), dtype=np.uint8)
        rasterize_lines(track_left, track, color=1)
        rasterize_lines(track_right, track, color=2)

        sample["track"] = track.astype(np.int64)

        return sample


class EgoTrackProcessor:
    """
    Provides round boundary point labels and target waypoints
    """
    def __init__(
        self,
        track: Track,
        n_track: int = 10,
        n_waypoints: int = 3,
        skip: int = 1,
    ):
        self.track = track
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.skip = skip

    def __call__(self, sample: dict):
        frames = sample["_frames"]
        idx = sample["_idx"]

        front = frames["front"][idx]
        location = frames["location"][idx]
        distance_down_track = frames["distance_down_track"][idx]

        # use future location as target waypoints
        waypoints = frames["location"][idx : idx + (self.n_waypoints + 1) * self.skip : self.skip][1:]
        waypoints = homogeneous(waypoints)

        sample_info = self.from_frame(location, front, distance_down_track, waypoints)
        sample.update(sample_info)

        return sample

    def from_frame(
        self,
        location: np.ndarray,
        front: np.ndarray,
        distance_down_track: float,
        waypoints: np.ndarray | None = None,
        **kwargs,
    ):
        if waypoints is None:
            waypoints = np.zeros((1, 4), dtype=np.float32)

        world2ego = create_pose_matrix(location, front)
        track_left, track_right = self.track.get_boundaries(
            distance_down_track,
            n_points=self.n_track,
        )

        # convert to frame of kart (ego)
        track_left = track_left @ world2ego.T
        track_right = track_right @ world2ego.T
        waypoints = waypoints @ world2ego.T

        # project to bird's eye view (bev)
        track_left = track_left[:, [0, 2]]
        track_right = track_right[:, [0, 2]]
        waypoints = waypoints[:, [0, 2]]

        # make sure points are expected size
        track_left, _ = pad(track_left, self.n_track)
        track_right, _ = pad(track_right, self.n_track)
        waypoints, waypoints_mask = pad(waypoints, self.n_waypoints)

        return {
            "track_left": track_left.astype(np.float32),
            "track_right": track_right.astype(np.float32),
            "waypoints": waypoints.astype(np.float32),
            "waypoints_mask": waypoints_mask,
        }

class Cutout:
    def __init__(self, num_holes=1, max_hole_size=(16, 16)):
        """
        Randomly masks out one or more square patches from the image.
        
        Args:
            num_holes (int): Number of patches to mask out of each image.
            max_hole_size (tuple): Maximum size (height, width) of each patch.
        """
        self.num_holes = num_holes
        self.max_hole_size = max_hole_size

    def __call__(self, sample: dict):
        image = sample["image"]
        h, w = image.shape[1], image.shape[2]

        for _ in range(self.num_holes):
            hole_height = random.randint(1, self.max_hole_size[0])
            hole_width = random.randint(1, self.max_hole_size[1])

            # Randomly choose where to place the hole
            top = random.randint(0, h - hole_height)
            left = random.randint(0, w - hole_width)

            # Apply cutout (zero-out the region)
            image[:, top:top + hole_height, left:left + hole_width] = 0
        
        sample["image"] = image
        return sample

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        """
        Adds Gaussian noise to the image.
        
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        image = sample["image"]
        noise = torch.randn(image.size()) * self.std + self.mean
        image = image + noise
        image = torch.clamp(image, 0.0, 1.0)  # Ensure values stay in valid range
        sample["image"] = image
        return sample

class Augmentations:
    def __init__(self):
        self.transforms = tv_transforms.Compose([
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.RandomVerticalFlip(),
            tv_transforms.RandomRotation(10),  # Rotate images by up to 10 degrees
            tv_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1),  # Adjust color properties
            tv_transforms.RandomGrayscale(p=0.1),
            tv_transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            tv_transforms.RandomCrop(size=(96, 128), padding=4),
            tv_transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            tv_transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            tv_transforms.ToTensor(),  # Convert to tensor  # Add Gaussian noise
            tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
        ])

    def __call__(self, sample: dict):
        # Convert the NumPy array to a PIL image
        image = PILImage.fromarray((sample["image"] * 255).astype(np.uint8).transpose(1, 2, 0))
        sample["image"] = self.transforms(image)
        return sample



    


