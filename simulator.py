"""Simulate synthetic 3D occupancy grids for the OctoMap conservation POC.

Environments:
  - Two rooms connected by a corridor (doorway)
  - Sensor noise (random occupancy flips)
  - Dynamic objects (moving entity)
  - Multi-robot observations (partial, noisy views)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OccupancyGrid:
    """3D occupancy grid with log-odds values."""
    grid: np.ndarray          # shape (X, Y, Z), float log-odds
    resolution: float = 0.1   # meters per voxel
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def shape(self):
        return self.grid.shape

    def to_probability(self) -> np.ndarray:
        """Convert log-odds to probability."""
        return 1.0 / (1.0 + np.exp(-self.grid))

    def occupied_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Binary occupancy at given probability threshold."""
        return self.to_probability() > threshold

    def free_mask(self, threshold: float = 0.5) -> np.ndarray:
        return self.to_probability() < (1.0 - threshold)


def logodds(p: float) -> float:
    return np.log(p / (1.0 - p))


def make_two_rooms(
    x_size: int = 20,
    y_size: int = 20,
    z_size: int = 8,
    corridor_width: int = 2,
    corridor_z: int = 4,
) -> OccupancyGrid:
    """Create a two-room environment connected by a corridor.

    Layout (top-down, z=mid):
      - Room A: left half (x < x_size//2)
      - Room B: right half (x >= x_size//2)
      - Wall divides them at x = x_size//2
      - Corridor through the wall at y center, spanning corridor_width
    """
    grid = np.full((x_size, y_size, z_size), logodds(0.3))  # mostly free

    # Walls (outer boundary) — occupied
    LO = logodds(0.95)
    grid[0, :, :] = LO
    grid[-1, :, :] = LO
    grid[:, 0, :] = LO
    grid[:, -1, :] = LO
    grid[:, :, 0] = LO    # floor
    grid[:, :, -1] = LO   # ceiling

    # Dividing wall at x = x_size // 2
    mid_x = x_size // 2
    grid[mid_x, :, :] = LO

    # Cut corridor through the dividing wall
    mid_y = y_size // 2
    y_start = mid_y - corridor_width // 2
    y_end = y_start + corridor_width
    z_start = 1  # above floor
    z_end = min(corridor_z, z_size - 1)
    grid[mid_x, y_start:y_end, z_start:z_end] = logodds(0.1)  # free space

    return OccupancyGrid(grid=grid, resolution=0.1)


def add_sensor_noise(grid: OccupancyGrid, noise_rate: float = 0.05, seed: int = 42) -> OccupancyGrid:
    """Flip random voxels to simulate sensor noise."""
    rng = np.random.default_rng(seed)
    mask = rng.random(grid.shape) < noise_rate
    new_grid = grid.grid.copy()
    # Flip toward opposite
    new_grid[mask] = -new_grid[mask] * 0.5
    return OccupancyGrid(grid=new_grid, resolution=grid.resolution, origin=grid.origin.copy())


def add_dynamic_object(
    grid: OccupancyGrid,
    positions: list[tuple[int, int, int]],
    radius: int = 2,
    occupancy_logodds: float = 2.0,
) -> OccupancyGrid:
    """Add a spherical dynamic object at given positions (simulates motion blur)."""
    new_grid = grid.grid.copy()
    xs, ys, zs = grid.shape
    for cx, cy, cz in positions:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        nx, ny, nz = cx + dx, cy + dy, cz + dz
                        if 0 <= nx < xs and 0 <= ny < ys and 0 <= nz < zs:
                            new_grid[nx, ny, nz] = occupancy_logodds
    return OccupancyGrid(grid=new_grid, resolution=grid.resolution, origin=grid.origin.copy())


def robot_observation(
    grid: OccupancyGrid,
    viewpoint: str = "left",
    observation_noise: float = 0.08,
    partial_fraction: float = 0.85,
    seed: int = 0,
) -> OccupancyGrid:
    """Simulate a single robot's partial, noisy observation of the environment.

    Args:
        viewpoint: "left" (room A) or "right" (room B)
        observation_noise: noise rate for this robot
        partial_fraction: fraction of voxels the robot can see
        seed: RNG seed
    """
    rng = np.random.default_rng(seed)
    new_grid = np.full_like(grid.grid, logodds(0.5))  # unknown = 0.5

    xs = grid.shape[0]
    if viewpoint == "left":
        vis_mask = np.zeros(grid.shape, dtype=bool)
        vis_mask[:xs//2 + 2, :, :] = True
        vis_mask[xs//2 + 2:, :, :] = rng.random((xs - xs//2 - 2, grid.shape[1], grid.shape[2])) < 0.3
    else:
        vis_mask = np.zeros(grid.shape, dtype=bool)
        vis_mask[xs//2 - 2:, :, :] = True
        vis_mask[:xs//2 - 2, :, :] = rng.random((xs//2 - 2, grid.shape[1], grid.shape[2])) < 0.3

    # Apply partial observability
    see_mask = vis_mask & (rng.random(grid.shape) < partial_fraction)
    new_grid[see_mask] = grid.grid[see_mask]

    # Add observation noise
    noise_mask = rng.random(grid.shape) < observation_noise
    new_grid[noise_mask] += rng.normal(0, 0.5, size=np.sum(noise_mask))

    return OccupancyGrid(grid=new_grid, resolution=grid.resolution, origin=grid.origin.copy())


def create_conflicting_observations(grid: OccupancyGrid, seed: int = 99) -> tuple[OccupancyGrid, OccupancyGrid]:
    """Create two robot observations that CONFLICT in the overlap region.

    Robot A thinks the corridor is occupied; Robot B thinks it's free.
    """
    obs_a = robot_observation(grid, viewpoint="left", seed=seed)
    obs_b = robot_observation(grid, viewpoint="right", seed=seed + 1)

    # Introduce conflict: Robot A thinks corridor area is occupied
    xs = grid.shape[0]
    mid_x = xs // 2
    mid_y = grid.shape[1] // 2
    obs_a.grid[mid_x-1:mid_x+2, mid_y-1:mid_y+2, 1:grid.shape[2]-1] = logodds(0.9)

    # Robot B thinks corridor is free (already is in ground truth)
    obs_b.grid[mid_x-1:mid_x+2, mid_y-1:mid_y+2, 1:grid.shape[2]-1] = logodds(0.1)

    return obs_a, obs_b
