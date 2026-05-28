"""Conservation and persistent homology analysis on 3D occupancy grids.

Core analyses:
  1. Build spatial adjacency graph from occupancy grid
  2. Compute Laplacian, eigenvectors, conservation ratios
  3. Fiedler vector → room partitioning
  4. Persistent homology → Betti numbers (topological skeleton)
  5. Multi-robot map fusion analysis (sheaf-inspired)
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

# Add SDK path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'conservation-spectral-python', 'src'))
from conservation_spectral import TensionGraph, build_laplacian, eigendecompose, conservation_ratio, conservation_ratios

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False


# ─── Graph Construction ──────────────────────────────────────────────

def occupancy_to_graph(
    grid: np.ndarray,
    connectivity: int = 6,
    weight_mode: str = "similarity",
) -> TensionGraph:
    """Build a TensionGraph from a 3D occupancy grid.

    Each voxel becomes a vertex. Edges connect spatial neighbors.
    Weight based on occupancy similarity.

    Args:
        grid: 3D numpy array (X, Y, Z) of log-odds values
        connectivity: 6 (face) or 26 (full 3x3x3 neighborhood minus center)
        weight_mode: "similarity" (exp(-|diff|)) or "distance" (1/distance)
    """
    xs, ys, zs = grid.shape
    n_voxels = xs * ys * zs
    flat = grid.flatten()

    # Build index mapping
    def idx(x, y, z):
        return x * ys * zs + y * zs + z

    # Neighbor offsets for 6-connectivity
    if connectivity == 6:
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    else:  # 26-connectivity
        offsets = [
            (dx, dy, dz)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

    # Build sparse adjacency directly
    rows, cols, weights = [], [], []
    for x in range(xs):
        for y in range(ys):
            for z in range(zs):
                i = idx(x, y, z)
                for dx, dy, dz in offsets:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < xs and 0 <= ny < ys and 0 <= nz < zs:
                        j = idx(nx, ny, nz)
                        diff = abs(flat[i] - flat[j])
                        if weight_mode == "similarity":
                            w = np.exp(-diff)
                        else:
                            w = 1.0 / (1.0 + diff)
                        rows.append(i)
                        cols.append(j)
                        weights.append(w)

    # Build TensionGraph
    g = TensionGraph(directed=False)
    for i in range(n_voxels):
        g.add_vertex(i)

    for i, j, w in zip(rows, cols, weights):
        if i < j:  # avoid double-adding for undirected
            g.add_edge(i, j, w)

    # Set occupancy as attribute
    g.set_attribute("occupancy_logodds", flat.astype(np.float64))
    g.set_attribute("occupancy_prob", (1.0 / (1.0 + np.exp(-flat))).astype(np.float64))

    return g


# ─── Conservation Analysis ────────────────────────────────────────────

@dataclass
class ConservationAnalysis:
    """Results of conservation analysis on an occupancy grid."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    fiedler_vector: np.ndarray
    fiedler_value: float
    conservation_scores: np.ndarray  # per-voxel conservation
    spectral_gap: float
    partition_labels: np.ndarray  # room labels from Fiedler


def analyze_conservation(
    grid: np.ndarray,
    k: int = 10,
    normalized: bool = True,
) -> ConservationAnalysis:
    """Run conservation spectral analysis on occupancy grid.

    Builds graph, computes Laplacian eigendecomposition, extracts Fiedler vector.
    """
    xs, ys, zs = grid.shape
    n = xs * ys * zs

    print(f"  Building spatial graph ({xs}x{ys}x{zs} = {n} voxels)...")

    # Build sparse adjacency directly (faster than TensionGraph for large grids)
    flat = grid.flatten()
    offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    rows, cols, weights = [], [], []

    for x in range(xs):
        for y in range(ys):
            for z in range(zs):
                i = x*ys*zs + y*zs + z
                for dx, dy, dz in offsets:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < xs and 0 <= ny < ys and 0 <= nz < zs:
                        j = nx*ys*zs + ny*zs + nz
                        diff = abs(flat[i] - flat[j])
                        w = np.exp(-diff)
                        rows.append(i)
                        cols.append(j)
                        weights.append(w)

    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sparse.diags(degrees)

    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L = sparse.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    print(f"  Computing {k} smallest eigenvectors...")
    k = min(k, n - 2)
    eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fiedler vector = eigenvector for 2nd smallest eigenvalue
    fiedler_vector = eigenvectors[:, 1]
    fiedler_value = eigenvalues[1]
    spectral_gap = eigenvalues[1] - eigenvalues[0]

    # Per-voxel conservation: how well the Fiedler partition preserves occupancy
    # Conservation score = 1 - variance of occupancy within each partition
    partition = (fiedler_vector > 0).astype(int)
    occ = 1.0 / (1.0 + np.exp(-flat))

    conservation_scores = np.zeros(n)
    for label in [0, 1]:
        mask = partition == label
        if mask.sum() > 0:
            local_var = np.var(occ[mask])
            conservation_scores[mask] = 1.0 - min(local_var * 4, 1.0)

    print(f"  Fiedler value: {fiedler_value:.6f}, Spectral gap: {spectral_gap:.6f}")
    print(f"  Partition: {(partition==0).sum()} vs {(partition==1).sum()} voxels")

    return ConservationAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        fiedler_vector=fiedler_vector,
        fiedler_value=fiedler_value,
        conservation_scores=conservation_scores,
        spectral_gap=spectral_gap,
        partition_labels=partition,
    )


# ─── Persistent Homology ─────────────────────────────────────────────

@dataclass
class TopologicalAnalysis:
    """Results of persistent homology computation."""
    betti_numbers: dict[int, int]  # dim -> count
    persistence_diagrams: dict[int, np.ndarray]  # dim -> array of (birth, death)
    significant_features: dict[int, list[tuple[float, float, float]]]  # dim -> [(birth, death, persistence)]

    def summary(self) -> str:
        lines = ["Topological Analysis:"]
        for dim in sorted(self.betti_numbers.keys()):
            lines.append(f"  β_{dim} = {self.betti_numbers[dim]}")
            if dim in self.significant_features:
                for b, d, p in self.significant_features[dim][:5]:
                    lines.append(f"    Feature: birth={b:.3f}, death={d:.3f}, persistence={p:.3f}")
        return "\n".join(lines)


def compute_persistent_homology(
    grid: np.ndarray,
    threshold: float = 0.5,
    max_alpha_square: float = 4.0,
    mode: str = "free",
) -> TopologicalAnalysis:
    """Compute persistent homology.

    Args:
        mode: "occupied" for occupied voxels, "free" for free space.
              Free space PH detects doorways/tunnels (β₁) and rooms (β₂).
    """
    prob = 1.0 / (1.0 + np.exp(-grid))
    if mode == "free":
        # Compute PH on FREE voxels — tunnels/doorways appear as β₁
        points = np.argwhere((prob < threshold) & (prob > 0.05))
    else:
        points = np.argwhere(prob > threshold)
    occupied = points

    if len(occupied) == 0:
        return TopologicalAnalysis(
            betti_numbers={0: 0, 1: 0, 2: 0},
            persistence_diagrams={},
            significant_features={},
        )

    print(f"  Computing persistent homology on {len(occupied)} occupied voxels...")

    if HAS_GUDHI:
        # Use alpha complex (much faster than Rips for 3D)
        try:
            alpha = gudhi.AlphaComplex(points=occupied.astype(np.float64))
            st = alpha.create_simplex_tree(max_alpha_square=max_alpha_square)
        except Exception:
            # Fallback to Rips for very small point clouds
            rips = gudhi.RipsComplex(points=occupied.astype(np.float64), max_edge_length=3.0)
            st = rips.create_simplex_tree(max_dimension=3)

        st.compute_persistence()

        betti = {0: 0, 1: 0, 2: 0}
        persistence_diagrams = {}
        significant = {0: [], 1: [], 2: []}

        for dim in range(3):
            pairs = st.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                persistence_diagrams[dim] = pairs
                for pair in pairs:
                    birth, death = pair[0], pair[1]
                    if np.isinf(death):
                        betti[dim] += 1
                        significant[dim].append((birth, death, float('inf')))
                    else:
                        pers = death - birth
                        if pers > 0.1:  # filter noise
                            significant[dim].append((birth, death, pers))

                # Sort by persistence
                for d in significant:
                    significant[d].sort(key=lambda x: x[2] if x[2] != float('inf') else 999, reverse=True)
    else:
        # Manual fallback: simple connected components for β₀
        print("  [gudhi not available, computing β₀ only via connected components]")
        betti = {0: 0, 1: 0, 2: 0}
        significant = {0: [], 1: [], 2: []}
        persistence_diagrams = {}

        # BFS connected components
        visited = set()
        occupied_set = set(map(tuple, occupied))
        for pt in occupied:
            pt = tuple(pt)
            if pt in visited:
                continue
            # BFS
            queue = [pt]
            visited.add(pt)
            while queue:
                curr = queue.pop(0)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nbr = (curr[0]+dx, curr[1]+dy, curr[2]+dz)
                            if nbr in occupied_set and nbr not in visited:
                                visited.add(nbr)
                                queue.append(nbr)
            betti[0] += 1

    return TopologicalAnalysis(
        betti_numbers=betti,
        persistence_diagrams=persistence_diagrams,
        significant_features=significant,
    )


# ─── Multi-Robot Fusion (Sheaf-Inspired) ─────────────────────────────

@dataclass
class FusionAnalysis:
    """Results of multi-robot map fusion analysis."""
    conservation_a: float
    conservation_b: float
    conservation_merged: float
    conservation_conflict: Optional[float]
    overlap_consistency: float  # consistency in overlap region (consistent maps)
    conflict_overlap_consistency: Optional[float]  # consistency in overlap (conflicting maps)
    consistent: bool  # True if overlap is consistent
    conflict_detected: bool


def analyze_fusion(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    grid_truth: np.ndarray,
    conflict_grid_a: Optional[np.ndarray] = None,
    conflict_grid_b: Optional[np.ndarray] = None,
) -> FusionAnalysis:
    """Analyze multi-robot map fusion from a sheaf-cohomology perspective.

    Core insight: If two partial maps are consistent, the merged map's conservation
    should be >= max(individual). If they conflict, conservation drops (H¹ ≠ 0).
    """
    def quick_conservation(grid):
        """Conservation score: 1 - average local variance of occupancy.
        
        High = smooth, coherent map. Low = noisy, inconsistent map.
        """
        prob = 1.0 / (1.0 + np.exp(-grid))
        xs, ys, zs = grid.shape
        # Compute local variance using convolution-like approach
        total_var = 0.0
        count = 0
        for x in range(1, xs-1):
            for y in range(1, ys-1):
                for z in range(1, zs-1):
                    center = prob[x, y, z]
                    nbrs = [prob[x-1,y,z], prob[x+1,y,z],
                            prob[x,y-1,z], prob[x,y+1,z],
                            prob[x,y,z-1], prob[x,y,z+1]]
                    local_var = sum((n - center)**2 for n in nbrs) / 6.0
                    total_var += local_var
                    count += 1
        avg_var = total_var / max(count, 1)
        return 1.0 - min(avg_var * 8, 1.0)

    def overlap_conservation(ga, gb, overlap_mask):
        """Conservation specifically in the overlap region — measures consistency."""
        prob_a = 1.0 / (1.0 + np.exp(-ga))
        prob_b = 1.0 / (1.0 + np.exp(-gb))
        if overlap_mask.sum() == 0:
            return 1.0
        diff = (prob_a[overlap_mask] - prob_b[overlap_mask])**2
        mse = diff.mean()
        return 1.0 - min(mse * 4, 1.0)

    print("  Computing conservation for Robot A...")
    cons_a = quick_conservation(grid_a)
    print(f"    Robot A conservation: {cons_a:.4f}")

    print("  Computing conservation for Robot B...")
    cons_b = quick_conservation(grid_b)
    print(f"    Robot B conservation: {cons_b:.4f}")

    # Consistent merge: weighted average where both observed
    print("  Merging consistent maps...")
    obs_a_mask = np.abs(grid_a) > 0.1  # observed by A
    obs_b_mask = np.abs(grid_b) > 0.1  # observed by B
    overlap = obs_a_mask & obs_b_mask

    merged = grid_truth.copy()
    merged[obs_a_mask & ~obs_b_mask] = grid_a[obs_a_mask & ~obs_b_mask]
    merged[obs_b_mask & ~obs_a_mask] = grid_b[obs_b_mask & ~obs_a_mask]
    merged[overlap] = (grid_a[overlap] + grid_b[overlap]) / 2.0

    cons_merged = quick_conservation(merged)
    print(f"    Merged conservation: {cons_merged:.4f}")

    # Overlap consistency for consistent maps
    cons_overlap = overlap_conservation(grid_a, grid_b, overlap)
    print(f"    Overlap consistency: {cons_overlap:.4f}")

    # Conflict analysis
    cons_conflict = None
    cons_conflict_overlap = None
    if conflict_grid_a is not None and conflict_grid_b is not None:
        print("  Merging CONFLICTING maps...")
        conflict_merged = grid_truth.copy()
        conflict_merged[obs_a_mask & ~obs_b_mask] = conflict_grid_a[obs_a_mask & ~obs_b_mask]
        conflict_merged[obs_b_mask & ~obs_a_mask] = conflict_grid_b[obs_b_mask & ~obs_a_mask]
        conflict_merged[overlap] = (conflict_grid_a[overlap] + conflict_grid_b[overlap]) / 2.0
        cons_conflict = quick_conservation(conflict_merged)
        cons_conflict_overlap = overlap_conservation(conflict_grid_a, conflict_grid_b, overlap)
        print(f"    Conflict merge conservation: {cons_conflict:.4f}")
        print(f"    Conflict overlap consistency: {cons_conflict_overlap:.4f}")

    consistent = cons_overlap > 0.7  # overlap is mostly consistent
    conflict_detected = cons_conflict_overlap is not None and cons_conflict_overlap < cons_overlap - 0.1

    return FusionAnalysis(
        conservation_a=cons_a,
        conservation_b=cons_b,
        conservation_merged=cons_merged,
        conservation_conflict=cons_conflict,
        overlap_consistency=cons_overlap,
        conflict_overlap_consistency=cons_conflict_overlap,
        consistent=consistent,
        conflict_detected=conflict_detected,
    )
