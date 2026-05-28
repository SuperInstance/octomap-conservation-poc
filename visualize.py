"""Visualization for the OctoMap Conservation POC.

Plots:
  1. Occupancy grid slice colored by conservation score
  2. Fiedler partition (rooms in different colors)
  3. Persistence diagram / Betti number display
  4. Multi-robot merge comparison
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import os

from simulator import OccupancyGrid, logodds
from analysis import ConservationAnalysis, TopologicalAnalysis, FusionAnalysis


def plot_occupancy_slice(
    grid: np.ndarray,
    z_slice: int = None,
    title: str = "Occupancy Grid",
    save_path: str = None,
    conservation_scores: np.ndarray = None,
    shape: tuple = None,
):
    """Plot a 2D slice of the occupancy grid, optionally colored by conservation."""
    xs, ys, zs = grid.shape
    if z_slice is None:
        z_slice = zs // 2

    prob = 1.0 / (1.0 + np.exp(-grid[:, :, z_slice]))

    fig, axes = plt.subplots(1, 2 if conservation_scores is not None else 1, figsize=(14, 6))
    if conservation_scores is None:
        axes = [axes]

    # Occupancy probability
    im0 = axes[0].imshow(prob.T, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_title(f"{title} — Occupancy (z={z_slice})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axes[0], label="P(occupied)")

    if conservation_scores is not None and shape is not None:
        cons_2d = conservation_scores.reshape(shape)[:, :, z_slice]
        im1 = axes[1].imshow(cons_2d.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title(f"{title} — Conservation Score (z={z_slice})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        plt.colorbar(im1, ax=axes[1], label="Conservation")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fiedler_partition(
    analysis: ConservationAnalysis,
    grid_shape: tuple,
    z_slice: int = None,
    save_path: str = None,
):
    """Visualize Fiedler vector partition (automatic room detection)."""
    xs, ys, zs = grid_shape
    if z_slice is None:
        z_slice = zs // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fiedler vector values
    fiedler_2d = analysis.fiedler_vector.reshape(grid_shape)[:, :, z_slice]
    im0 = axes[0].imshow(fiedler_2d.T, origin="lower", cmap="coolwarm")
    axes[0].set_title(f"Fiedler Vector (z={z_slice})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axes[0], label="Fiedler value")

    # Partition
    part_2d = analysis.partition_labels.reshape(grid_shape)[:, :, z_slice]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#3498db", "#e74c3c"])
    im1 = axes[1].imshow(part_2d.T, origin="lower", cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"Fiedler Partition — Room Detection (z={z_slice})")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#3498db", label="Room A"),
                       Patch(facecolor="#e74c3c", label="Room B")]
    axes[1].legend(handles=legend_elements, loc="upper right")

    # Eigenvalue spectrum
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    save_path: str = None,
):
    """Plot eigenvalue spectrum with spectral gap highlighted."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(eigenvalues))
    ax.bar(x, eigenvalues, color=["#2ecc71" if i == 1 else "#95a5a6" for i in x])
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Laplacian Eigenvalue Spectrum (Fiedler = green bar)")
    ax.axhline(y=eigenvalues[1], color="#e74c3c", linestyle="--", alpha=0.5, label=f"Fiedler = {eigenvalues[1]:.4f}")
    ax.legend()

    # Annotate spectral gap
    gap = eigenvalues[1] - eigenvalues[0]
    ax.annotate(f"Spectral gap = {gap:.4f}", xy=(1, eigenvalues[1]),
                xytext=(3, eigenvalues[1] + 0.1),
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_persistence(
    topo: TopologicalAnalysis,
    save_path: str = None,
):
    """Plot persistence diagrams and Betti number summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Persistence diagram
    ax = axes[0]
    colors = {0: "#3498db", 1: "#e74c3c", 2: "#2ecc71"}
    max_val = 0
    for dim in range(3):
        if dim in topo.persistence_diagrams and len(topo.persistence_diagrams[dim]) > 0:
            pts = topo.persistence_diagrams[dim]
            finite = pts[~np.isinf(pts[:, 1])]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], c=colors[dim],
                          label=f"H{dim}", alpha=0.7, s=30)
                max_val = max(max_val, finite.max())
            # Infinite features (actual topological features)
            infinite = pts[np.isinf(pts[:, 1])]
            if len(infinite) > 0:
                for pt in infinite:
                    ax.scatter(pt[0], max_val + 1, c=colors[dim],
                              marker="*", s=200, edgecolors="black", zorder=5)

    if max_val > 0:
        ax.plot([0, max_val + 1], [0, max_val + 1], "k--", alpha=0.3, label="diagonal")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram")
    ax.legend()

    # Betti number bar chart
    ax2 = axes[1]
    dims = [0, 1, 2]
    bettis = [topo.betti_numbers.get(d, 0) for d in dims]
    bars = ax2.bar([f"β{d}" for d in dims], bettis, color=[colors[d] for d in dims])
    ax2.set_ylabel("Betti Number")
    ax2.set_title("Betti Numbers (Topological Features)")

    for bar, val in zip(bars, bettis):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha="center", fontweight="bold", fontsize=14)

    # Interpretation
    interp = []
    if topo.betti_numbers.get(0, 0) > 0:
        interp.append(f"β₀={topo.betti_numbers[0]}: {topo.betti_numbers[0]} connected region(s)")
    if topo.betti_numbers.get(1, 0) > 0:
        interp.append(f"β₁={topo.betti_numbers[1]}: {topo.betti_numbers[1]} tunnel(s)/doorway(s)")
    if topo.betti_numbers.get(2, 0) > 0:
        interp.append(f"β₂={topo.betti_numbers[2]}: {topo.betti_numbers[2]} enclosed void(s)")

    if interp:
        ax2.text(0.5, -0.2, "\n".join(interp), transform=ax2.transAxes,
                ha="center", fontsize=10, style="italic",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fusion_comparison(
    fusion: FusionAnalysis,
    save_path: str = None,
):
    """Visualize multi-robot fusion results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart comparing conservations
    ax = axes[0]
    labels = ["Robot A", "Robot B", "Merged\n(consistent)"]
    values = [fusion.conservation_a, fusion.conservation_b, fusion.conservation_merged]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    if fusion.conservation_conflict is not None:
        labels.append("Merged\n(conflict)")
        values.append(fusion.conservation_conflict)
        colors.append("#f39c12")

    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_ylabel("Conservation Score")
    ax.set_title("Sheaf-Inspired Map Fusion Analysis")
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{val:.3f}", ha="center", fontweight="bold")

    # Threshold line
    max_ind = max(fusion.conservation_a, fusion.conservation_b)
    ax.axhline(y=max_ind, color="gray", linestyle="--", alpha=0.5,
              label=f"max(individual) = {max_ind:.3f}")
    ax.legend()

    # Interpretation panel
    ax2 = axes[1]
    ax2.axis("off")
    text_lines = [
        "Sheaf Cohomology Interpretation",
        "=" * 40,
        "",
        f"Overlap consistency (consistent): {fusion.overlap_consistency:.3f}",
        f"Overlaps agree: {'YES ✓ (H¹ = 0)' if fusion.consistent else 'NO ✗ (H¹ ≠ 0)'},",
        "",
    ]

    if fusion.conflict_detected:
        conflict_str = f"{fusion.conflict_overlap_consistency:.3f}" if fusion.conflict_overlap_consistency else "N/A"
        text_lines += [
            f"Overlap consistency (conflict):   {conflict_str}",
            "Conflict Detection: ✓ DETECTED",
            "  → H¹ ≠ 0: sheaf obstruction",
            "  → Maps disagree at same location",
        ]
    else:
        text_lines += [
            "Conflict Detection: None",
            "  → Consistent maps merge cleanly",
        ]

    text_lines += [
        "",
        "Physical Interpretation:",
        "  H⁰ = globally consistent map",
        "  H¹ ≠ 0 = localization/conflict",
        "  Overlap drop = sheaf obstruction",
    ]

    ax2.text(0.05, 0.95, "\n".join(text_lines), transform=ax2.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
