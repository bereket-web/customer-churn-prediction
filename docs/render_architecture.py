"""Render a compact architecture diagram for the Project 12 README."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent


def add_box(ax, center_x: float, center_y: float, width: float, height: float, title: str, body: str, color: str) -> None:
    box = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=1.4,
        edgecolor="#34495e",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(center_x, center_y + 0.38, title, ha="center", va="center", fontsize=11, weight="bold", color="#1d2939")
    ax.text(center_x, center_y - 0.20, body, ha="center", va="center", fontsize=9.2, linespacing=1.35, color="#344054")


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#475467") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 6.4), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(8, 6.05, "Customer Churn Prediction — System Architecture", ha="center", fontsize=17, weight="bold", color="#101828")
    ax.text(8, 5.68, "Provenance-aware data  →  leakage-safe evaluation  →  local inference demo", ha="center", fontsize=10.5, color="#475467")

    centers = [1.6, 4.75, 7.9, 11.05, 14.2]
    y, width, height = 4.15, 2.55, 1.65
    boxes = [
        ("Public source", "UCI Iranian Churn\n3,150 rows · CC BY 4.0", "#e8eef8"),
        ("Validate & audit", "Drop ambiguous Status\nGroup repeated profiles", "#e8eef8"),
        ("Features & models", "Usage ratios + fold-fitted\npreprocessing; 5 candidates", "#e2f1ed"),
        ("Evaluate honestly", "Grouped 5-fold CV\nOOF threshold + held-out test", "#fff1cc"),
        ("API & interactive demo", "FastAPI /predict + /explain\nStreamlit local dashboard", "#eee8f7"),
    ]
    for x, (title, body, color) in zip(centers, boxes, strict=True):
        add_box(ax, x, y, width, height, title, body, color)
    for index in range(len(centers) - 1):
        arrow(ax, (centers[index] + width / 2 + 0.04, y), (centers[index + 1] - width / 2 - 0.04, y))

    artifact_centers = [4.2, 8, 11.8]
    artifact_boxes = [
        ("Model comparison", "CV average precision,\nthresholded test metrics", "#f2f4f7"),
        ("Error diagnostics", "Precision–recall curves +\nconfusion matrix", "#f2f4f7"),
        ("Explainability", "Permutation importance +\nlocal reference perturbations", "#f2f4f7"),
    ]
    for x, (title, body, color) in zip(artifact_centers, artifact_boxes, strict=True):
        add_box(ax, x, 1.65, 3.0, 1.15, title, body, color)
    arrow(ax, (centers[3], y - height / 2 - 0.04), (artifact_centers[0], 2.27))
    arrow(ax, (centers[3], y - height / 2 - 0.04), (artifact_centers[1], 2.27))
    arrow(ax, (centers[3], y - height / 2 - 0.04), (artifact_centers[2], 2.27))
    ax.text(8, 0.55, "`Status` withheld as a possible outcome proxy · duplicate input profiles stay inside one split · no production deployment claimed", ha="center", fontsize=9.2, color="#667085")

    fig.savefig(ROOT / "architecture.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(ROOT / "architecture.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    main()
