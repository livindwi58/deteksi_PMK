"""Plot dataset scatter and save figure.

Usage examples:
  python utils/plot_dataset.py                     # uses features/dataset.csv -> results/dataset_scatter.png
  python utils/plot_dataset.py --input features/data_train.csv --output results/train_scatter.png --show

The script will attempt to detect a label column (one of common names) and color points by label.
If the dataset has more than 3 numeric features, PCA->2D is applied before plotting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


LABEL_CANDIDATES = [
    "label_name",
    "label",
    "class",
    "target",
    "y",
    "prediction",
    "category",
    "label_encoded",
]


def detect_label_column(df: pd.DataFrame) -> str | None:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    # also if there is a non-numeric column with few unique values, consider it label
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].nunique() <= 50:
                return col
    return None


def numeric_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    num = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in num if c not in exclude]


def plot_2d_scatter(X: np.ndarray, labels: pd.Series | None, out_path: Path, title: str | None = None, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.7)
    else:
        uniq = pd.Series(labels).unique()
        for u in uniq:
            mask = labels == u
            ax.scatter(X[mask, 0], X[mask, 1], s=18, alpha=0.8, label=str(u))
        ax.legend(markerscale=2, fontsize="small", title="label")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_bar(labels: pd.Series, out_path: Path, title: str | None = None, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    s = pd.Series(labels).value_counts().sort_index()
    s.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot dataset scatter (PCA when needed) and save image")
    p.add_argument("--input", "-i", default="features/dataset.csv", help="CSV input path")
    p.add_argument("--output", "-o", default="results/dataset_scatter.png", help="Output image path")
    p.add_argument("--sample", "-s", type=int, default=0, help="Max sample rows (0 = all)")
    p.add_argument("--show", action="store_true", help="Show figure after creating")
    p.add_argument("--force-pca", action="store_true", help="Force PCA to 2D before plotting")
    p.add_argument("--plot-type", choices=["scatter", "bar"], default="scatter", help="Type of plot to create")
    p.add_argument("--label-column", "-l", default=None, help="Specify column to use as label (overrides auto-detect)")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return 2

    df = pd.read_csv(in_path)
    if args.sample and args.sample > 0 and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=0)

    # determine label column: CLI override > explicit 'label_name' > auto-detect
    label_col = None
    if args.label_column:
        if args.label_column in df.columns:
            label_col = args.label_column
        else:
            print(f"Requested label column '{args.label_column}' not found in CSV columns.")
    if label_col is None and "label_name" in df.columns:
        label_col = "label_name"
    if label_col is None:
        label_col = detect_label_column(df)

    labels = None
    if label_col is not None:
        labels = df[label_col]

    # If user requested a bar chart, require a label column
    if args.plot_type == "bar":
        if label_col is None:
            print("No label column detected; bar plot requires a label column.")
            return 4
        title = f"Class distribution of '{label_col}'"
        plot_bar(labels, out_path, title=title, show=args.show)
        print(f"Saved bar chart to {out_path}")
        return 0

    # Otherwise, produce a scatter (PCA if needed)
    num_cols = numeric_feature_columns(df, exclude=[label_col] if label_col else None)

    if len(num_cols) == 0:
        print("No numeric feature columns found to plot.")
        return 3

    # Choose plotting path
    if not args.force_pca and len(num_cols) == 2:
        X = df[num_cols].to_numpy()
        title = f"Scatter: {num_cols[0]} vs {num_cols[1]}"
        plot_2d_scatter(X, labels, out_path, title=title, show=args.show)
        print(f"Saved scatter to {out_path}")
        return 0

    # If more than 2 numeric features, project to 2 with PCA for a nice 2D view
    X = df[num_cols].to_numpy()
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    title = f"PCA 2D of {len(num_cols)} numeric features"
    plot_2d_scatter(X2, labels, out_path, title=title, show=args.show)
    print(f"Saved PCA scatter to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
