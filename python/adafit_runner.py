#!/usr/bin/env python3
"""Bridge script to run AdaFit normals on a point cloud.

Strict mode: official AdaFit repository inference only (Runsong123/AdaFit).
Any failure raises an error immediately (no fallback).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_xyz(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}, got shape {points.shape}")
    return points


def save_xyz(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%.9f")


def resolve_default_checkpoint(repo: Path) -> Path:
    return repo / "trained_model" / "AdaFit" / "my_experiment_model_599.pth"


def resolve_default_params(checkpoint: Path) -> Path:
    stem = checkpoint.stem
    if "_model_" in stem:
        prefix = stem.split("_model_")[0]
    elif stem.endswith("_model"):
        prefix = stem[: -len("_model")]
    else:
        prefix = "my_experiment"
    return checkpoint.with_name(f"{prefix}_params.pth")


def _build_patch(points: np.ndarray, tree, center_idx: int, points_per_patch: int, use_pca: bool) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree  # type: ignore

    if not isinstance(tree, cKDTree):
        raise TypeError("tree must be a scipy.spatial.cKDTree")

    n_points = points.shape[0]
    k_query = min(points_per_patch, n_points)
    dists, inds = tree.query(points[center_idx], k=k_query)
    dists = np.atleast_1d(dists)
    inds = np.atleast_1d(inds)

    if inds.size == 0:
        inds = np.array([center_idx], dtype=np.int64)
        dists = np.array([0.0], dtype=np.float64)

    rad = float(np.max(dists))
    if rad < 1e-12:
        rad = 1.0

    patch = np.zeros((points_per_patch, 3), dtype=np.float32)
    count = int(min(points_per_patch, inds.size))
    patch[:count] = points[inds[:count]].astype(np.float32)

    if count < points_per_patch:
        fill = np.random.choice(inds[:count], size=points_per_patch - count, replace=True)
        patch[count:] = points[fill].astype(np.float32)
        count = points_per_patch

    center = points[center_idx].astype(np.float32)
    patch[:count] = (patch[:count] - center) / np.float32(rad)

    trans = np.eye(3, dtype=np.float32)
    if use_pca:
        valid = patch[:count]
        pts_mean = valid.mean(axis=0, keepdims=True)
        valid = valid - pts_mean
        # torch.svd in the original code corresponds to full SVD on transposed patch
        u, _, _ = np.linalg.svd(valid.T, full_matrices=True)
        trans = u.astype(np.float32)
        valid = valid @ trans
        cp_new = (-pts_mean.squeeze(0)) @ trans
        valid = valid - cp_new
        patch[:count] = valid

    return patch, trans


def run_official_adafit(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    try:
        import torch
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError(
            "Official AdaFit mode requires torch and scipy. Install dependencies from python/requirements-adafit.txt"
        ) from exc

    if not args.repo:
        raise RuntimeError("Official AdaFit mode requires --repo")

    repo = Path(args.repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"AdaFit repo path does not exist: {repo}")

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else resolve_default_checkpoint(repo)
    params_path = Path(args.params).resolve() if args.params else resolve_default_params(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"AdaFit checkpoint not found: {checkpoint}")
    if not params_path.exists():
        raise FileNotFoundError(f"AdaFit params file not found: {params_path}")

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "models"))
    sys.path.insert(0, str(repo / "utils"))

    trainopt = torch.load(params_path, map_location="cpu", weights_only=False)
    points_per_patch = int(getattr(trainopt, "points_per_patch", max(128, args.k)))
    use_pca = bool(getattr(trainopt, "use_pca", True))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    def _instantiate(scale: str):
        if scale == "single":
            import AdaFit_single_scale as adafit_module
        else:
            import AdaFit_multi_scale as adafit_module
        return adafit_module.DeepFit(
            1,
            num_points=points_per_patch,
            use_point_stn=bool(getattr(trainopt, "use_point_stn", True)),
            use_feat_stn=bool(getattr(trainopt, "use_feat_stn", True)),
            point_tuple=int(getattr(trainopt, "point_tuple", 1)),
            sym_op=str(getattr(trainopt, "sym_op", "max")),
            arch=str(getattr(trainopt, "arch", "simple")),
            n_gaussians=int(getattr(trainopt, "n_gaussians", 1)),
            jet_order=int(getattr(trainopt, "jet_order", 3)),
            weight_mode=str(getattr(trainopt, "weight_mode", "sigmoid")),
        )

    candidates = [args.model_scale] if args.model_scale != "auto" else ["multi", "single"]
    model = None
    load_errors: list[str] = []
    for scale in candidates:
        try:
            candidate = _instantiate(scale)
            candidate.load_state_dict(state)
            model = candidate
            break
        except Exception as exc:
            load_errors.append(f"{scale}: {exc}")
    if model is None:
        raise RuntimeError("Could not match checkpoint with AdaFit single/multi model variant. " + " | ".join(load_errors))

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    tree = cKDTree(points)
    batch_size = max(1, int(args.batch_size))
    all_normals: list[np.ndarray] = []

    for start in range(0, points.shape[0], batch_size):
        end = min(points.shape[0], start + batch_size)
        patches = []
        transforms = []
        for idx in range(start, end):
            patch, trans = _build_patch(points, tree, idx, points_per_patch, use_pca)
            patches.append(patch)
            transforms.append(trans)

        patch_tensor = torch.from_numpy(np.stack(patches, axis=0)).to(device)
        patch_tensor = patch_tensor.transpose(2, 1)
        data_trans = torch.from_numpy(np.stack(transforms, axis=0)).to(device)

        with torch.no_grad():
            n_est, _, _, trans, _, _, _ = model(patch_tensor)
            if bool(getattr(trainopt, "use_point_stn", True)):
                n_est = torch.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)
            if use_pca:
                n_est = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

        n_np = n_est.detach().cpu().numpy()
        n_norm = np.linalg.norm(n_np, axis=1, keepdims=True)
        n_np = n_np / np.clip(n_norm, 1e-12, None)
        all_normals.append(n_np)

    return np.concatenate(all_normals, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AdaFit normals through a thin bridge script.")
    parser.add_argument("--input", required=True, help="Input XYZ point cloud file")
    parser.add_argument("--output", required=True, help="Output XYZ normals file")
    parser.add_argument("--repo", default="", help="Path to a local AdaFit repository checkout")
    parser.add_argument("--checkpoint", default="", help="Path to a trained AdaFit checkpoint")
    parser.add_argument("--params", default="", help="Path to the AdaFit *_params.pth file (optional, inferred from checkpoint if omitted)")
    parser.add_argument("--model-scale", default="auto", choices=["auto", "single", "multi"], help="Select official AdaFit model variant")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used for official AdaFit inference")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device for official AdaFit mode")
    parser.add_argument("--k", type=int, default=64, help="Neighborhood size used when deriving local patches for AdaFit preprocessing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    points = load_xyz(input_path)

    normals = run_official_adafit(points, args)
    if normals.shape != points.shape:
        raise ValueError(
            f"AdaFit output shape mismatch: expected {points.shape}, got {normals.shape}"
        )
    save_xyz(output_path, normals)
    print(f"[AdaFit bridge] Wrote {normals.shape[0]} normals with official AdaFit mode")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI-friendly error path
        print(f"[AdaFit bridge] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)




