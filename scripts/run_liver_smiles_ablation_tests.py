#!/usr/bin/env python3
"""Run liver SMILES-drop vs zero-SMILES ablation benchmarks.

This mirrors the HNSC ablation that performed best:
  - baseline: existing slim input that drops invalid/no-SMILES drugs
  - ablation: keep all drugs, zero SMILES-derived numeric features for invalid
    SMILES rows, and retrain with the same 4-fold random/drug/scaffold policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_liver_pipeline as liver


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "20260422_liver_smiles_ablation_v1"

SMILES_DERIVED_PREFIXES = (
    "drug_morgan_",
    "drug_desc_",
    "smiles_svd_",
)
SMILES_DERIVED_EXACT = {
    "drug__smiles_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--stage", default="all", choices=["all", "build", "train", "report"])
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-slim-crispr", type=int, default=3000)
    parser.add_argument("--max-slim-lincs", type=int, default=768)
    parser.add_argument("--morgan-var-threshold", type=float, default=0.01)
    parser.add_argument("--max-morgan", type=int, default=1600)
    parser.add_argument(
        "--models",
        default="lightgbm,xgboost,extratrees,randomforest,residual_mlp,flat_mlp",
    )
    parser.add_argument("--tree-estimators", type=int, default=160)
    parser.add_argument("--mlp-epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.root / "data/processed/liver_smiles_ablation" / args.run_id
    report_dir = args.root / "outputs/reports" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "legacy_drop_smiles_baseline_4fold": run_dir / "legacy_drop_smiles_baseline_4fold",
        "legacy_all_drugs_zero_smiles": run_dir / "legacy_all_drugs_zero_smiles",
    }
    for path in variants.values():
        path.mkdir(parents=True, exist_ok=True)

    stages = expand_stage(args.stage)
    if "build" in stages:
        build_drop_smiles_baseline(args, variants["legacy_drop_smiles_baseline_4fold"])
        build_all_drugs_zero_smiles(args, variants["legacy_all_drugs_zero_smiles"])
    if "train" in stages:
        for name, path in variants.items():
            train_variant(name, path, args)
    if "report" in stages:
        write_report(args, run_dir, report_dir, variants)

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "run_dir": str(run_dir),
                "report": str(report_dir / "liver_smiles_ablation_summary.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def expand_stage(stage: str) -> list[str]:
    if stage == "all":
        return ["build", "train", "report"]
    if stage == "train":
        return ["train", "report"]
    if stage == "report":
        return ["report"]
    return [stage]


def build_drop_smiles_baseline(args: argparse.Namespace, out_dir: Path) -> None:
    source_path = args.root / "data/processed/slim_inputs/train_table.parquet"
    feature_path = args.root / "data/processed/slim_inputs/feature_names.json"
    train = pd.read_parquet(source_path).copy()
    feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
    train.to_parquet(out_dir / "train_table.parquet", index=False)
    write_json(out_dir / "feature_names.json", feature_names)
    write_json(
        out_dir / "input_summary.json",
        {
            "variant": "legacy_drop_smiles_baseline_4fold",
            "source": str(source_path),
            "shape": [int(train.shape[0]), int(train.shape[1])],
            "feature_count": int(len(feature_names)),
            "drugs": int(train["canonical_drug_id"].astype(str).nunique()),
            "valid_smiles_drugs": int(train["canonical_drug_id"].astype(str).nunique()),
            "valid_smiles_pairs": int(pd.to_numeric(train.get("drug_has_valid_smiles", 0), errors="coerce").fillna(0).sum()),
            "invalid_smiles_pairs_kept": 0,
            "counts": count_feature_groups(feature_names),
        },
    )


def build_all_drugs_zero_smiles(args: argparse.Namespace, out_dir: Path) -> None:
    source_path = args.root / "data/processed/model_inputs/train_table.parquet"
    train = pd.read_parquet(source_path).copy()
    valid_mask = pd.to_numeric(train.get("drug_has_valid_smiles", 0), errors="coerce").fillna(0).gt(0)

    smiles_cols = smiles_derived_columns(train)
    for col in smiles_cols:
        if col in train.columns and pd.api.types.is_numeric_dtype(train[col]):
            train.loc[~valid_mask, col] = 0.0

    numeric_cols = liver.numeric_feature_columns(train)
    crispr_cols = [c for c in numeric_cols if c.startswith("sample__crispr__")]
    lincs_cols = [c for c in numeric_cols if c.startswith("lincs__")]
    morgan_cols = [c for c in numeric_cols if c.startswith("drug_morgan_")]
    context_cols = [c for c in numeric_cols if c.startswith("ctxcat__")]
    smiles_svd_cols = [c for c in numeric_cols if c.startswith("smiles_svd_")]
    grouped = set(crispr_cols + lincs_cols + morgan_cols + context_cols + smiles_svd_cols)
    other_cols = [c for c in numeric_cols if c not in grouped]

    crispr_keep = liver.top_variance_columns(train, crispr_cols, args.max_slim_crispr)
    lincs_keep = liver.top_variance_columns(train, lincs_cols, args.max_slim_lincs)
    morgan_var = train[morgan_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).var(axis=0)
    morgan_keep = morgan_var[morgan_var >= args.morgan_var_threshold].index.tolist()
    if len(morgan_keep) > args.max_morgan:
        morgan_keep = liver.top_variance_columns(train, morgan_keep, args.max_morgan)

    feature_names = other_cols + context_cols + smiles_svd_cols + crispr_keep + lincs_keep + morgan_keep
    meta_cols = [
        "pair_id",
        "sample_id",
        "cell_line_name",
        "model_id",
        "canonical_drug_id",
        "DRUG_ID",
        "drug_name",
        "TCGA_DESC",
        "label_regression",
        "label_binary",
        "drug__canonical_smiles",
        "canonical_smiles",
        "drug__target_list",
    ]
    meta_cols = [c for c in meta_cols if c in train.columns]
    out = train[meta_cols + feature_names].copy()
    out[feature_names] = out[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out.to_parquet(out_dir / "train_table.parquet", index=False)
    write_json(out_dir / "feature_names.json", feature_names)
    write_json(
        out_dir / "input_summary.json",
        {
            "variant": "legacy_all_drugs_zero_smiles",
            "source": str(source_path),
            "shape": [int(out.shape[0]), int(out.shape[1])],
            "feature_count": int(len(feature_names)),
            "drugs": int(out["canonical_drug_id"].astype(str).nunique()),
            "valid_smiles_drugs": int(
                out.loc[pd.to_numeric(out.get("drug_has_valid_smiles", 0), errors="coerce").fillna(0).gt(0), "canonical_drug_id"]
                .astype(str)
                .nunique()
            ),
            "valid_smiles_pairs": int(valid_mask.sum()),
            "invalid_smiles_pairs_kept": int((~valid_mask).sum()),
            "smiles_zeroed_column_count": len(smiles_cols),
            "smiles_zeroed_column_examples": smiles_cols[:20],
            "counts": {
                "crispr": len(crispr_keep),
                "lincs": len(lincs_keep),
                "morgan": len(morgan_keep),
                "smiles_svd": len(smiles_svd_cols),
                "context": len(context_cols),
                "other": len(other_cols),
            },
        },
    )


def smiles_derived_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in SMILES_DERIVED_EXACT or any(col.startswith(prefix) for prefix in SMILES_DERIVED_PREFIXES):
            cols.append(col)
    return cols


def count_feature_groups(feature_names: list[str]) -> dict[str, int]:
    return {
        "crispr": len([c for c in feature_names if c.startswith("sample__crispr__")]),
        "lincs": len([c for c in feature_names if c.startswith("lincs__")]),
        "morgan": len([c for c in feature_names if c.startswith("drug_morgan_")]),
        "smiles_svd": len([c for c in feature_names if c.startswith("smiles_svd_")]),
        "context": len([c for c in feature_names if c.startswith("ctxcat__")]),
        "drug_desc": len([c for c in feature_names if c.startswith("drug_desc_")]),
    }


def train_variant(name: str, variant_dir: Path, args: argparse.Namespace) -> None:
    train = pd.read_parquet(variant_dir / "train_table.parquet")
    feature_names = json.loads((variant_dir / "feature_names.json").read_text(encoding="utf-8"))
    X = train[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = train["label_regression"].to_numpy(dtype=np.float32)
    drug_groups = train["canonical_drug_id"].astype(str).to_numpy()
    scaffold_groups = build_scaffold_groups(train, variant_dir)

    splits = {
        "random_4fold": make_splits(X, y, None, args.n_splits, args.random_state, "random"),
        "drug_group_4fold": make_splits(X, y, drug_groups, args.n_splits, args.random_state, "group"),
        "scaffold_group_4fold": make_splits(X, y, scaffold_groups, args.n_splits, args.random_state, "group"),
    }
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results: list[dict[str, Any]] = []
    oof_store: dict[tuple[str, str], np.ndarray] = {}
    step4_dir = variant_dir / "step4_benchmark"
    step4_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_indices in splits.items():
        for model_name in model_names:
            print(f"[{name}] Training {model_name} / {split_name}", flush=True)
            metrics, oof = fit_model_cv(model_name, X, y, split_indices, args)
            if metrics.get("status") == "skipped":
                results.append(
                    {
                        "variant": name,
                        "split": split_name,
                        "model": model_name,
                        "status": "skipped",
                        "reason": metrics.get("reason", ""),
                    }
                )
                continue
            oof_store[(split_name, model_name)] = oof
            save_oof(step4_dir, train, y, oof, split_name, model_name)
            results.append(
                {
                    "variant": name,
                    "split": split_name,
                    "model": model_name,
                    "status": "completed",
                    "spearman": metrics["spearman"],
                    "rmse": metrics["rmse"],
                    "fold_spearman_mean": metrics["fold_spearman_mean"],
                    "fold_spearman_std": metrics["fold_spearman_std"],
                    "folds": metrics["folds"],
                }
            )
        ensemble = build_ensemble(step4_dir, train, y, split_name, name, results, oof_store)
        if ensemble:
            results.append(ensemble)

    all_df = pd.DataFrame(results)
    completed = all_df[all_df["status"].eq("completed")].copy()
    if not completed.empty:
        completed = completed.sort_values(["split", "spearman"], ascending=[True, False])
    all_df.to_csv(step4_dir / "step4_benchmark_summary_all.csv", index=False)
    completed.to_csv(step4_dir / "step4_benchmark_summary_completed.csv", index=False)
    write_json(
        step4_dir / "step4_metrics_summary.json",
        {
            "variant": name,
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_splits": int(args.n_splits),
            "models": model_names,
            "feature_names": feature_names,
            "results": results,
        },
    )


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    random_state: int,
    mode: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if mode == "group":
        if groups is None:
            raise ValueError("groups required for group split")
        n = min(n_splits, len(np.unique(groups)))
        return list(GroupKFold(n_splits=n).split(X, y, groups=groups))
    n = min(n_splits, len(y))
    return list(KFold(n_splits=n, shuffle=True, random_state=random_state).split(X, y))


def build_scaffold_groups(train: pd.DataFrame, variant_dir: Path) -> np.ndarray:
    smiles_col = "canonical_smiles" if "canonical_smiles" in train.columns else "drug__canonical_smiles"
    rows = []
    for _, row in train[["canonical_drug_id", smiles_col]].drop_duplicates("canonical_drug_id").iterrows():
        drug_id = str(row["canonical_drug_id"])
        smiles = str(row.get(smiles_col, "") or "")
        scaffold = compute_scaffold(smiles)
        rows.append(
            {
                "canonical_drug_id": drug_id,
                "canonical_smiles": smiles,
                "scaffold": scaffold or f"NO_SCAFFOLD_{drug_id}",
            }
        )
    scaffolds = pd.DataFrame(rows)
    scaffolds.to_parquet(variant_dir / "drug_scaffolds.parquet", index=False)
    lookup = scaffolds.set_index("canonical_drug_id")["scaffold"].to_dict()
    return train["canonical_drug_id"].astype(str).map(lookup).fillna("NO_SCAFFOLD_UNKNOWN").to_numpy()


def compute_scaffold(smiles: str) -> str:
    mol = liver.mol_from_smiles_or_none(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return ""


def fit_model_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    oof = np.zeros(len(y), dtype=np.float32)
    folds = []
    for fold_idx, (tr, va) in enumerate(split_indices, start=1):
        try:
            preds = fit_one_model(model_name, X[tr], y[tr], X[va], args.random_state + fold_idx, args)
        except ModuleNotFoundError as exc:
            return {"status": "skipped", "reason": str(exc)}, oof
        oof[va] = preds.astype(np.float32)
        folds.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "spearman": metric_spearman(y[va], preds),
                "rmse": metric_rmse(y[va], preds),
            }
        )
    fold_spearman = [f["spearman"] for f in folds if np.isfinite(f["spearman"])]
    metrics = {
        "status": "completed",
        "spearman": metric_spearman(y, oof),
        "rmse": metric_rmse(y, oof),
        "fold_spearman_mean": float(np.mean(fold_spearman)) if fold_spearman else float("nan"),
        "fold_spearman_std": float(np.std(fold_spearman)) if fold_spearman else float("nan"),
        "folds": folds,
    }
    return metrics, oof


def fit_one_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if model_name == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=args.tree_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_valid).astype(np.float32)

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_estimators=args.tree_estimators,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        return model.predict(X_valid).astype(np.float32)

    if model_name == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=args.tree_estimators,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_valid).astype(np.float32)

    if model_name == "randomforest":
        model = RandomForestRegressor(
            n_estimators=max(80, args.tree_estimators // 2),
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_valid).astype(np.float32)

    if model_name == "residual_mlp":
        return fit_mlp(X_train, y_train, X_valid, seed, args.mlp_epochs, residual=True)

    if model_name == "flat_mlp":
        return fit_mlp(X_train, y_train, X_valid, seed, args.mlp_epochs, residual=False)

    raise ValueError(f"Unsupported model: {model_name}")


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    seed: int,
    epochs: int,
    residual: bool,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    X_train_s, X_valid_s = liver.scale_by_train(X_train, X_valid)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    y_train_s = ((y_train - y_mean) / y_std).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    model: nn.Module
    if residual:
        model = liver.ResidualMLP(X_train.shape[1], hidden_dim=128, num_blocks=3, dropout=0.1)
    else:
        model = FlatMLP(X_train.shape[1], hidden_dim=192, dropout=0.15)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_valid_s).to(device)).squeeze(1).cpu().numpy()
    return (preds * y_std + y_mean).astype(np.float32)


class FlatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_oof(step4_dir: Path, train: pd.DataFrame, y: np.ndarray, oof: np.ndarray, split: str, model: str) -> None:
    cols = [c for c in ["pair_id", "sample_id", "canonical_drug_id", "drug_name"] if c in train.columns]
    out = train[cols].copy()
    out["y_true"] = y
    out["y_pred"] = oof
    out["split"] = split
    out["model"] = model
    out.to_parquet(step4_dir / f"oof_{split}_{model}.parquet", index=False)


def build_ensemble(
    step4_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    split_name: str,
    variant: str,
    results: list[dict[str, Any]],
    oof_store: dict[tuple[str, str], np.ndarray],
) -> dict[str, Any] | None:
    candidates = [
        r
        for r in results
        if r.get("variant") == variant
        and r.get("split") == split_name
        and r.get("status") == "completed"
        and r.get("model") != "weighted_top3_ensemble"
    ]
    candidates = sorted(candidates, key=lambda r: float(r.get("spearman", -np.inf)), reverse=True)[:3]
    if not candidates:
        return None
    raw_weights = np.array([max(float(r.get("spearman", 0.0)), 0.0) for r in candidates], dtype=np.float32)
    if float(raw_weights.sum()) <= 1e-8:
        raw_weights = np.ones(len(candidates), dtype=np.float32)
    weights = raw_weights / raw_weights.sum()
    pred = np.zeros(len(y), dtype=np.float32)
    members = []
    for result, weight in zip(candidates, weights):
        model = str(result["model"])
        pred += weight * oof_store[(split_name, model)]
        members.append({"model": model, "spearman": float(result["spearman"]), "weight": float(weight)})
    save_oof(step4_dir, train, y, pred, split_name, "weighted_top3_ensemble")
    return {
        "variant": variant,
        "split": split_name,
        "model": "weighted_top3_ensemble",
        "status": "completed",
        "spearman": metric_spearman(y, pred),
        "rmse": metric_rmse(y, pred),
        "fold_spearman_mean": np.nan,
        "fold_spearman_std": np.nan,
        "folds": [],
        "members": members,
    }


def write_report(
    args: argparse.Namespace,
    run_dir: Path,
    report_dir: Path,
    variants: dict[str, Path],
) -> None:
    all_completed = []
    sections = [
        "# Liver SMILES ablation tests\n",
        f"- Run ID: `{args.run_id}`",
        f"- Fold policy: {args.n_splits}-fold random/drug/scaffold",
    ]
    for name, path in variants.items():
        summary = json.loads((path / "input_summary.json").read_text(encoding="utf-8"))
        sections.append(f"\n## {name}")
        sections.append(f"- Input shape: {summary.get('shape')}")
        sections.append(f"- Feature count: {summary.get('feature_count')}")
        sections.append(f"- Drugs: {summary.get('drugs')}")
        sections.append(f"- Valid SMILES drugs: {summary.get('valid_smiles_drugs')}")
        sections.append(f"- Invalid SMILES pairs kept: {summary.get('invalid_smiles_pairs_kept')}")
        completed_path = path / "step4_benchmark/step4_benchmark_summary_completed.csv"
        if completed_path.exists():
            completed = pd.read_csv(completed_path)
            all_completed.append(completed)
            for split in sorted(completed["split"].dropna().unique()):
                top = completed[completed["split"].eq(split)].sort_values("spearman", ascending=False).head(5)
                sections.append(f"\n### {name} / {split}")
                for _, row in top.iterrows():
                    sections.append(f"- {row['model']}: Spearman {float(row['spearman']):.4f}, RMSE {float(row['rmse']):.4f}")

    if all_completed:
        combined = pd.concat(all_completed, ignore_index=True)
        combined.to_csv(report_dir / "liver_smiles_ablation_completed.csv", index=False)
        best = combined.sort_values(["split", "spearman"], ascending=[True, False]).groupby("split").head(6)
        best.to_csv(report_dir / "liver_smiles_ablation_best_by_split.csv", index=False)

    report = "\n".join(sections) + "\n"
    (report_dir / "liver_smiles_ablation_summary.md").write_text(report, encoding="utf-8")
    write_json(
        report_dir / "liver_smiles_ablation_index.json",
        {
            "run_id": args.run_id,
            "run_dir": str(run_dir),
            "variants": {name: str(path) for name, path in variants.items()},
            "report": str(report_dir / "liver_smiles_ablation_summary.md"),
        },
    )


def metric_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    value = spearmanr(y_true, y_pred).correlation
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
