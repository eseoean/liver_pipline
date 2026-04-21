#!/usr/bin/env python3
"""End-to-end LIHC/HCC drug-repurposing pipeline.

The script intentionally keeps the workflow compact and reproducible:
S3 raw/staged sources -> standardized tables -> slim model input -> fast
GroupCV/random models -> ensemble ranking -> ADMET gate -> report.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski, rdMolDescriptors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_S3_ROOT = "s3://say2-4team/Liver_raw"
DEFAULT_UPLOAD_PREFIX = "s3://say2-4team/20260409_eseo/20260421_liver"

KEY_COLUMNS = {
    "pair_id",
    "sample_id",
    "cell_line_name",
    "model_id",
    "canonical_drug_id",
    "DRUG_ID",
    "drug_name",
    "label_regression",
    "label_binary",
    "label_main",
    "label_aux",
    "binary_threshold",
    "label_main_type",
    "label_aux_type",
    "ln_IC50",
    "gdsc_version",
    "TCGA_DESC",
    "cohort",
    "PATHWAY_NAME_NORMALIZED",
    "classification",
    "drug_bridge_strength",
    "stage3_resolution_status",
    "canonical_smiles",
    "drug__canonical_smiles",
    "drug__target_list",
}

STRONG_CONTEXT_COLS = [
    "TCGA_DESC",
    "PATHWAY_NAME_NORMALIZED",
    "classification",
    "drug_bridge_strength",
    "stage3_resolution_status",
]

KNOWN_LIVER_DRUGS = {
    "sorafenib",
    "lenvatinib",
    "regorafenib",
    "cabozantinib",
    "ramucirumab",
    "nivolumab",
    "pembrolizumab",
    "atezolizumab",
    "bevacizumab",
    "durvalumab",
    "tremelimumab",
}

ADMET_ASSAYS = {
    "ames": {"type": "binary", "good": 0, "hard_bad": 1},
    "dili": {"type": "binary", "good": 0, "hard_bad": 1},
    "herg": {"type": "binary", "good": 0, "hard_bad": 1},
    "hia_hou": {"type": "binary", "good": 1},
    "bioavailability_ma": {"type": "binary", "good": 1},
    "pgp_broccatelli": {"type": "binary", "good": 0},
    "caco2_wang": {"type": "regression", "good_direction": "high", "threshold": -5.15},
    "half_life_obach": {"type": "regression", "good_direction": "high", "threshold": 3.0},
    "solubility_aqsoldb": {"type": "regression", "good_direction": "high"},
    "lipophilicity_astrazeneca": {"type": "regression", "ideal_range": (-0.4, 5.6)},
}


@dataclass
class PipelinePaths:
    root: Path
    raw_cache: Path
    processed: Path
    standardized: Path
    disease_context: Path
    model_inputs: Path
    slim_inputs: Path
    model_runs: Path
    final_selection: Path
    reports: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", "download", "inputs", "train", "rank", "report", "upload"])
    parser.add_argument("--s3-root", default=DEFAULT_S3_ROOT)
    parser.add_argument("--upload-prefix", default=DEFAULT_UPLOAD_PREFIX)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", default="lightgbm,xgboost,residual_mlp,tabnet")
    parser.add_argument("--max-crispr-features", type=int, default=3500)
    parser.add_argument("--max-lincs-features", type=int, default=768)
    parser.add_argument("--max-slim-crispr", type=int, default=3000)
    parser.add_argument("--max-slim-lincs", type=int, default=768)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--upload-s3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = make_paths(args.root)
    ensure_dirs(paths)
    stages = expand_stages(args.stage)

    if "download" in stages and not args.skip_download:
        download_sources(paths, args.s3_root)
    if "inputs" in stages:
        build_inputs(paths, args)
    if "train" in stages:
        train_models(paths, args)
    if "rank" in stages:
        rank_and_filter(paths, args)
    if "report" in stages:
        render_final_report(paths, args)
    if args.upload_s3 or "upload" in stages:
        upload_outputs(paths, args.upload_prefix)

    print(json.dumps(build_run_index(paths), indent=2, ensure_ascii=False))


def expand_stages(stage: str) -> list[str]:
    if stage == "all":
        return ["download", "inputs", "train", "rank", "report"]
    if stage == "report":
        return ["report"]
    return [stage]


def make_paths(root: Path) -> PipelinePaths:
    return PipelinePaths(
        root=root,
        raw_cache=root / "data/raw_cache",
        processed=root / "data/processed",
        standardized=root / "data/processed/standardized",
        disease_context=root / "data/processed/disease_context",
        model_inputs=root / "data/processed/model_inputs",
        slim_inputs=root / "data/processed/slim_inputs",
        model_runs=root / "outputs/model_runs",
        final_selection=root / "outputs/final_selection",
        reports=root / "outputs/reports",
    )


def ensure_dirs(paths: PipelinePaths) -> None:
    for value in paths.__dict__.values():
        if isinstance(value, Path):
            value.mkdir(parents=True, exist_ok=True)


def download_sources(paths: PipelinePaths, s3_root: str) -> None:
    files = {
        "gdsc_ic50.parquet": f"{s3_root}/gdsc_ic50.parquet",
        "drug_features_catalog.parquet": f"{s3_root}/drug_features_catalog.parquet",
        "drug_target_mapping.parquet": f"{s3_root}/drug_target_mapping.parquet",
        "lincs_drug_signature_normalized.parquet": f"{s3_root}/lincs_drug_signature_normalized.parquet",
        "cellline_cohort_from_depmap_model.csv": f"{s3_root}/raw_meta/cellline_cohort_from_depmap_model.csv",
        "cellline_tcga_mapping_from_raw.csv": f"{s3_root}/raw_meta/cellline_tcga_mapping_from_raw.csv",
        "Model.csv": f"{s3_root}/depmap/Model.csv",
        "Gene.csv": f"{s3_root}/depmap/Gene.csv",
        "CRISPRGeneEffect.csv": f"{s3_root}/depmap/CRISPRGeneEffect.csv",
        "screened_compounds_rel_8.5.csv": f"{s3_root}/GDSC/screened_compounds_rel_8.5.csv",
        "TCGA.LIHC.sampleMap_HiSeqV2.gz": f"{s3_root}/tcga/xena_tcgahub/TCGA.LIHC.sampleMap_HiSeqV2.gz",
        "TCGA.LIHC.sampleMap_LIHC_clinicalMatrix.tsv": (
            f"{s3_root}/tcga/xena_tcgahub/TCGA.LIHC.sampleMap_LIHC_clinicalMatrix.tsv"
        ),
        "clinicaltrials_liver_cancer_summary.json": (
            f"{s3_root}/additional_sources/clinicaltrials/clinicaltrials_liver_cancer_summary.json"
        ),
    }
    for local_name, remote in files.items():
        local = paths.raw_cache / local_name
        if not local.exists():
            run(["aws", "s3", "cp", remote, str(local)])

    admet_dir = paths.raw_cache / "admet"
    if not (admet_dir / "tdc_admet_group/admet_group").exists():
        run(["aws", "s3", "sync", f"{s3_root}/admet/", str(admet_dir)])


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_inputs(paths: PipelinePaths, args: argparse.Namespace) -> None:
    gdsc = load_gdsc(paths)
    cell_master = build_cell_line_master(paths, gdsc)
    response_labels = build_response_labels(gdsc, cell_master)
    drug_master = build_drug_master(paths, response_labels)
    target_mapping = load_target_mapping(paths, drug_master)
    sample_crispr = build_sample_crispr(paths, cell_master, args.max_crispr_features)
    disease_signature = build_disease_context(paths)
    sample_features = build_sample_features(cell_master, sample_crispr)
    drug_features, lincs_full = build_drug_features(paths, drug_master, target_mapping, args.max_lincs_features)
    pair_features = build_pair_features(response_labels, target_mapping, disease_signature, lincs_full)
    train_table = assemble_train_table(response_labels, sample_features, drug_features, pair_features)
    train_table, context_smiles_summary = add_strong_context_and_smiles_features(train_table, target_mapping)
    slim_table, feature_names, slim_summary = build_slim_table(train_table, args)

    gdsc.to_parquet(paths.standardized / "gdsc_lihc_response.parquet", index=False)
    cell_master.to_parquet(paths.standardized / "cell_line_master.parquet", index=False)
    response_labels.to_parquet(paths.standardized / "response_labels.parquet", index=False)
    drug_master.to_parquet(paths.standardized / "drug_master.parquet", index=False)
    target_mapping.to_parquet(paths.standardized / "drug_target_mapping.parquet", index=False)
    sample_crispr.to_parquet(paths.standardized / "sample_crispr_wide.parquet", index=False)
    disease_signature.to_parquet(paths.disease_context / "lihc_signature.parquet", index=False)
    sample_features.to_parquet(paths.model_inputs / "sample_features.parquet", index=False)
    drug_features.to_parquet(paths.model_inputs / "drug_features.parquet", index=False)
    pair_features.to_parquet(paths.model_inputs / "pair_features.parquet", index=False)
    train_table.to_parquet(paths.model_inputs / "train_table.parquet", index=False)
    response_labels.to_parquet(paths.model_inputs / "labels_y.parquet", index=False)
    slim_table.to_parquet(paths.slim_inputs / "train_table.parquet", index=False)
    slim_table[["pair_id", "sample_id", "canonical_drug_id", "drug_name"]].to_parquet(
        paths.slim_inputs / "keys.parquet", index=False
    )
    np.save(paths.slim_inputs / "X.npy", slim_table[feature_names].to_numpy(dtype=np.float32))
    np.save(paths.slim_inputs / "y.npy", slim_table["label_regression"].to_numpy(dtype=np.float32))
    write_json(paths.slim_inputs / "feature_names.json", feature_names)

    source_summary = {
        "gdsc_rows": int(gdsc.shape[0]),
        "gdsc_cell_lines": int(gdsc["cell_line_name"].nunique()),
        "gdsc_drugs": int(gdsc["DRUG_ID"].nunique()),
        "depmap_mapped_cell_lines": int(cell_master["model_id"].astype(str).ne("").sum()),
        "drug_master_rows": int(drug_master.shape[0]),
        "drug_master_has_smiles": int(drug_master["has_smiles"].sum()),
        "target_mapping_rows": int(target_mapping.shape[0]),
        "sample_crispr_features": int(len([c for c in sample_crispr.columns if c.startswith("sample__crispr__")])),
        "lincs_drug_features": int(len([c for c in drug_features.columns if c.startswith("lincs__")])),
        "train_shape": [int(train_table.shape[0]), int(train_table.shape[1])],
        "slim_shape": [int(slim_table.shape[0]), int(slim_table.shape[1])],
        "slim_feature_count": int(len(feature_names)),
        "strong_context_onehot_features": int(context_smiles_summary["strong_context"]["onehot_dim_for_ml"]),
        "smiles_svd_features": int(context_smiles_summary["smiles"]["svd_dim"]),
        "context_smiles": context_smiles_summary,
        "slim_summary": slim_summary,
    }
    write_json(paths.standardized / "source_crosswalks.json", source_summary)
    write_json(paths.model_inputs / "model_input_summary.json", source_summary)
    write_json(paths.slim_inputs / "slim_input_summary.json", source_summary)
    write_json(paths.slim_inputs / "context_smiles_bundle_summary.json", context_smiles_summary)


def load_gdsc(paths: PipelinePaths) -> pd.DataFrame:
    gdsc = pd.read_parquet(paths.raw_cache / "gdsc_ic50.parquet").copy()
    gdsc["cell_line_name"] = gdsc["cell_line_name"].astype(str).str.strip()
    gdsc["DRUG_ID"] = pd.to_numeric(gdsc["DRUG_ID"], errors="coerce").astype("Int64")
    gdsc["drug_name"] = gdsc["drug_name"].astype(str).str.strip()
    gdsc["ln_IC50"] = pd.to_numeric(gdsc["ln_IC50"], errors="coerce")
    gdsc = gdsc.dropna(subset=["DRUG_ID", "ln_IC50"]).copy()
    gdsc["DRUG_ID"] = gdsc["DRUG_ID"].astype(int)
    gdsc["TCGA_DESC"] = "LIHC"
    gdsc["gdsc_version"] = "GDSC2"
    gdsc["WEBRELEASE"] = "Y"

    compounds_path = paths.raw_cache / "screened_compounds_rel_8.5.csv"
    if compounds_path.exists():
        compounds = pd.read_csv(compounds_path)
        compounds["DRUG_ID"] = pd.to_numeric(compounds["DRUG_ID"], errors="coerce").astype("Int64")
        compounds = compounds.dropna(subset=["DRUG_ID"]).copy()
        compounds["DRUG_ID"] = compounds["DRUG_ID"].astype(int)
        compounds = compounds.rename(
            columns={"TARGET": "putative_target", "TARGET_PATHWAY": "pathway_name", "SCREENING_SITE": "screening_site"}
        )
        gdsc = gdsc.merge(
            compounds[["DRUG_ID", "putative_target", "pathway_name", "screening_site"]].drop_duplicates("DRUG_ID"),
            on="DRUG_ID",
            how="left",
        )
    for column in ["putative_target", "pathway_name", "screening_site"]:
        if column not in gdsc.columns:
            gdsc[column] = ""
        gdsc[column] = gdsc[column].fillna("").astype(str)
    return gdsc.reset_index(drop=True)


def build_cell_line_master(paths: PipelinePaths, gdsc: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_csv(paths.raw_cache / "cellline_cohort_from_depmap_model.csv")
    cohort["cell_line_name"] = cohort["cell_line_name"].astype(str).str.strip()
    cohort = cohort.rename(
        columns={
            "ModelID": "model_id",
            "OncotreeCode": "depmap_oncotree_code",
            "OncotreeSubtype": "depmap_oncotree_subtype",
            "cohort": "depmap_cohort",
        }
    )
    base = gdsc[["cell_line_name"]].drop_duplicates().copy()
    master = base.merge(cohort, on="cell_line_name", how="left")
    master["sample_id"] = master["cell_line_name"]
    master["model_id"] = master["model_id"].fillna("").astype(str)
    master["is_depmap_mapped"] = master["model_id"].ne("").astype(int)
    master["TCGA_DESC"] = "LIHC"
    master["cohort"] = "LIHC"
    master["depmap_primary_disease"] = master.get("depmap_oncotree_subtype", "").fillna("").astype(str)
    keep = [
        "sample_id",
        "cell_line_name",
        "model_id",
        "is_depmap_mapped",
        "TCGA_DESC",
        "cohort",
        "depmap_oncotree_code",
        "depmap_primary_disease",
    ]
    for column in keep:
        if column not in master.columns:
            master[column] = ""
    return master[keep].sort_values("sample_id").reset_index(drop=True)


def build_response_labels(gdsc: pd.DataFrame, cell_master: pd.DataFrame) -> pd.DataFrame:
    labels = gdsc.copy()
    threshold = float(labels["ln_IC50"].quantile(0.3))
    labels["sample_id"] = labels["cell_line_name"]
    labels["canonical_drug_id"] = labels["DRUG_ID"].astype(int).astype(str)
    labels["pair_id"] = labels["sample_id"] + "__" + labels["canonical_drug_id"]
    labels["label_regression"] = labels["ln_IC50"]
    labels["label_binary"] = (labels["label_regression"] <= threshold).astype(int)
    labels["label_main"] = labels["label_regression"]
    labels["label_aux"] = labels["label_binary"]
    labels["label_main_type"] = "regression"
    labels["label_aux_type"] = "binary"
    labels["binary_threshold"] = threshold
    labels = labels.merge(
        cell_master[["sample_id", "model_id", "is_depmap_mapped"]],
        on="sample_id",
        how="left",
    )
    keep = [
        "pair_id",
        "sample_id",
        "cell_line_name",
        "model_id",
        "is_depmap_mapped",
        "canonical_drug_id",
        "DRUG_ID",
        "drug_name",
        "TCGA_DESC",
        "gdsc_version",
        "putative_target",
        "pathway_name",
        "WEBRELEASE",
        "label_regression",
        "label_binary",
        "label_main",
        "label_aux",
        "label_main_type",
        "label_aux_type",
        "binary_threshold",
    ]
    return labels[keep].reset_index(drop=True)


def build_drug_master(paths: PipelinePaths, labels: pd.DataFrame) -> pd.DataFrame:
    catalog = pd.read_parquet(paths.raw_cache / "drug_features_catalog.parquet").copy()
    catalog["DRUG_ID"] = pd.to_numeric(catalog["DRUG_ID"], errors="coerce").astype("Int64")
    catalog = catalog.dropna(subset=["DRUG_ID"]).copy()
    catalog["DRUG_ID"] = catalog["DRUG_ID"].astype(int)
    if "DRUG_NAME" in catalog.columns:
        catalog = catalog.rename(columns={"DRUG_NAME": "drug_name"})
    catalog["canonical_drug_id"] = catalog["DRUG_ID"].astype(str)
    catalog["drug_name"] = catalog["drug_name"].fillna("").astype(str)
    catalog["canonical_smiles"] = catalog["canonical_smiles"].fillna("").astype(str)
    catalog["has_smiles"] = pd.to_numeric(catalog["has_smiles"], errors="coerce").fillna(0).astype(int)

    label_drugs = labels[["DRUG_ID", "drug_name", "putative_target", "pathway_name"]].drop_duplicates("DRUG_ID")
    master = label_drugs.merge(catalog, on=["DRUG_ID", "drug_name"], how="left")
    fallback_name = labels.groupby("DRUG_ID")["drug_name"].first()
    master["canonical_drug_id"] = master["DRUG_ID"].astype(str)
    master["drug_name"] = master["drug_name"].fillna(master["DRUG_ID"].map(fallback_name)).fillna("").astype(str)
    master["drug_name_norm"] = master["drug_name"].map(normalize_name)
    master["canonical_smiles"] = master["canonical_smiles"].fillna("").astype(str)
    master["has_smiles"] = (master["canonical_smiles"].str.len() > 0).astype(int)
    master["match_source"] = master["match_source"].fillna("unmatched").astype(str)
    master["target_pathway"] = master["pathway_name"].fillna("").astype(str)
    master["target"] = master["putative_target"].fillna("").astype(str)
    return master.sort_values("canonical_drug_id").reset_index(drop=True)


def load_target_mapping(paths: PipelinePaths, drug_master: pd.DataFrame) -> pd.DataFrame:
    target_mapping = pd.read_parquet(paths.raw_cache / "drug_target_mapping.parquet").copy()
    target_mapping["canonical_drug_id"] = target_mapping["canonical_drug_id"].astype(str).str.strip()
    target_mapping["target_gene_symbol"] = target_mapping["target_gene_symbol"].astype(str).str.strip()
    valid_drugs = set(drug_master["canonical_drug_id"].astype(str))
    target_mapping = target_mapping[target_mapping["canonical_drug_id"].isin(valid_drugs)].copy()
    return target_mapping.drop_duplicates().reset_index(drop=True)


def build_sample_crispr(paths: PipelinePaths, cell_master: pd.DataFrame, max_features: int) -> pd.DataFrame:
    model_ids = cell_master.loc[cell_master["model_id"].astype(str).ne(""), "model_id"].astype(str).tolist()
    crispr = pd.read_csv(paths.raw_cache / "CRISPRGeneEffect.csv", index_col=0)
    crispr.index = crispr.index.astype(str)
    present = [model_id for model_id in model_ids if model_id in crispr.index]
    if not present:
        return cell_master[["sample_id"]].assign(sample__has_crispr_profile=0)

    sub = crispr.loc[present].apply(pd.to_numeric, errors="coerce")
    sub.columns = [clean_gene_name(column) for column in sub.columns]
    sub = sub.T.groupby(level=0).mean().T
    variances = sub.var(axis=0).sort_values(ascending=False)
    selected = variances.head(min(max_features, len(variances))).index.tolist()
    sub = sub[selected].copy()
    sub.columns = [f"sample__crispr__{safe_feature_name(column)}" for column in sub.columns]
    sub.insert(0, "model_id", sub.index.astype(str))
    model_to_sample = cell_master.set_index("model_id")["sample_id"].to_dict()
    sub["sample_id"] = sub["model_id"].map(model_to_sample)
    sub["sample__has_crispr_profile"] = 1
    out = cell_master[["sample_id", "model_id"]].merge(sub.drop(columns=["model_id"]), on="sample_id", how="left")
    out["sample__has_crispr_profile"] = out["sample__has_crispr_profile"].fillna(0).astype(int)
    crispr_cols = [c for c in out.columns if c.startswith("sample__crispr__")]
    out[crispr_cols] = out[crispr_cols].fillna(out[crispr_cols].median(numeric_only=True)).fillna(0.0)
    return out.drop(columns=["model_id"]).sort_values("sample_id").reset_index(drop=True)


def build_disease_context(paths: PipelinePaths) -> pd.DataFrame:
    expr = pd.read_csv(paths.raw_cache / "TCGA.LIHC.sampleMap_HiSeqV2.gz", sep="\t", index_col=0)
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tumor_cols = [col for col in expr.columns if tcga_sample_type(col) in {"01", "02"}]
    normal_cols = [col for col in expr.columns if tcga_sample_type(col) in {"10", "11"}]
    if not tumor_cols:
        tumor_cols = list(expr.columns)
    tumor_mean = expr[tumor_cols].mean(axis=1)
    if normal_cols:
        normal_mean = expr[normal_cols].mean(axis=1)
        delta = tumor_mean - normal_mean
    else:
        normal_mean = pd.Series(0.0, index=expr.index)
        delta = tumor_mean - tumor_mean.median()
    signature = pd.DataFrame(
        {
            "gene_symbol": expr.index.astype(str),
            "tcga_lihc_tumor_mean": tumor_mean.values,
            "tcga_lihc_normal_mean": normal_mean.reindex(expr.index).values,
            "delta_tumor_vs_normal": delta.values,
            "abs_delta": np.abs(delta.values),
        }
    )
    signature = signature.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    signature["abs_delta_rank"] = np.arange(1, len(signature) + 1)
    write_json(
        paths.disease_context / "disease_context_summary.json",
        {
            "source": "TCGA.LIHC.sampleMap_HiSeqV2.gz",
            "n_genes": int(signature.shape[0]),
            "n_tumor_samples": int(len(tumor_cols)),
            "n_normal_samples": int(len(normal_cols)),
            "top10_delta_genes": signature.head(10)[["gene_symbol", "delta_tumor_vs_normal"]].to_dict("records"),
        },
    )
    return signature


def build_sample_features(cell_master: pd.DataFrame, sample_crispr: pd.DataFrame) -> pd.DataFrame:
    base = cell_master.copy()
    base["sample__is_depmap_mapped"] = base["is_depmap_mapped"].fillna(0).astype(int)
    base["sample__is_lihc"] = 1
    for sample_id in sorted(base["sample_id"].astype(str).unique()):
        base[f"sample__cell_line__{safe_feature_name(sample_id)}"] = (base["sample_id"].astype(str) == sample_id).astype(int)
    out = base.merge(sample_crispr, on="sample_id", how="left")
    out["sample__has_crispr_profile"] = out["sample__has_crispr_profile"].fillna(0).astype(int)
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].fillna(0.0)
    return out.sort_values("sample_id").reset_index(drop=True)


def build_drug_features(
    paths: PipelinePaths,
    drug_master: pd.DataFrame,
    target_mapping: pd.DataFrame,
    max_lincs_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = drug_master.copy()
    features["canonical_drug_id"] = features["canonical_drug_id"].astype(str)
    target_summary = (
        target_mapping.groupby("canonical_drug_id")["target_gene_symbol"]
        .agg(lambda values: "|".join(sorted({str(v).strip() for v in values if str(v).strip()})))
        .rename("drug__target_list")
        .reset_index()
    )
    target_summary["drug__target_count"] = target_summary["drug__target_list"].map(
        lambda value: len([v for v in str(value).split("|") if v])
    )
    features = features.merge(target_summary, on="canonical_drug_id", how="left")
    features["drug__target_list"] = features["drug__target_list"].fillna("")
    features["drug__target_count"] = pd.to_numeric(features["drug__target_count"], errors="coerce").fillna(0).astype(int)
    features["drug__has_target_mapping"] = (features["drug__target_count"] > 0).astype(int)
    features["drug__canonical_smiles"] = features["canonical_smiles"].fillna("").astype(str)
    features["drug__has_smiles"] = (features["drug__canonical_smiles"].str.len() > 0).astype(int)
    features["drug__smiles_length"] = features["drug__canonical_smiles"].str.len()
    rdkit_df = build_rdkit_drug_features(features[["canonical_drug_id", "drug__canonical_smiles"]])
    features = features.merge(rdkit_df, on="canonical_drug_id", how="left")

    lincs = pd.read_parquet(paths.raw_cache / "lincs_drug_signature_normalized.parquet")
    lincs["canonical_drug_id"] = lincs["canonical_drug_id"].astype(str)
    lincs_feature_cols = [c for c in lincs.columns if c != "canonical_drug_id"]
    if lincs_feature_cols:
        var = lincs[lincs_feature_cols].var(axis=0).sort_values(ascending=False)
        selected = var.head(min(max_lincs_features, len(var))).index.tolist()
        lincs_selected = lincs[["canonical_drug_id", *selected]].copy()
        rename_map = {col: "lincs__" + safe_feature_name(col.replace("crispr__", "")) for col in selected}
        lincs_selected = lincs_selected.rename(columns=rename_map)
        features = features.merge(lincs_selected, on="canonical_drug_id", how="left")
        features["drug__has_lincs_signature"] = features[[rename_map[c] for c in selected]].notna().any(axis=1).astype(int)
        features[[rename_map[c] for c in selected]] = features[[rename_map[c] for c in selected]].fillna(0.0)
    else:
        features["drug__has_lincs_signature"] = 0
    return features.sort_values("canonical_drug_id").reset_index(drop=True), lincs


def build_rdkit_drug_features(drugs: pd.DataFrame, nbits: int = 2048) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in drugs.iterrows():
        drug_id = str(row["canonical_drug_id"])
        smiles = str(row.get("drug__canonical_smiles", "") or "")
        mol = mol_from_smiles_or_none(smiles)
        base = {"canonical_drug_id": drug_id, "drug_has_valid_smiles": int(mol is not None)}
        fp_bits = np.zeros((nbits,), dtype=np.int8)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
            DataStructs.ConvertToNumpyArray(fp, fp_bits)
            base.update(
                {
                    "drug_desc_mol_wt": float(Descriptors.MolWt(mol)),
                    "drug_desc_logp": float(Crippen.MolLogP(mol)),
                    "drug_desc_tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                    "drug_desc_hbd": float(Lipinski.NumHDonors(mol)),
                    "drug_desc_hba": float(Lipinski.NumHAcceptors(mol)),
                    "drug_desc_rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
                    "drug_desc_ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
                    "drug_desc_heavy_atoms": float(mol.GetNumHeavyAtoms()),
                    "drug_desc_frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                }
            )
        else:
            base.update(
                {
                    "drug_desc_mol_wt": 0.0,
                    "drug_desc_logp": 0.0,
                    "drug_desc_tpsa": 0.0,
                    "drug_desc_hbd": 0.0,
                    "drug_desc_hba": 0.0,
                    "drug_desc_rot_bonds": 0.0,
                    "drug_desc_ring_count": 0.0,
                    "drug_desc_heavy_atoms": 0.0,
                    "drug_desc_frac_csp3": 0.0,
                }
            )
        for idx, value in enumerate(fp_bits):
            base[f"drug_morgan_{idx:04d}"] = int(value)
        rows.append(base)
    return pd.DataFrame(rows)


def build_pair_features(
    labels: pd.DataFrame,
    target_mapping: pd.DataFrame,
    signature: pd.DataFrame,
    lincs: pd.DataFrame,
) -> pd.DataFrame:
    sig = signature.set_index("gene_symbol")
    sig_lookup = sig[["delta_tumor_vs_normal", "abs_delta_rank", "tcga_lihc_tumor_mean"]].to_dict("index")
    target_lookup = (
        target_mapping.groupby("canonical_drug_id")["target_gene_symbol"]
        .apply(lambda values: [str(v).strip().upper() for v in values if is_gene_like(str(v).strip())])
        .to_dict()
    )
    lincs_scores = compute_lincs_reversal_scores(lincs, signature)
    rows = []
    for _, row in labels[["pair_id", "sample_id", "canonical_drug_id"]].iterrows():
        drug_id = str(row["canonical_drug_id"])
        targets = target_lookup.get(drug_id, [])
        matched = [sig_lookup[t] for t in targets if t in sig_lookup]
        lincs_row = lincs_scores.get(drug_id, {})
        if matched:
            deltas = [float(x["delta_tumor_vs_normal"]) for x in matched]
            ranks = [float(x["abs_delta_rank"]) for x in matched]
            exprs = [float(x["tcga_lihc_tumor_mean"]) for x in matched]
        else:
            deltas, ranks, exprs = [], [], []
        rows.append(
            {
                "pair_id": row["pair_id"],
                "sample_id": row["sample_id"],
                "canonical_drug_id": drug_id,
                "pair__cohort_lihc": 1.0,
                "pair__target_count": float(len(targets)),
                "pair__matched_target_count": float(len(matched)),
                "pair__target_match_fraction": float(len(matched) / len(targets)) if targets else 0.0,
                "pair__mean_target_delta": float(np.mean(deltas)) if deltas else 0.0,
                "pair__mean_abs_target_delta": float(np.mean(np.abs(deltas))) if deltas else 0.0,
                "pair__max_abs_target_delta": float(np.max(np.abs(deltas))) if deltas else 0.0,
                "pair__mean_target_expression": float(np.mean(exprs)) if exprs else 0.0,
                "pair__top50_target_hits": float(sum(rank <= 50 for rank in ranks)),
                "pair__top200_target_hits": float(sum(rank <= 200 for rank in ranks)),
                "pair__has_lincs_signature": float(lincs_row.get("has_lincs_signature", 0.0)),
                "pair__lincs_disease_cosine": float(lincs_row.get("lincs_disease_cosine", 0.0)),
                "pair__lincs_reversal_score": float(lincs_row.get("lincs_reversal_score", 0.0)),
                "pair__lincs_overlap_genes": float(lincs_row.get("lincs_overlap_genes", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def compute_lincs_reversal_scores(lincs: pd.DataFrame, signature: pd.DataFrame) -> dict[str, dict[str, float]]:
    if lincs.empty:
        return {}
    sig = signature.set_index("gene_symbol")["delta_tumor_vs_normal"].astype(float)
    feature_cols = [c for c in lincs.columns if c != "canonical_drug_id"]
    genes = [str(c).replace("crispr__", "").upper() for c in feature_cols]
    overlap_idx = [idx for idx, gene in enumerate(genes) if gene in sig.index]
    if not overlap_idx:
        return {}
    sig_vec = sig.reindex([genes[idx] for idx in overlap_idx]).to_numpy(dtype=np.float32)
    sig_norm = float(np.linalg.norm(sig_vec))
    if sig_norm < 1e-8:
        return {}
    out = {}
    values = lincs[feature_cols].to_numpy(dtype=np.float32)
    for row_idx, drug_id in enumerate(lincs["canonical_drug_id"].astype(str)):
        vec = values[row_idx, overlap_idx]
        denom = float(np.linalg.norm(vec)) * sig_norm
        cosine = float(np.dot(vec, sig_vec) / denom) if denom > 1e-8 else 0.0
        out[drug_id] = {
            "has_lincs_signature": 1.0,
            "lincs_disease_cosine": cosine,
            "lincs_reversal_score": -cosine,
            "lincs_overlap_genes": float(len(overlap_idx)),
        }
    return out


def assemble_train_table(
    labels: pd.DataFrame,
    sample_features: pd.DataFrame,
    drug_features: pd.DataFrame,
    pair_features: pd.DataFrame,
) -> pd.DataFrame:
    sample_cols = [c for c in sample_features.columns if c not in {"cell_line_name", "model_id"}]
    drug_cols = [c for c in drug_features.columns if c not in {"DRUG_ID", "drug_name"}]
    train = labels.merge(sample_features[sample_cols], on="sample_id", how="left")
    train = train.merge(drug_features[drug_cols], on="canonical_drug_id", how="left")
    train = train.merge(pair_features, on=["pair_id", "sample_id", "canonical_drug_id"], how="left")
    numeric = train.select_dtypes(include=[np.number]).columns
    train[numeric] = train[numeric].fillna(0.0)
    return train.sort_values("pair_id").reset_index(drop=True)


def add_strong_context_and_smiles_features(
    train: pd.DataFrame,
    target_mapping: pd.DataFrame,
    smiles_svd_dim: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add lung/BRCA-style strong context one-hot and SMILES SVD features."""
    enriched = train.copy()
    enriched["TCGA_DESC"] = enriched.get("TCGA_DESC", "LIHC")
    enriched["TCGA_DESC"] = enriched["TCGA_DESC"].fillna("LIHC").astype(str).replace({"": "LIHC"})

    pathway = first_existing_series(enriched, ["pathway_name", "target_pathway"], "__MISSING__")
    enriched["PATHWAY_NAME_NORMALIZED"] = pathway.map(normalize_context_value)

    target_class = build_target_resolution_class(target_mapping, enriched)
    enriched["classification"] = enriched["canonical_drug_id"].astype(str).map(target_class).fillna("mixed_gene_and_ambiguous")

    source_count = pd.Series(0, index=enriched.index, dtype=np.int16)
    if "drug_has_valid_smiles" in enriched.columns:
        source_count += pd.to_numeric(enriched["drug_has_valid_smiles"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    elif "drug__has_smiles" in enriched.columns:
        source_count += pd.to_numeric(enriched["drug__has_smiles"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "drug__has_lincs_signature" in enriched.columns:
        source_count += pd.to_numeric(enriched["drug__has_lincs_signature"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "drug__target_count" in enriched.columns:
        source_count += (pd.to_numeric(enriched["drug__target_count"], errors="coerce").fillna(0) > 0).astype(int)
    enriched["drug_bridge_strength"] = np.where(source_count >= 2, "multi_source", "single_source")
    enriched["stage3_resolution_status"] = np.where(
        enriched["classification"].eq("all_tokens_gene_matched"),
        "resolved_or_cleaned",
        "partial_gene_resolution_with_family_remaining",
    )

    onehot_cols: list[str] = []
    onehot_levels: dict[str, list[str]] = {}
    for column in STRONG_CONTEXT_COLS:
        values = enriched[column].fillna("__MISSING__").astype(str).replace({"": "__MISSING__"})
        levels = sorted(values.unique().tolist())
        onehot_levels[column] = levels
        for level in levels:
            feature = f"ctxcat__{safe_feature_name(column)}__{safe_feature_name(level)}"
            enriched[feature] = values.eq(level).astype(np.int8)
            onehot_cols.append(feature)

    smiles_features, smiles_summary = build_smiles_svd_features(
        enriched[["canonical_drug_id", "drug__canonical_smiles"]].drop_duplicates("canonical_drug_id"),
        requested_dim=smiles_svd_dim,
    )
    enriched = enriched.merge(smiles_features, on="canonical_drug_id", how="left")
    smiles_cols = [c for c in smiles_features.columns if c.startswith("smiles_svd_")]
    enriched[smiles_cols] = enriched[smiles_cols].fillna(0.0)

    summary = {
        "strong_context": {
            "columns": STRONG_CONTEXT_COLS,
            "onehot_dim_for_ml": int(len(onehot_cols)),
            "onehot_columns": onehot_cols,
            "levels": onehot_levels,
        },
        "smiles": smiles_summary,
        "shape_before": [int(train.shape[0]), int(train.shape[1])],
        "shape_after": [int(enriched.shape[0]), int(enriched.shape[1])],
    }
    return enriched, summary


def build_smiles_svd_features(drug_smiles: pd.DataFrame, requested_dim: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = drug_smiles.copy()
    base["canonical_drug_id"] = base["canonical_drug_id"].astype(str)
    base["drug__canonical_smiles"] = base["drug__canonical_smiles"].fillna("").astype(str)
    texts = base["drug__canonical_smiles"].where(base["drug__canonical_smiles"].str.strip().ne(""), "__EMPTY__")

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
    tfidf = vectorizer.fit_transform(texts.tolist())
    max_dim = max(1, min(tfidf.shape[0], tfidf.shape[1]) - 1)
    dim = min(int(requested_dim), max_dim)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    values = svd.fit_transform(tfidf).astype(np.float32)
    svd_cols = [f"smiles_svd_{idx:02d}" for idx in range(dim)]
    out = pd.concat(
        [base[["canonical_drug_id"]].reset_index(drop=True), pd.DataFrame(values, columns=svd_cols)],
        axis=1,
    )
    summary = {
        "requested_dim": int(requested_dim),
        "svd_dim": int(dim),
        "tfidf_shape": [int(tfidf.shape[0]), int(tfidf.shape[1])],
        "vectorizer_vocab_size": int(len(vectorizer.vocabulary_)),
        "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
        "drug_count": int(base.shape[0]),
        "non_empty_smiles_drugs": int(base["drug__canonical_smiles"].str.strip().ne("").sum()),
        "columns": svd_cols,
    }
    return out, summary


def build_target_resolution_class(target_mapping: pd.DataFrame, train: pd.DataFrame) -> dict[str, str]:
    mapped_tokens = (
        target_mapping.assign(canonical_drug_id=target_mapping["canonical_drug_id"].astype(str))
        .groupby("canonical_drug_id")["target_gene_symbol"]
        .apply(lambda values: [str(v).strip() for v in values if str(v).strip()])
        .to_dict()
    )
    raw_targets = (
        train[["canonical_drug_id", "putative_target"]]
        .drop_duplicates("canonical_drug_id")
        .assign(canonical_drug_id=lambda df: df["canonical_drug_id"].astype(str))
        if "putative_target" in train.columns
        else pd.DataFrame(columns=["canonical_drug_id", "putative_target"])
    )
    raw_lookup = raw_targets.set_index("canonical_drug_id")["putative_target"].astype(str).to_dict()
    out: dict[str, str] = {}
    for drug_id in train["canonical_drug_id"].astype(str).unique():
        mapped = mapped_tokens.get(drug_id, [])
        raw = split_target_tokens(raw_lookup.get(drug_id, ""))
        tokens = mapped or raw
        if not tokens:
            out[drug_id] = "mixed_gene_and_ambiguous"
        elif all(is_gene_like(token) for token in tokens):
            out[drug_id] = "all_tokens_gene_matched"
        elif any(is_gene_like(token) for token in tokens):
            out[drug_id] = "mixed_gene_and_non_gene"
        else:
            out[drug_id] = "mixed_gene_and_ambiguous"
    return out


def split_target_tokens(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[|,;/]+", text) if token.strip()]


def first_existing_series(df: pd.DataFrame, candidates: list[str], default: str) -> pd.Series:
    result = pd.Series(default, index=df.index, dtype="object")
    for column in candidates:
        if column not in df.columns:
            continue
        values = df[column].fillna("").astype(str).str.strip()
        result = result.where(result.astype(str).str.strip().ne(default), values)
        result = result.where(result.astype(str).str.strip().ne(""), values)
    return result.replace({"": default})


def normalize_context_value(value: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return "__MISSING__"
    return safe_feature_name(text).upper()


def build_slim_table(train: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    filtered = train.copy()
    if "drug_has_valid_smiles" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["drug_has_valid_smiles"], errors="coerce").fillna(0) > 0].copy()

    numeric_cols = numeric_feature_columns(filtered)
    crispr_cols = [c for c in numeric_cols if c.startswith("sample__crispr__")]
    lincs_cols = [c for c in numeric_cols if c.startswith("lincs__")]
    morgan_cols = [c for c in numeric_cols if c.startswith("drug_morgan_")]
    context_cols = [c for c in numeric_cols if c.startswith("ctxcat__")]
    smiles_svd_cols = [c for c in numeric_cols if c.startswith("smiles_svd_")]
    grouped_cols = set(crispr_cols + lincs_cols + morgan_cols + context_cols + smiles_svd_cols)
    other_cols = [c for c in numeric_cols if c not in grouped_cols]

    crispr_keep = top_variance_columns(filtered, crispr_cols, args.max_slim_crispr)
    lincs_keep = top_variance_columns(filtered, lincs_cols, args.max_slim_lincs)
    morgan_keep = [c for c in morgan_cols if float(pd.to_numeric(filtered[c], errors="coerce").fillna(0).var()) >= 0.01]
    if len(morgan_keep) > 1600:
        morgan_keep = top_variance_columns(filtered, morgan_keep, 1600)

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
        "ln_IC50",
        "drug__canonical_smiles",
        "canonical_smiles",
        "drug__target_list",
    ]
    meta_cols = [c for c in meta_cols if c in filtered.columns]
    out = filtered[meta_cols + feature_names].copy()
    out[feature_names] = out[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    summary = {
        "input_shape": [int(train.shape[0]), int(train.shape[1])],
        "slim_shape": [int(out.shape[0]), int(out.shape[1])],
        "feature_count": int(len(feature_names)),
        "crispr_input": int(len(crispr_cols)),
        "crispr_keep": int(len(crispr_keep)),
        "lincs_input": int(len(lincs_cols)),
        "lincs_keep": int(len(lincs_keep)),
        "morgan_input": int(len(morgan_cols)),
        "morgan_keep": int(len(morgan_keep)),
        "strong_context_onehot_keep": int(len(context_cols)),
        "smiles_svd_keep": int(len(smiles_svd_cols)),
        "other_numeric_keep": int(len(other_cols)),
        "invalid_smiles_rows_removed": int(train.shape[0] - filtered.shape[0]),
    }
    return out.reset_index(drop=True), feature_names, summary


def train_models(paths: PipelinePaths, args: argparse.Namespace) -> None:
    train = pd.read_parquet(paths.slim_inputs / "train_table.parquet")
    feature_names = json.loads((paths.slim_inputs / "feature_names.json").read_text())
    X = train[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = train["label_regression"].to_numpy(dtype=np.float32)
    groups = train["canonical_drug_id"].astype(str).to_numpy()
    pair_ids = train["pair_id"].astype(str).to_numpy()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    all_metrics: dict[str, Any] = {
        "n_rows": int(len(train)),
        "n_features": int(X.shape[1]),
        "models": {},
    }
    all_oof: dict[tuple[str, str], np.ndarray] = {}

    for split_type in ["groupcv", "randomcv"]:
        split_indices = make_splits(X, y, groups, args.n_splits, args.random_state, split_type)
        for model_name in models:
            print(f"Training {model_name} / {split_type}", flush=True)
            metrics, oof = fit_cv_model(model_name, X, y, split_indices, args.random_state)
            metrics["split_type"] = split_type
            metrics["model"] = model_name
            all_metrics["models"][f"{split_type}__{model_name}"] = metrics
            all_oof[(split_type, model_name)] = oof
            pd.DataFrame(
                {
                    "pair_id": pair_ids,
                    "sample_id": train["sample_id"].astype(str),
                    "canonical_drug_id": groups,
                    "drug_name": train["drug_name"].astype(str),
                    "y_true": y,
                    "y_pred": oof,
                    "split_type": split_type,
                    "model": model_name,
                }
            ).to_parquet(paths.model_runs / f"oof_{split_type}_{model_name}.parquet", index=False)

    ensemble = build_ensemble(paths, all_metrics, all_oof, pair_ids, train, y)
    all_metrics["groupcv_ensemble"] = ensemble
    write_json(paths.model_runs / "metrics_summary.json", all_metrics)


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
    split_type: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if split_type == "groupcv":
        n = min(n_splits, len(np.unique(groups)))
        return list(GroupKFold(n_splits=n).split(X, y, groups))
    n = min(n_splits, len(y))
    return list(KFold(n_splits=n, shuffle=True, random_state=random_state).split(X, y))


def fit_cv_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> tuple[dict[str, Any], np.ndarray]:
    oof = np.zeros(len(y), dtype=np.float32)
    folds = []
    for fold_idx, (tr, va) in enumerate(split_indices, start=1):
        if model_name == "lightgbm":
            preds = fit_lightgbm(X[tr], y[tr], X[va], random_state + fold_idx)
        elif model_name == "xgboost":
            preds = fit_xgboost(X[tr], y[tr], X[va], random_state + fold_idx)
        elif model_name == "residual_mlp":
            preds = fit_residual_mlp(X[tr], y[tr], X[va], random_state + fold_idx)
        elif model_name == "tabnet":
            preds = fit_tabnet(X[tr], y[tr], X[va], y[va], random_state + fold_idx)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        oof[va] = preds
        folds.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "spearman": spearman(y[va], preds),
                "rmse": rmse(y[va], preds),
            }
        )
    return {"spearman": spearman(y, oof), "rmse": rmse(y, oof), "fold_metrics": folds}, oof


def fit_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, seed: int) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_valid).astype(np.float32)


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, seed: int) -> np.ndarray:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=180,
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


def fit_residual_mlp(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    X_train_s, X_valid_s = scale_by_train(X_train, X_valid)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    y_train_s = ((y_train - y_mean) / y_std).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    model = ResidualMLP(X_train.shape[1], hidden_dim=128, num_blocks=3, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(12):
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


def fit_tabnet(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, seed: int) -> np.ndarray:
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor(
        n_d=12,
        n_a=12,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 0.02},
        seed=seed,
        device_name="cpu",
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train.astype(np.float32),
            y_train.reshape(-1, 1).astype(np.float32),
            eval_set=[(X_valid.astype(np.float32), y_valid.reshape(-1, 1).astype(np.float32))],
            eval_name=["valid"],
            eval_metric=["rmse"],
            max_epochs=12,
            patience=4,
            batch_size=1024,
            virtual_batch_size=128,
        )
    return model.predict(X_valid.astype(np.float32)).reshape(-1).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


def build_ensemble(
    paths: PipelinePaths,
    metrics: dict[str, Any],
    oof: dict[tuple[str, str], np.ndarray],
    pair_ids: np.ndarray,
    train: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    group_models = []
    for key, payload in metrics["models"].items():
        if key.startswith("groupcv__"):
            group_models.append((key.split("__", 1)[1], float(payload["spearman"])))
    group_models = sorted(group_models, key=lambda item: item[1], reverse=True)[:3]
    if not group_models:
        return {"status": "skipped"}
    raw_weights = np.array([max(score, 0.0) for _, score in group_models], dtype=np.float32)
    if float(raw_weights.sum()) <= 1e-8:
        raw_weights = np.ones(len(group_models), dtype=np.float32)
    weights = raw_weights / raw_weights.sum()
    pred = np.zeros(len(y), dtype=np.float32)
    for (model_name, _), weight in zip(group_models, weights):
        pred += weight * oof[("groupcv", model_name)]
    out = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "sample_id": train["sample_id"].astype(str),
            "canonical_drug_id": train["canonical_drug_id"].astype(str),
            "drug_name": train["drug_name"].astype(str),
            "y_true": y,
            "y_pred": pred,
            "split_type": "groupcv",
            "model": "weighted_top3_ensemble",
        }
    )
    out.to_parquet(paths.model_runs / "oof_groupcv_weighted_top3_ensemble.parquet", index=False)
    summary = {
        "models": [{"model": m, "spearman": s, "weight": float(w)} for (m, s), w in zip(group_models, weights)],
        "spearman": spearman(y, pred),
        "rmse": rmse(y, pred),
        "path": str(paths.model_runs / "oof_groupcv_weighted_top3_ensemble.parquet"),
    }
    write_json(paths.model_runs / "ensemble_summary.json", summary)
    return summary


def rank_and_filter(paths: PipelinePaths, args: argparse.Namespace) -> None:
    pred_path = paths.model_runs / "oof_groupcv_weighted_top3_ensemble.parquet"
    if not pred_path.exists():
        pred_path = paths.model_runs / "oof_groupcv_lightgbm.parquet"
    preds = pd.read_parquet(pred_path)
    train = pd.read_parquet(paths.slim_inputs / "train_table.parquet")
    drug_meta = pd.read_parquet(paths.standardized / "drug_master.parquet")
    pair = preds.merge(
        train[["pair_id", "drug__canonical_smiles", "drug_has_valid_smiles", "drug__target_list"]].drop_duplicates("pair_id"),
        on="pair_id",
        how="left",
    )
    pair["pred_rank_within_sample"] = pair.groupby("sample_id")["y_pred"].rank(method="min", ascending=True)
    pair["true_rank_within_sample"] = pair.groupby("sample_id")["y_true"].rank(method="min", ascending=True)
    for k in [5, 10, 20]:
        pair[f"pred_top{k}"] = (pair["pred_rank_within_sample"] <= k).astype(int)
        pair[f"true_top{k}"] = (pair["true_rank_within_sample"] <= k).astype(int)
    pair["abs_error"] = (pair["y_true"] - pair["y_pred"]).abs()

    drug_scores = aggregate_drug_scores(pair, drug_meta)
    drug_scores = drug_scores.sort_values(
        ["final_selection_score", "pred_top20_rate", "mean_y_pred"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    drug_scores["final_rank"] = np.arange(1, len(drug_scores) + 1)
    collapsed_scores = collapse_drug_name_duplicates(drug_scores)
    selected = collapsed_scores.head(args.candidate_limit).copy()
    selected["liver_relevance_class"] = selected["drug_name"].map(classify_liver_relevance)

    admet = apply_admet_gate(selected, paths.raw_cache / "admet/tdc_admet_group/admet_group")
    admet = admet.sort_values(["admet_strict_pass", "admet_adjusted_score", "final_selection_score"], ascending=[False, False, False])
    admet["admet_filtered_rank"] = np.arange(1, len(admet) + 1)

    pair.to_parquet(paths.final_selection / "lihc_pair_predictions.parquet", index=False)
    drug_scores.to_parquet(paths.final_selection / "lihc_final_drug_scores.parquet", index=False)
    collapsed_scores.to_parquet(paths.final_selection / "lihc_final_drug_scores_collapsed_by_name.parquet", index=False)
    selected.to_csv(paths.final_selection / "lihc_selected_drugs_top50.csv", index=False)
    admet.to_csv(paths.final_selection / "lihc_admet_candidate_gate.csv", index=False)
    admet[admet["admet_strict_pass"]].head(15).to_csv(paths.final_selection / "lihc_admet_filtered_top15.csv", index=False)
    write_json(
        paths.final_selection / "selection_summary.json",
        {
            "prediction_source": str(pred_path),
            "n_pairs": int(pair.shape[0]),
            "n_drugs": int(drug_scores.shape[0]),
            "n_collapsed_drugs": int(collapsed_scores.shape[0]),
            "top10": collapsed_scores.head(10)[
                ["final_rank", "drug_name", "canonical_drug_id", "final_selection_score", "mean_y_pred", "pred_top20_rate"]
            ].to_dict("records"),
            "admet_pass_count_top50": int(admet["admet_strict_pass"].sum()),
        },
    )


def aggregate_drug_scores(pair: pd.DataFrame, drug_meta: pd.DataFrame) -> pd.DataFrame:
    agg = (
        pair.groupby("canonical_drug_id")
        .agg(
            n_pairs=("pair_id", "size"),
            n_samples=("sample_id", "nunique"),
            drug_name=("drug_name", lambda s: most_common(s)),
            mean_y_pred=("y_pred", "mean"),
            median_y_pred=("y_pred", "median"),
            q25_y_pred=("y_pred", lambda s: float(np.quantile(s, 0.25))),
            q75_y_pred=("y_pred", lambda s: float(np.quantile(s, 0.75))),
            mean_y_true=("y_true", "mean"),
            median_y_true=("y_true", "median"),
            pred_top5_rate=("pred_top5", "mean"),
            pred_top10_rate=("pred_top10", "mean"),
            pred_top20_rate=("pred_top20", "mean"),
            true_top10_rate=("true_top10", "mean"),
            true_top20_rate=("true_top20", "mean"),
            mean_abs_error=("abs_error", "mean"),
            canonical_smiles=("drug__canonical_smiles", lambda s: most_common(s)),
            has_valid_smiles=("drug_has_valid_smiles", "max"),
            target_list=("drug__target_list", lambda s: most_common(s)),
        )
        .reset_index()
    )
    meta_cols = [c for c in ["canonical_drug_id", "target", "target_pathway", "match_source", "has_smiles"] if c in drug_meta.columns]
    agg = agg.merge(drug_meta[meta_cols].drop_duplicates("canonical_drug_id"), on="canonical_drug_id", how="left")
    agg["mean_pred_score"] = percentile_score(-agg["mean_y_pred"])
    agg["top20_score"] = percentile_score(agg["pred_top20_rate"])
    agg["top10_score"] = percentile_score(agg["pred_top10_rate"])
    agg["observed_score"] = percentile_score(-agg["mean_y_true"])
    agg["error_score"] = percentile_score(-agg["mean_abs_error"])
    agg["coverage_score"] = percentile_score(agg["n_samples"])
    agg["final_selection_score"] = (
        0.35 * agg["mean_pred_score"]
        + 0.25 * agg["top20_score"]
        + 0.15 * agg["top10_score"]
        + 0.15 * agg["observed_score"]
        + 0.05 * agg["error_score"]
        + 0.05 * agg["coverage_score"]
    )
    return agg


def collapse_drug_name_duplicates(drug_scores: pd.DataFrame) -> pd.DataFrame:
    df = drug_scores.copy()
    df["drug_group_key"] = df["drug_name"].map(normalize_name)
    empty = df["drug_group_key"].eq("")
    df.loc[empty, "drug_group_key"] = df.loc[empty, "canonical_drug_id"].astype(str)
    df = df.sort_values(
        ["drug_group_key", "final_selection_score", "pred_top20_rate", "mean_y_pred"],
        ascending=[True, False, False, True],
    )
    representative = df.groupby("drug_group_key", as_index=False).head(1).copy()
    duplicate_summary = (
        df.groupby("drug_group_key")
        .agg(
            n_screening_entries=("canonical_drug_id", "size"),
            all_canonical_drug_ids=("canonical_drug_id", lambda s: ",".join(s.astype(str))),
            duplicate_best_score=("final_selection_score", "max"),
            duplicate_mean_score=("final_selection_score", "mean"),
            duplicate_best_pred_top20_rate=("pred_top20_rate", "max"),
        )
        .reset_index()
    )
    out = representative.merge(duplicate_summary, on="drug_group_key", how="left")
    out = out.sort_values(
        ["final_selection_score", "pred_top20_rate", "mean_y_pred"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out["final_rank"] = np.arange(1, len(out) + 1)
    return out


def apply_admet_gate(candidates: pd.DataFrame, admet_dir: Path) -> pd.DataFrame:
    out = candidates.copy()
    rdkit_rows = []
    pains = make_pains_catalog()
    for _, row in out.iterrows():
        smiles = str(row.get("canonical_smiles", "") or "")
        mol = mol_from_smiles_or_none(smiles)
        rdkit_rows.append(compute_candidate_rdkit(row["canonical_drug_id"], smiles, mol, pains))
    out = out.merge(pd.DataFrame(rdkit_rows), on="canonical_drug_id", how="left")

    assay_lookup = lookup_admet_assays(out, admet_dir)
    out = out.merge(assay_lookup, on="canonical_drug_id", how="left")
    hard_fail = (
        (out["rdkit_valid_smiles"] == 0)
        | (out["pains_alert_count"].fillna(99) > 0)
        | (out["lipinski_violations"].fillna(99) > 2)
    )
    for assay, cfg in ADMET_ASSAYS.items():
        pred_col = f"admet_{assay}_nearest_y"
        sim_col = f"admet_{assay}_nearest_similarity"
        if pred_col not in out.columns:
            continue
        confident = out[sim_col].fillna(0.0) >= 0.70
        if "hard_bad" in cfg:
            hard_fail = hard_fail | (confident & (out[pred_col].round().astype("Int64") == int(cfg["hard_bad"])))
    out["admet_strict_pass"] = ~hard_fail
    good_cols = [c for c in out.columns if c.endswith("_good_flag")]
    out["admet_good_signal_count"] = out[good_cols].sum(axis=1) if good_cols else 0
    out["admet_adjusted_score"] = out["final_selection_score"] - 0.04 * out["lipinski_violations"].fillna(3)
    out.loc[~out["admet_strict_pass"], "admet_adjusted_score"] -= 0.25
    return out


def compute_candidate_rdkit(drug_id: str, smiles: str, mol: Chem.Mol | None, pains: Any) -> dict[str, Any]:
    if mol is None:
        return {
            "canonical_drug_id": drug_id,
            "rdkit_valid_smiles": 0,
            "mol_weight": np.nan,
            "logp": np.nan,
            "tpsa": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "rot_bonds": np.nan,
            "pains_alert_count": np.nan,
            "lipinski_violations": np.nan,
        }
    mol_weight = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    violations = int(mol_weight > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return {
        "canonical_drug_id": drug_id,
        "rdkit_valid_smiles": 1,
        "mol_weight": mol_weight,
        "logp": logp,
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": hbd,
        "hba": hba,
        "rot_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "pains_alert_count": int(len(pains.GetMatches(mol))) if pains is not None else 0,
        "lipinski_violations": violations,
    }


def lookup_admet_assays(candidates: pd.DataFrame, admet_dir: Path) -> pd.DataFrame:
    rows = [{"canonical_drug_id": str(v)} for v in candidates["canonical_drug_id"]]
    out = pd.DataFrame(rows)
    mols = {
        str(row["canonical_drug_id"]): mol_from_smiles_or_none(str(row.get("canonical_smiles", "") or ""))
        for _, row in candidates.iterrows()
    }
    fps = {
        drug_id: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for drug_id, mol in mols.items()
        if mol is not None
    }
    for assay, cfg in ADMET_ASSAYS.items():
        assay_path = admet_dir / assay / "train_val.csv"
        if not assay_path.exists():
            continue
        ref = pd.read_csv(assay_path)
        ref["mol"] = ref["Drug"].map(lambda s: mol_from_smiles_or_none(str(s)))
        ref = ref[ref["mol"].notna()].copy()
        if ref.empty:
            continue
        ref["fp"] = ref["mol"].map(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        assay_rows = []
        for drug_id, fp in fps.items():
            sims = [DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in ref["fp"]]
            if not sims:
                assay_rows.append({"canonical_drug_id": drug_id})
                continue
            best_idx = int(np.argmax(sims))
            y = float(ref.iloc[best_idx]["Y"])
            sim = float(sims[best_idx])
            good = admet_good_flag(y, cfg)
            assay_rows.append(
                {
                    "canonical_drug_id": drug_id,
                    f"admet_{assay}_nearest_y": y,
                    f"admet_{assay}_nearest_similarity": sim,
                    f"admet_{assay}_good_flag": good if sim >= 0.70 else 0,
                }
            )
        out = out.merge(pd.DataFrame(assay_rows), on="canonical_drug_id", how="left")
    return out


def admet_good_flag(y: float, cfg: dict[str, Any]) -> int:
    if cfg.get("type") == "binary" and "good" in cfg:
        return int(round(y) == int(cfg["good"]))
    if "threshold" in cfg and cfg.get("good_direction") == "high":
        return int(y >= float(cfg["threshold"]))
    if "threshold" in cfg and cfg.get("good_direction") == "low":
        return int(y <= float(cfg["threshold"]))
    if "ideal_range" in cfg:
        low, high = cfg["ideal_range"]
        return int(float(low) <= y <= float(high))
    return 0


def render_final_report(paths: PipelinePaths, args: argparse.Namespace) -> None:
    metrics = json.loads((paths.model_runs / "metrics_summary.json").read_text()) if (paths.model_runs / "metrics_summary.json").exists() else {}
    inputs = json.loads((paths.slim_inputs / "slim_input_summary.json").read_text()) if (paths.slim_inputs / "slim_input_summary.json").exists() else {}
    selected_path = paths.final_selection / "lihc_selected_drugs_top50.csv"
    admet_path = paths.final_selection / "lihc_admet_candidate_gate.csv"
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    admet = pd.read_csv(admet_path) if admet_path.exists() else pd.DataFrame()

    md = render_markdown_report(inputs, metrics, selected, admet)
    html_text = render_html_report(inputs, metrics, selected, admet)
    (paths.reports / "liver_pipeline_final_report.md").write_text(md, encoding="utf-8")
    (paths.reports / "liver_pipeline_final_report.html").write_text(html_text, encoding="utf-8")
    if paths.final_selection.exists():
        (paths.final_selection / "liver_pipeline_final_report.html").write_text(html_text, encoding="utf-8")


def render_markdown_report(inputs: dict[str, Any], metrics: dict[str, Any], selected: pd.DataFrame, admet: pd.DataFrame) -> str:
    lines = [
        "# LIHC/HCC Drug Repurposing Pipeline Report",
        "",
        "## Model Input",
        f"- GDSC rows: {inputs.get('gdsc_rows', 'NA')}",
        f"- Cell lines: {inputs.get('gdsc_cell_lines', 'NA')}",
        f"- Drugs: {inputs.get('gdsc_drugs', 'NA')}",
        f"- Slim input shape: {inputs.get('slim_shape', 'NA')}",
        f"- Slim feature count: {inputs.get('slim_feature_count', 'NA')}",
        f"- Strong-context one-hot features: {inputs.get('strong_context_onehot_features', 'NA')}",
        f"- SMILES SVD features: {inputs.get('smiles_svd_features', 'NA')}",
        "",
        "## Model Metrics",
    ]
    for key, payload in metrics.get("models", {}).items():
        lines.append(f"- {key}: Spearman {payload.get('spearman', float('nan')):.4f}, RMSE {payload.get('rmse', float('nan')):.4f}")
    if "groupcv_ensemble" in metrics:
        e = metrics["groupcv_ensemble"]
        lines.append(f"- groupcv weighted top3 ensemble: Spearman {e.get('spearman', float('nan')):.4f}, RMSE {e.get('rmse', float('nan')):.4f}")
    lines.extend(["", "## Top Drugs"])
    if not selected.empty:
        for _, row in selected.head(15).iterrows():
            lines.append(
                f"- #{int(row['final_rank'])} {row['drug_name']} "
                f"(score={row['final_selection_score']:.4f}, class={row.get('liver_relevance_class', '')})"
            )
    lines.extend(["", "## ADMET Pass Top Candidates"])
    if not admet.empty:
        passed = admet[admet["admet_strict_pass"].astype(bool)].head(15)
        for _, row in passed.iterrows():
            lines.append(f"- {row['drug_name']} (adjusted={row['admet_adjusted_score']:.4f})")
    return "\n".join(lines) + "\n"


def render_html_report(inputs: dict[str, Any], metrics: dict[str, Any], selected: pd.DataFrame, admet: pd.DataFrame) -> str:
    metric_rows = []
    for key, payload in metrics.get("models", {}).items():
        metric_rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{payload.get('spearman', float('nan')):.4f}</td>"
            f"<td>{payload.get('rmse', float('nan')):.4f}</td></tr>"
        )
    if "groupcv_ensemble" in metrics:
        e = metrics["groupcv_ensemble"]
        metric_rows.append(
            f"<tr><td><b>groupcv weighted top3 ensemble</b></td><td>{e.get('spearman', float('nan')):.4f}</td>"
            f"<td>{e.get('rmse', float('nan')):.4f}</td></tr>"
        )
    top_rows = ""
    if not selected.empty:
        for _, row in selected.head(20).iterrows():
            top_rows += (
                f"<tr><td>{int(row['final_rank'])}</td><td>{html.escape(str(row['drug_name']))}</td>"
                f"<td>{html.escape(str(row.get('liver_relevance_class', '')))}</td>"
                f"<td>{row['final_selection_score']:.4f}</td><td>{row['mean_y_pred']:.4f}</td>"
                f"<td>{row['pred_top20_rate']:.3f}</td></tr>"
            )
    admet_rows = ""
    if not admet.empty:
        for _, row in admet.head(20).iterrows():
            badge = "PASS" if bool(row["admet_strict_pass"]) else "FAIL"
            admet_rows += (
                f"<tr><td>{html.escape(str(row['drug_name']))}</td><td>{badge}</td>"
                f"<td>{row['admet_adjusted_score']:.4f}</td><td>{row.get('lipinski_violations', '')}</td>"
                f"<td>{row.get('pains_alert_count', '')}</td></tr>"
            )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>LIHC/HCC Drug Repurposing Pipeline Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1, h2 {{ color: #173f35; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; }}
    .card {{ background: #f2f7f1; border: 1px solid #c8dec7; border-radius: 12px; padding: 14px; }}
    .label {{ color: #5f6f64; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d9e3dc; padding: 9px; text-align: left; }}
    th {{ background: #e7f0e5; }}
    .note {{ background: #fff8dc; border-left: 4px solid #d9a441; padding: 12px; }}
  </style>
</head>
<body>
  <h1>LIHC/HCC Drug Repurposing Pipeline Report</h1>
  <p class="note">This report is generated from the liver-specific pipeline using GDSC2 LIHC/HCC labels, DepMap CRISPR, drug SMILES, LINCS signatures, and TCGA-LIHC disease context.</p>
  <h2>Input Summary</h2>
  <div class="cards">
    <div class="card"><div class="label">GDSC rows</div><div class="value">{inputs.get('gdsc_rows', 'NA')}</div></div>
    <div class="card"><div class="label">Cell lines</div><div class="value">{inputs.get('gdsc_cell_lines', 'NA')}</div></div>
    <div class="card"><div class="label">Drugs</div><div class="value">{inputs.get('gdsc_drugs', 'NA')}</div></div>
    <div class="card"><div class="label">Slim features</div><div class="value">{inputs.get('slim_feature_count', 'NA')}</div></div>
    <div class="card"><div class="label">Strong context one-hot</div><div class="value">{inputs.get('strong_context_onehot_features', 'NA')}</div></div>
    <div class="card"><div class="label">SMILES SVD</div><div class="value">{inputs.get('smiles_svd_features', 'NA')}</div></div>
  </div>
  <h2>Model Metrics</h2>
  <table><thead><tr><th>Track</th><th>Spearman</th><th>RMSE</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Top Ranked Drugs</h2>
  <table><thead><tr><th>Rank</th><th>Drug</th><th>Liver relevance</th><th>Score</th><th>Mean pred</th><th>Top20 rate</th></tr></thead><tbody>{top_rows}</tbody></table>
  <h2>ADMET Gate</h2>
  <table><thead><tr><th>Drug</th><th>Gate</th><th>ADMET-adjusted score</th><th>Lipinski violations</th><th>PAINS alerts</th></tr></thead><tbody>{admet_rows}</tbody></table>
</body>
</html>
"""


def upload_outputs(paths: PipelinePaths, upload_prefix: str) -> None:
    run(["aws", "s3", "sync", str(paths.processed), f"{upload_prefix}/data/processed/"])
    run(["aws", "s3", "sync", str(paths.root / "outputs"), f"{upload_prefix}/outputs/"])
    run(
        [
            "aws",
            "s3",
            "sync",
            str(paths.root / "scripts"),
            f"{upload_prefix}/scripts/",
            "--exclude",
            "__pycache__/*",
            "--exclude",
            "*.pyc",
        ]
    )
    if (paths.root / "docs").exists():
        run(["aws", "s3", "sync", str(paths.root / "docs"), f"{upload_prefix}/docs/"])
    run(["aws", "s3", "cp", str(paths.root / "README.md"), f"{upload_prefix}/README.md"])
    if (paths.root / "requirements.txt").exists():
        run(["aws", "s3", "cp", str(paths.root / "requirements.txt"), f"{upload_prefix}/requirements.txt"])


def build_run_index(paths: PipelinePaths) -> dict[str, Any]:
    return {
        "repo": str(paths.root),
        "processed": str(paths.processed),
        "model_runs": str(paths.model_runs),
        "final_selection": str(paths.final_selection),
        "report_html": str(paths.reports / "liver_pipeline_final_report.html"),
    }


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(KEY_COLUMNS)
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def top_variance_columns(df: pd.DataFrame, cols: list[str], n: int) -> list[str]:
    if not cols or n <= 0:
        return []
    var = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).var(axis=0).sort_values(ascending=False)
    return var.head(min(n, len(var))).index.tolist()


def clean_gene_name(value: str) -> str:
    return re.sub(r"\s+\(\d+\)$", "", str(value)).strip().upper()


def safe_feature_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return cleaned or "missing"


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def is_gene_like(value: str) -> bool:
    token = str(value).strip().upper()
    if not token or token in {"NAN", "NA", "NONE", "UNKNOWN"}:
        return False
    if any(ch in token for ch in [" ", "/", ",", "(", ")"]):
        return False
    return bool(re.fullmatch(r"[A-Z0-9-]+", token))


def tcga_sample_type(barcode: str) -> str:
    parts = str(barcode).split("-")
    if len(parts) >= 4:
        return parts[3][:2]
    return ""


def scale_by_train(X_train: np.ndarray, X_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((X_train - mean) / std).astype(np.float32), ((X_valid - mean) / std).astype(np.float32)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def percentile_score(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if values.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=values.index)
    return (values.rank(method="average", ascending=True) - 1.0) / (len(values) - 1.0)


def most_common(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    return str(clean.value_counts().index[0])


def classify_liver_relevance(drug_name: str) -> str:
    name = str(drug_name).lower().strip()
    norm = normalize_name(name)
    known_norm = {normalize_name(v) for v in KNOWN_LIVER_DRUGS}
    if norm in known_norm:
        return "current_liver_cancer_use"
    liver_research_terms = {
        "gemcitabine",
        "cisplatin",
        "doxorubicin",
        "paclitaxel",
        "docetaxel",
        "oxaliplatin",
        "erlotinib",
        "gefitinib",
        "afatinib",
        "trametinib",
        "selumetinib",
        "palbociclib",
        "ribociclib",
        "everolimus",
        "temsirolimus",
        "vorinostat",
    }
    if norm in {normalize_name(v) for v in liver_research_terms}:
        return "liver_cancer_research_or_related"
    return "repurposing_candidate_unclear_liver_relation"


def make_pains_catalog() -> Any:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


def mol_from_smiles_or_none(smiles: str) -> Chem.Mol | None:
    value = str(smiles or "").strip()
    if value.lower() in {"", "nan", "none", "<na>", "null"}:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()
