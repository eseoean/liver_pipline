# Liver Drug Repurposing Pipeline

간암(LIHC/HCC) 전용 약물 반응 예측 파이프라인입니다. 폐암에서 검증한 BRCA-style strong-context + SMILES 계열 흐름을 간암 데이터에 맞춰 재구성합니다.

## 실행 요약

```bash
python3 scripts/run_liver_pipeline.py --stage all
```

기본 입력 원천은 `s3://say2-4team/Liver_raw/`이며, 실행 시 필요한 파일만 `data/raw_cache/`로 내려받습니다. `data/raw_cache/`는 Git에 올리지 않고 S3 원천에서 재현합니다.

## 주요 산출물

- `data/processed/standardized/`: GDSC-LIHC label, DepMap cell line mapping, drug catalog, target mapping
- `data/processed/disease_context/`: TCGA-LIHC tumor-normal disease signature
- `data/processed/model_inputs/`: sample/drug/pair feature table
- `data/processed/slim_inputs/`: 모델 입력용 slim numeric table, `X.npy`, `y.npy`, feature names
- `outputs/model_runs/`: GroupCV/random CV 모델 성능 및 OOF prediction
- `outputs/final_selection/`: 약물 랭킹, ADMET 필터링, 최종 HTML/Markdown 보고서
- `docs/liver_performance_gap_analysis_20260421.md`: BRCA 15-cell-line downsample 기반 간암 성능 하락 해석
- `outputs/reports/performance_gap_brca_downsample_20260421.json`: 간암/BRCA downsample 비교 요약 JSON

## 현재 설계

- Label: GDSC2 LIHC/HCC cell line-drug `ln_IC50`
- Sample block: DepMap CRISPRGeneEffect에서 간암 cell line별 dependency feature
- Drug block: DrugBank/ChEMBL 기반 SMILES, RDKit Morgan fingerprint, descriptors, target count
- Strong context block: TCGA cohort, pathway, target-resolution class, drug bridge strength를 one-hot feature로 추가
- SMILES SVD block: canonical SMILES 문자열을 character n-gram TF-IDF 후 64차원 SVD numeric feature로 추가
- LINCS block: 약물 perturbation signature coverage가 있는 약물에 drug-level signature feature 추가
- Pair block: drug target과 TCGA-LIHC disease signature의 overlap, LINCS reversal score
- Split: drug 기준 GroupCV와 random CV 모두 수행
- Final ranking: OOF prediction 기반 약물별 sensitivity score 집계 후 ADMET gate 적용
