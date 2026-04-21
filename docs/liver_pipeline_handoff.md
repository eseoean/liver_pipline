# Liver Pipeline Handoff

## 목적

폐암에서 정리한 BRCA-style strong-context + SMILES 계열 파이프라인을 간암(LIHC/HCC)에 맞춰 새로 구현했다. 핵심은 암종만 바꾸는 것이 아니라, label과 sample context를 간암 전용으로 다시 구성하는 것이다.

## 입력 데이터

- `GDSC2`: 간암/HCC cell line-drug `ln_IC50` label
- `DepMap`: 간암 cell line과 `ModelID` 매칭, `CRISPRGeneEffect` sample-specific dependency feature
- `DrugBank/ChEMBL staged catalog`: GDSC drug의 SMILES 매핑
- `GDSC screened compounds`: target/pathway annotation
- `LINCS`: drug perturbation signature coverage 및 disease reversal score
- `TCGA-LIHC`: tumor vs normal expression disease signature
- `TDC ADMET`: 최종 후보 ADMET gate

## 모델 입력셋 생성 과정

1. `GDSC2` staged LIHC table에서 15개 간암 cell line, 295개 GDSC drug, 4,164개 cell-drug pair를 읽는다.
2. `drug_features_catalog.parquet`의 SMILES를 붙이고 RDKit에서 유효 구조가 있는 약물만 slim input에 남긴다.
3. 이 기준으로 295개 약물 중 243개 약물이 최종 모델 입력 약물로 유지된다.
4. DepMap `CRISPRGeneEffect.csv`에서 15개 간암 cell line의 gene effect를 추출하고, 분산 상위 CRISPR feature를 sample block으로 구성한다.
5. TCGA-LIHC `HiSeqV2` expression matrix에서 tumor sample 373개와 normal sample 50개를 분리해 tumor-normal disease signature를 만든다.
6. LINCS drug signature와 TCGA-LIHC disease signature의 overlap을 이용해 LINCS reversal score를 pair feature로 만든다.
7. RDKit Morgan fingerprint와 descriptor를 drug block에 추가한다.
8. `sample + drug + pair` feature를 합쳐 full train table을 만들고, CRISPR/LINCS/Morgan block을 slim selection하여 약 5천 개 컬럼 입력셋을 만든다.

## 현재 산출물 요약

- Full train table: `4164 rows x 6398 columns`
- Slim train table: `3396 rows x 4914 columns`
- Numeric model features: `4902`
- 최종 모델 입력 약물: `243`
- 최종 모델 입력 cell line: `15`
- SMILES 없는 약물에서 제거된 pair: `768`

## 모델 실행

동일 입력셋으로 drug 기준 GroupCV와 random CV를 모두 실행했다.

- ML: LightGBM, XGBoost
- DL: ResidualMLP, TabNet
- 최종 ranking source: GroupCV top3 weighted ensemble

## 최종 후보 산출

모델 OOF prediction은 낮은 `ln_IC50`를 높은 sensitivity로 해석한다. 약물별로 평균 예측값, sample 내 top-k 진입률, 관측 IC50, error, coverage를 집계해 최종 score를 만들었다. 이후 drug name 기준으로 중복 screening ID를 collapse하고, top50 후보에 ADMET gate를 적용했다.

## 재현 명령

```bash
python3 scripts/run_liver_pipeline.py --stage all
```

이미 `data/raw_cache/`가 있으면 다음처럼 다운로드를 생략할 수 있다.

```bash
python3 scripts/run_liver_pipeline.py --stage all --skip-download
```

S3 업로드까지 포함하려면 다음을 사용한다.

```bash
python3 scripts/run_liver_pipeline.py --stage all --upload-s3
```

