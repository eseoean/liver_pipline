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
8. 폐암 최종 입력셋과 맞추기 위해 `TCGA_DESC`, `PATHWAY_NAME_NORMALIZED`, `classification`, `drug_bridge_strength`, `stage3_resolution_status`를 strong context one-hot feature로 추가한다.
9. canonical SMILES 문자열은 character n-gram TF-IDF로 변환한 뒤 64차원 TruncatedSVD feature로 압축해 numeric input에 추가한다.
10. `sample + drug + pair + strong context + SMILES SVD` feature를 합쳐 full train table을 만들고, CRISPR/LINCS/Morgan block을 slim selection하여 약 5천 개 컬럼 입력셋을 만든다.

## 현재 산출물 요약

- Full train table: `4164 rows x 6499 columns`
- Slim train table: `3396 rows x 5011 columns`
- Numeric model features: `4998`
- Strong context one-hot features: `32`
- SMILES SVD features: `64`
- 최종 모델 입력 약물: `243`
- 최종 모델 입력 cell line: `15`
- SMILES 없는 약물에서 제거된 pair: `768`

## 모델 실행

동일 입력셋으로 drug 기준 GroupCV와 random CV를 모두 실행했다.

- ML: LightGBM, XGBoost
- DL: ResidualMLP, TabNet
- 최종 ranking source: GroupCV top3 weighted ensemble

## 성능 해석

최종 LIHC/HCC 입력셋의 GroupCV weighted ensemble은 Spearman `0.5158`, RMSE `2.1912`였다. RandomCV의 주력 모델은 Spearman `0.81-0.84` 수준까지 올라가므로, 모델이 반응 패턴 자체를 못 배우는 것은 아니다. 다만 drug holdout GroupCV에서는 새 약물 일반화 성능이 BRCA/lung 기준보다 낮다.

이 차이가 간암의 작은 supervised label 규모에서 오는지 확인하기 위해 BRCA exact slim + strong context + SMILES 입력을 15개 cell line으로 downsample해 같은 모델군으로 재평가했다. BRCA full 29개 cell line의 GroupCV ensemble Spearman은 `0.5836`이었고, 15개 cell line으로 줄이면 `0.5428`로 낮아졌다. 이는 LIHC/HCC의 `0.5158`에 가까워지는 결과다.

따라서 간암 성능 하락은 파이프라인 실패라기보다 제한된 LIHC/HCC cell line 수와 pair 다양성 부족의 영향을 크게 받는 것으로 해석한다. 다만 BRCA 15개 subset이 여전히 간암보다 약간 높으므로, HCC cell-line 대표성, label noise, target-response signal sparsity, TCGA-LIHC context와 GDSC cell-line response 간 mismatch도 함께 작용할 가능성이 높다.

상세 비교 문서: `docs/liver_performance_gap_analysis_20260421.md`

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
