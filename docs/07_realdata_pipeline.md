# 07. 実データ前処理パイプライン（OneK1K 等）

多ドナー scRNA ＋ genotype を、モデルが消費できる形に整える骨組み。実データは取得・倫理・
ストレージ制約が大きいので、本パイプラインは **インターフェースと処理段** を定義し、
`load_fake_onek1k()` により実データ無しでも end-to-end に動かせる
（`src/biomodel/data_pipeline.py`, `scripts/preprocess_onek1k.py`）。

## 1. 想定データと取得

| 種別 | 例 | 取得・注意 |
|---|---|---|
| scRNA（無摂動 or 摂動） | OneK1K（~125万 PBMC × ~1,000 ドナー） | GEO/ArrayExpress 等で公開 |
| genotype（SNP ドーズ） | OneK1K SNP アレイ | **dbGaP/EGA 等のアクセス制御・IRB が必要** |
| PGx 事前知識 | PharmGKB / CPIC | 公開（star-allele→phenotype） |

> genotype は機微情報。本リポジトリのデモは合成データのみで、実 genotype は扱わない。

## 2. インターフェース（`data_pipeline.py`）

```
RawScRNA   : counts(細胞×遺伝子), gene_ids, donor, batch, perturbation(controlは-1), cell_type
RawGenotype: dosage(ドナー×variant 0/1/2), variant_ids, donor_ids
        ↓ build_processed
ProcessedDataset: geno_features, control_cells, delta(観測), observed, gene_ids, pert_names, batch
        ↓ processed_to_simdata
SimData 互換 → 既存の train.py / evaluate.py がそのまま使える
```

## 3. 処理段

1. **正規化** `normalize_log1p`: library-size 正規化 ＋ log1p（scanpy 標準に対応）。
2. **遺伝子整列** `harmonize_genes` / `select_hvgs`: Ensembl ID 統一・共通 HVG パネル選択。
3. **pseudobulk と delta** `pseudobulk_and_delta`: ドナーごとに control をサブサンプル、
   $\Delta_{d,p} = \text{mean(treated)} - \text{mean(control)}$。未観測の (donor,pert) は
   `observed=0`（実データは疎になりがち）。
4. **genotype 特徴** `GenotypeFeaturizer`:
   - 既定 `pca`: 標準化＋PCA（依存を増やさないフォールバック）。
   - 実運用は **cis-eQTL/GReX**（PrediXcan 等）や **PGx star-allele** に差し替え（docs/02 §2）。

## 4. モデルへの接続と確認

`scripts/preprocess_onek1k.py` は fake OneK1K（Poisson カウント、個人差は genotype 依存）を
処理して `ProcessedDataset` を作り、`processed_to_simdata` で SimData 互換へ変換、
既存の FiLM+genotype モデルで leave-one-donor-out まで通す。

### 実測（fake OneK1K, leave-one-donor-out）

```
[raw] cells=28000 genes=200 donors=100 variants=500
[processed] donors=100 genes(HVG)=64 perts=6 geno_feat=16  観測率=1.00
FiLM+genotype    overall_r≈+0.80  indiv_R2≈+0.45  rank_rho≈+0.70
population-mean  overall_r≈+0.60  indiv_R2≈ 0.00  rank_rho≈ 0.00
```

raw counts からの一連の前処理を経ても、pipeline → model が連結し個人差予測まで通る。

## 5. 実データ化のチェックリスト

- [ ] `load_fake_onek1k` を `load_anndata`（.h5ad）＋ genotype 読込（plink/VCF）に差し替え。
- [ ] `obs` のカラム（donor/batch/perturbation/cell_type）を `RawScRNA` に割り当て。
- [ ] genotype のアレル方向（strand）統一・imputation・cis-window 定義。
- [ ] `GenotypeFeaturizer` を GReX/eQTL/PGx 実装に差し替え。
- [ ] 疎な (donor,pert) を `observed` マスクで損失・評価から除外（学習側の対応が必要）。
- [ ] batch/donor の交絡チェック（docs/02 §4）と leave-one-donor 分割（docs/04）。
- [ ] 倫理・アクセス制御（dbGaP/EGA, IRB）の遵守。
