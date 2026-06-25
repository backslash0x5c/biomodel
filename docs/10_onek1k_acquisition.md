# 10. OneK1K データ取得手順

多ドナー scRNA ＋ genotype の代表例 **OneK1K**（Yazar et al., Science 2022）を取得して
本リポジトリのパイプラインに接続する手順。スクリプト: `scripts/fetch_onek1k.py`
（既定 dry-run、無断ダウンロードはしない）。

## 1. データ概要（2022 論文時点）

| 項目 | 内容 |
|---|---|
| 論文 | Yazar et al., *Science* 2022（doi:10.1126/science.abf3041, PMID 35389779） |
| scRNA | 1,267,758 PBMC / 982 ドナー / 14 免疫細胞型（10x 3' v2） |
| genotype | Illumina Infinium Global Screening Array, 759,993 marker / 1,104 個体 |
| imputation | Michigan Imputation Server, 1000G phase III v5（Minimac4 + Eagle v2.4） |
| cis-eQTL SNP | R²≥0.8 かつ MAF>0.05 の 5,849,361 SNP |

## 2. アクセス先

| ポータル | URL | 内容 |
|---|---|---|
| GEO **GSE196830** | ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196830 | scRNA-seq ＋ array genotype |
| CELLxGENE / HCA | cellxgene.cziscience.com | 前処理済み .h5ad（cell-by-gene） |
| OneK1K portal | onek1k.org | eQTL のブラウズ |

> OneK1K の genotype は GEO で配布されるが、個人ゲノムは機微情報。利用規約を遵守。

## 3. 取得 → 前処理 → モデルの流れ

```bash
# 1) 取得手順を表示（dry-run; ダウンロードはしない）
python scripts/fetch_onek1k.py

# 2) 取得したファイルを置き、存在を検証
python scripts/fetch_onek1k.py --data-dir ./onek1k

# 3) 前処理 → モデル接続（要 anndata/pandas）
python scripts/fetch_onek1k.py --data-dir ./onek1k --run
```

想定ファイル（`--data-dir` 配下、命名は配布形式に合わせて調整）:
- `onek1k.h5ad` … 単一細胞発現（`load_anndata` で読む。obs の donor/batch/perturbation キー名を指定）
- `genotype_dosage.tsv` … donor × variant のドーズ（`align_genotype` で donor 整列）
- `predixcan_weights.tsv` …（任意）GReX 重み（docs/02 §2, §下記）

## 4. genotype 特徴: PCA → GReX（PrediXcan 流）

`GReXFeaturizer`（`data_pipeline.py`）で **遺伝的に制御された発現** を特徴にできる:

$$\text{GReX}_j(d) = \sum_{s \in \text{cis}(j)} w_{js}\, \text{dosage}_{d,s}$$

- 重み $w$ は PrediXcan/FUSION（GTEx 等で学習）の弾性ネット重み。`load_predixcan_weights`
  で `(gene, variant, weight)` の表から読む。
- 生物学的に解釈しやすい低次元表現になり、PCA フォールバックより eQTL に忠実
  （`scripts/preprocess_onek1k.py --genotype-features grex`）。
- PGx（CYP 等）star-allele 特徴も同様に追加可能（docs/02 §2.2）。

## 5. アクセス制御・倫理（重要）

- OneK1K は GEO 配布だが、**多くの集団・臨床データは controlled access**:
  - **dbGaP**: データ利用申請（DUC）→ DAC 承認 → 暗号化ダウンロード（prefetch/sra-tools）。
  - **EGA**: DAC へ申請 → EGA download client / `pyega3` で取得。
- 個人ゲノムは再識別リスクのある機微情報。**機関の IRB / データガバナンスに従う**こと。
- 本リポジトリのデモ・テストはすべて合成データで、実 genotype は扱わない。

## 6. 前処理時の注意（再掲, docs/07）

- 遺伝子 ID を Ensembl に統一、共通 HVG パネル（実運用 n_hvg≈2000）。
- genotype のアレル方向（strand）統一・imputation・cis-window 定義。
- batch/donor の交絡チェック（docs/02 §4, docs/09）と leave-one-donor 分割（docs/04）。
- 疎な (donor,pert) は `observed` マスクで学習から除外（docs/07）。
