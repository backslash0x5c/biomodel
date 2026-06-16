# 05. データセット候補と「多ドナー × 薬剤 × single-cell」希少問題

技術よりも **データの intersection** が律速。何を持っていて何が足りないかを直視する。

## 1. データ三象限

```
                 摂動あり
                    │
   細胞株摂動         │   ★欲しい領域★
 （多摂動・少ドナー） │  多ドナー × 薬剤 × scRNA
 sci-Plex/Tahoe/LINCS│       （希少）
 ───────────────────┼───────────────────  多ドナー
                    │
        —           │   集団アトラス
                    │（多ドナー・無摂動・genotype付き）
                    │   OneK1K, CZ CELLxGENE
                 摂動なし
```

**問題**: 右上（多ドナー × 薬剤 × single-cell）が希少。
**戦略**: 左上で「薬→応答」、右下で「個人差構造」を学び、$\Phi$ で結合 → 少量の
患者由来データで fine-tune（`docs/01` §3.3）。

## 2. データセット候補

### 2.1 事前学習（段1）— 無摂動 scRNA
| データ | 規模 | 用途 |
|---|---|---|
| **CZ CELLxGENE Census** | 数千万〜億細胞、汎組織 | MGM 事前学習の主データ |
| **Human Cell Atlas** | 汎組織 | 同上 |

### 2.2 個人差構造（段3）— 多ドナー ＋ genotype
| データ | 内容 | 用途 |
|---|---|---|
| **OneK1K** | ~100万 PBMC × ~1,000 ドナー ＋ genotype | sc-eQTL、個人差の事前学習の本命 |
| **sc-eQTLGen / 各種 population scRNA** | 多ドナー scRNA ＋ genotype | eQTL 事前知識 $e_{\text{geno}}$ |
| **GTEx**（bulk だが eQTL 豊富） | 多組織 eQTL | GReX モデルの重み源 |

### 2.3 摂動応答（段2）— 薬剤・遺伝子摂動
| データ | 種類 | 用途 |
|---|---|---|
| **Tahoe-100M（2025）** | ~1億細胞・多薬剤の細胞株摂動アトラス | 「薬→応答」大規模学習 |
| **sci-Plex** | 用量応答付き化合物摂動 | 用量モデリング |
| **LINCS L1000** | bulk だが大規模化合物×細胞株 | 化合物 encoder 事前学習 |
| **Perturb-seq 各種** | CRISPR 遺伝子摂動 | 遺伝子摂動 encoder（GEARS 流） |

### 2.4 個別化（fine-tune）— 患者由来 × 薬剤応答
| データ | 内容 | 用途 |
|---|---|---|
| **BeatAML** | 患者検体 × 薬剤感受性（＋一部 omics） | ex vivo 薬剤応答の個人差 |
| **患者由来オルガノイド / PDX 薬剤スクリーニング** | 個人検体での薬剤応答 | 個別化の最終 fine-tune |
| **臨床 PGx コホート** | genotype × 薬効・有害事象 | PGx 事前知識・検証 |

### 2.5 PGx 事前知識
| リソース | 内容 |
|---|---|
| **PharmGKB / CPIC** | 薬–遺伝子関係、star-allele → metabolizer phenotype |
| **DrugBank / ChEMBL** | 化合物構造・標的（段2 の化合物特徴） |

## 3. 前処理・整合の注意

- **遺伝子空間の統一**: データ間で遺伝子 ID（Ensembl）に揃え、共通 HVG パネルを定義。
- **正規化**: library size 正規化＋log1p、または scGPT 流 binning。
- **genotype 整合**: imputation・アレル方向（strand）統一、cis-window 定義。
- **batch メタデータ保持**: identifiability のため batch/donor を必ず別カラムで保持
  （`docs/02` §4）。
- **倫理・ガバナンス**: genotype は機微情報。dbGaP/EGA 等のアクセス制御・IRB を遵守。
  本リポジトリのデモは **すべて合成データ**で、実 genotype は扱わない。

## 4. まず動かすための最小データ

実データ取得前に概念実証するため、本リポジトリは
`src/biomodel/simulate.py` で **個人差（genotype × 摂動の相互作用）を埋め込んだ
合成データ**を生成する。これにより、評価プロトコル（`docs/04`）と
モデル（`docs/01`）を実データなしで検証できる。
