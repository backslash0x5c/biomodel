# 02. 個人条件づけの深掘り — genotype / eQTL / PGx をどう入れるか

個人差予測の成否は「個人 embedding $z_{\text{indiv}}$ に何を、どう入れるか」でほぼ決まる。
ここを掘り下げる。

## 1. 個人差の源と観測量

$$z_{\text{indiv}} = g\big(e_{\text{geno}},\ z_{\text{donor\_baseline}},\ e_{\text{cov}}\big)$$

| 源 | 観測量 | 寄与する個人差 |
|---|---|---|
| germline 変異 / eQTL | genotype（SNP アレイ / WGS） | 遺伝子発現の制御差、薬剤標的・代謝酵素の差 |
| 薬物動態遺伝子（PGx: CYP2D6, CYP2C19, …） | genotype | 薬の代謝速度（poor/extensive metabolizer） |
| ベースライン細胞状態（疾患・炎症・老化） | 投与前 scRNA | 応答の出発点・経路の活性度 |
| 細胞組成（cell-type 比） | scRNA / deconvolution | 「どの細胞が反応するか」の比率 |
| 環境・共変量 | metadata（年齢・性別・薬歴） | 全体的な感受性シフト |

**重要な直観**: 個人差の相当部分は (a) 投与前トランスクリプトーム と (b) genotype に
既にエンコードされている。(a) は段1 encoder が拾う。(b) を明示的に入れるのがここの主題。

## 2. genotype をどう embedding するか

genotype は数百万 SNP と高次元・スパース。そのまま全 SNP を入れるのは非現実的。
現実的な順に:

### 2.1 eQTL 事前知識で次元圧縮（推奨の出発点）

- cis-eQTL（遺伝子近傍の制御変異）に絞る。各遺伝子 $j$ について、その発現を説明する
  上位 eQTL SNP の遺伝子型（0/1/2）を特徴に使う。
- さらに **GReX（genetically regulated expression）** = PrediXcan/FUSION 流に
  「genotype から予測した発現」を特徴量にすると、生物学的に解釈しやすい低次元表現になる。

  $$\widehat{\text{expr}}_j^{\text{germline}} = \sum_{s \in \text{cis}(j)} w_{js}\, \text{SNP}_s$$

  この $\widehat{\text{expr}}^{\text{germline}}$ を encoder に通して $e_{\text{geno}}$ とする。

### 2.2 PGx（薬物動態）遺伝子の明示的特徴

- PharmGKB / CPIC のガイドラインにある薬物代謝酵素（CYP450 ファミリー、TPMT, DPYD,
  UGT1A1, SLCO1B1 …）の **star-allele → metabolizer phenotype**（PM/IM/EM/UM）を
  カテゴリ特徴として入れる。
- 薬剤ごとに「どの PGx 遺伝子が効くか」が違うので、**薬剤 embedding と PGx 特徴の
  相互作用**を $\Phi$ に持たせると、薬剤特異的な個人差を拾いやすい。

### 2.3 polygenic score 的な集約

- 薬剤感受性・疾患に関する PRS（polygenic risk/response score）を共変量として入れる。
  低次元で頑健だが解像度は粗い。

### 2.4 学習可能な donor embedding（注意点あり）

- 各ドナーに自由な学習埋め込みを割り当てる方式は **未知ドナーに汎化しない**
  （embedding が学習集合のドナーに固定される）。
- 使うなら **amortized**: ドナーの genotype / baseline から $z_{\text{indiv}}$ を
  「推論する」encoder にする（free embedding ではなく関数で生成）。本設計はこちら。

## 3. 相互作用 $\Phi$ に個人差を担わせる設計

$\Phi(z_{\text{pert}}, z_{\text{indiv}})$ の三方式と、それぞれが表現できる個人差:

| 方式 | 数式（概略） | 表現できる個人差 | コスト |
|---|---|---|---|
| **FiLM** | $\gamma(z_{\text{indiv}})\odot v_p + \beta(z_{\text{indiv}})$ | 効果の **強度・符号のスケール／シフト** | 低 |
| **Cross-attention** | $\text{Attn}(Q{=}z_{\text{pert}}, K{,}V{=}z_{\text{indiv}})$ | 摂動と個人特徴の **選択的結合** | 中 |
| **Hypernetwork** | $\theta = h(z_{\text{indiv}});\ \text{effect}=f_\theta(z_{\text{pert}})$ | **個人ごとに異なる応答関数** | 高 |

実務上は **FiLM から始めて、データ量に応じて cross-attn / hypernet に拡張**するのが安全。
FiLM でも「薬の効果が個人で N 倍／逆向き」程度の異質性は表現できる。

## 4. identifiability — 「真の生物差」と「batch effect」の分離（最重要の落とし穴）

ドナーごとの差には、**真の生物学的個人差**と**技術的 batch effect**（実験日・試薬ロット・
プラットフォーム）が交絡する。これを分離できないと「個人差を当てた」が実は batch を
当てているだけ、になる。

対策:

1. **genotype を入れる**: batch は genotype と独立。genotype で説明できる差は生物差と
   解釈できる（少なくとも batch ではない）。
2. **batch 敵対的不変化**: encoder 表現が batch ラベルを予測できないように adversarial
   に学習。ただし donor 生物差まで消さないよう、batch と donor を分けて扱う。
3. **デザインで対処**: 可能なら同一バッチ内に複数ドナー、複数バッチに同一ドナーを配置
   （confounding を壊す実験計画）。
4. **negative control / 感度分析**: 既知の null 摂動で「偽の個人差」が出ないか確認。

## 5. 因果的妥当性のためのチェックリスト

- [ ] control / treated が同一ドナー由来か（unpaired なら分布マッチングで近似）。
- [ ] batch と donor が交絡していないか（交絡なら個人差の推定は不可能）。
- [ ] selection bias（誰が検体を提供したか）を共変量で考慮したか。
- [ ] 予測した個人差を **null 摂動・既知 PGx** で sanity check したか。
- [ ] 個人レベル予測の **較正（calibration）** を測ったか（平均だけでなく分散も）。

## 6. プロトタイプでの簡略化

`src/biomodel/simulate.py` では、上記を抽象化して

- 各ドナーに低次元 **latent genotype** $g_d \in \mathbb{R}^{k}$ を割り当て、
- 摂動効果を $v_p \odot (1 + W g_d)$ のように **genotype で変調**（= 真の個人差）し、
- 観測ノイズと **batch effect** も別途加えて identifiability の難しさを再現する。

モデル側（`IndividualEncoder`）は genotype 特徴 + baseline pseudobulk を入力に
$z_{\text{indiv}}$ を推論し、FiLM でこの個人差を復元できるかを評価する。
