# 01. アーキテクチャ設計

## 0. 目的と問題設定

**やりたいこと**: scGPT のような単一細胞基盤モデルを事前学習から作り、
摂動（薬剤・遺伝子ノックアウト等）への応答を予測する。
**かつ**「同じ薬を別の人に与えても効果が違う」という個人差を予測できるようにする。

**問題設定（形式化）**:

- ドナー $d$（個人）、細胞 $i$、摂動 $p$ を考える。
- 投与前状態 $x^{\text{ctrl}}_{d,i}$（遺伝子発現ベクトル）と、投与後状態 $x^{\text{pert}}_{d,i,p}$ がある。
- 個人ごとの応答 $\Delta_{d,p} = \mathbb{E}_i[x^{\text{pert}}_{d,i,p} - x^{\text{ctrl}}_{d,i}]$ は **ドナー $d$ に依存**する。
- 目標: 未知ドナー・未知摂動に対して $\Delta_{d,p}$（個別化された処置効果）を予測する。

これは因果推論でいう **条件付き平均処置効果 (CATE)** の推定そのものである:

$$\tau(x, p, d) = \mathbb{E}[Y(p) - Y(\text{ctrl}) \mid X = x,\ \text{donor} = d]$$

ここで「個人差」とは $\tau$ がドナー文脈（遺伝型・ベースライン状態）で変動すること。
平均処置効果 (ATE) しか当てられないモデルでは個人差は表現できない。

## 1. なぜ「加法モデル」では不十分か

CPA（Compositional Perturbation Autoencoder）流の素朴な定式化は

$$z^{\text{pert}} = z^{\text{ctrl}} + v_p$$

と、潜在空間で **摂動ベクトル $v_p$ を足す**。これは摂動効果が個人に依らず一定という仮定で、
個人差（$v_p$ がドナーで変わること）を構造的に表現できない。

本設計では **摂動 × 個人の相互作用 $\Phi(z_{\text{pert}}, z_{\text{indiv}})$ を明示的にモデル化**する。
$\Phi$ がドナー embedding $z_{\text{indiv}}$ で変調されることが、個人差の本体である。

## 2. 三段構えのアーキテクチャ

```
┌──────────────────────────────────────────────────────────────────┐
│ 段1: 事前学習 encoder（自己教師あり, scGPT/Geneformer 流）          │
│   入力: 投与前 scRNA（マスク発現予測で事前学習）                    │
│   出力: 細胞 embedding z_cell   →  集約して z_donor(baseline)       │
├──────────────────────────────────────────────────────────────────┤
│ 段2: 摂動 encoder                                                  │
│   薬剤 → 分子 GNN / 化学基盤モデル → z_pert                         │
│   遺伝子摂動 → 遺伝子 embedding（GEARS 流の遺伝子グラフ事前知識）   │
├──────────────────────────────────────────────────────────────────┤
│ 段3: 個人条件づけ ＋ 相互作用 Φ（個人差の本体）                    │
│   z_indiv = g(genotype, z_donor_baseline, covariates)              │
│   effect  = Φ(z_pert, z_indiv)     ← FiLM / cross-attn / hypernet  │
│   x_pred  = Decoder(z_cell, effect)                                │
└──────────────────────────────────────────────────────────────────┘
```

### 段1: 事前学習 encoder

- **目的**: ラベルなし大規模 scRNA から汎用的な細胞状態表現を獲得。
  個人差の多くは「投与前のトランスクリプトーム状態」に既にエンコードされている
  ので、ここが個人差予測の土台になる。
- **入力表現**: 遺伝子を token、発現量を value とする（scGPT 流の value binning、
  あるいは連続値の value encoder）。HVG（高変動遺伝子）パネル or 全遺伝子。
- **自己教師あり目的**: マスク発現予測（masked gene expression modeling, MGM）。
  一部の遺伝子の発現値をマスクし復元させる。
- **集約**: 細胞 embedding を donor 単位で集約（attention pooling / pseudobulk）し、
  `z_donor_baseline` を得る。これが個人のベースライン状態を表す。

> プロトタイプ（`src/biomodel/model.py` の `ExpressionEncoder`）では、
> value encoder + Transformer or MLP + MGM 事前学習を最小実装している。

### 段2: 摂動 encoder

- **化合物（薬剤）**: SMILES → 分子グラフ → GNN、あるいは事前学習化学モデル
  （MolFormer 系）で `z_pert`。**未知薬への外挿**が効くのが利点。
- **遺伝子摂動（CRISPR KO/KD など）**: 摂動対象遺伝子の embedding。GEARS のように
  遺伝子–遺伝子グラフ（GO / coexpression）を事前知識として与えると、
  **未観測の摂動への汎化**が効く。
- プロトタイプでは「摂動 ID → 学習埋め込み」＋「摂動特徴ベクトル → MLP」の両対応。

### 段3: 個人条件づけ ＋ 相互作用 $\Phi$（最重要）

個人 embedding:

$$z_{\text{indiv}} = g\big(\underbrace{e_{\text{geno}}}_{\text{遺伝型}},\ \underbrace{z_{\text{donor\_baseline}}}_{\text{ベースライン}},\ \underbrace{e_{\text{cov}}}_{\text{年齢・性別等}}\big)$$

相互作用 $\Phi$ の実装候補（重要度順、`docs/03_related_work.md` も参照）:

1. **FiLM 条件づけ**（既定）: $z_{\text{indiv}}$ から $(\gamma, \beta)$ を生成し、
   摂動効果を $\gamma \odot \text{effect} + \beta$ でスケール／シフト。軽量・安定。
2. **Cross-attention**: 薬剤トークンと個人トークンを相互注意。組み合わせ汎化に強い。
3. **Hypernetwork**: $z_{\text{indiv}}$ が摂動応答デコーダの重みを生成。
   「人ごとに違う応答関数」を最も直接に表現できるがデータを食う。

デコーダ:

$$\hat{x}^{\text{pert}} = x^{\text{ctrl}} + \text{Decoder}\big(z_{\text{cell}},\ \Phi(z_{\text{pert}}, z_{\text{indiv}})\big)$$

**残差（delta）予測**にするのがポイント。control からの差分を予測することで、
個人のベースラインを基準にした応答を学べる（同一ドナーの control/treated ペアが
あれば contrast を直接学習できる）。

## 3. 学習戦略

### 3.1 二段階学習

1. **事前学習**: 無摂動 scRNA で MGM。encoder を初期化。
2. **摂動学習**: (donor, perturbation, control, treated) の組で delta 予測を教師あり学習。
   理想は同一ドナーの control/treated ペア。なければ control 分布から反実仮想を構成
   （S-/T-/X-learner 的）。

### 3.2 損失関数

$$\mathcal{L} = \underbrace{\|\hat{\Delta} - \Delta\|^2}_{\text{再構成}}
  + \lambda_{\text{dist}}\, \mathrm{MMD}(\hat{P}^{\text{pert}}, P^{\text{pert}})
  + \lambda_{\text{adv}}\, \mathcal{L}_{\text{batch-adv}}$$

- 再構成: delta の MSE（または分布マッチング）。
- 分布損失: 細胞は対応がつかない（unpaired）ので、予測分布と実分布を MMD / OT で合わせる。
- batch 敵対損失: ドナー由来の **真の生物差** と **batch effect** を分離するため、
  batch 変数に対しては adversarial に不変化（ただし donor 生物差は残す）。
  → identifiability の核心。詳細は `docs/02_individual_conditioning.md`。

### 3.3 転移学習（データのボトルネック対策）

「多ドナー×薬剤×single-cell」は希少。以下で橋渡し（`docs/05_datasets.md`）:

- 細胞株摂動（Tahoe-100M / sci-Plex / LINCS）→ 段2「薬→応答」を学習。
- 集団アトラス＋genotype（OneK1K 等）→ 段3「個人差構造（eQTL）」を学習。
- 両者を $\Phi$ で結合し、少量の患者由来データで fine-tune。

## 4. 推論：個別化予測

未知ドナー $d^\star$ について:
1. $d^\star$ の投与前 scRNA から `z_donor_baseline` を計算。
2. （あれば）$d^\star$ の genotype から $e_{\text{geno}}$。
3. 候補薬剤群 $\{p\}$ について $\hat{\Delta}_{d^\star, p}$ を予測。
4. 「この患者で最も効く薬」を応答の大きさ／方向でランキング → 個別化医療への接続。

## 5. プロトタイプとの対応

| 設計要素 | 実装 |
|---|---|
| 段1 encoder + MGM | `src/biomodel/model.py: ExpressionEncoder`, `pretrain.py` |
| 段2 摂動 encoder | `model.py: PerturbationEncoder` |
| 段3 個人 encoder | `model.py: IndividualEncoder` |
| 相互作用 Φ | `model.py: FiLMInteraction`（既定）/ `CrossAttnInteraction` |
| デコーダ（delta予測） | `model.py: ResponseDecoder` |
| 個人差を持つ合成データ | `simulate.py`（genotype × 摂動の相互作用を埋め込む） |
| leave-one-donor 評価 | `evaluate.py`, `docs/04_evaluation.md` |

詳細な数式・データ・評価は各 `docs/` を参照。
