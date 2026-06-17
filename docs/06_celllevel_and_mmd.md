# 06. cell-level 拡張と分布マッチング（unpaired）

`docs/01` のプロトタイプは pseudobulk（ドナー×摂動の平均 delta）を予測した。実データの
single-cell 摂動では、**control 細胞と処置後細胞の対応がつかない**（unpaired）。同じ細胞の
「投与前／投与後」は観測できないからだ。そこで点ごとの誤差ではなく **分布間距離** を最小化する。

## 1. なぜ unpaired か / なぜ分布マッチングか

- scRNA-seq は破壊的測定。細胞 $i$ の control と treated を同時に観ることはできない。
- よって「予測した処置後細胞の集合 $\hat{P}^{\text{pert}}$」と「観測した処置後細胞の集合
  $P^{\text{pert}}$」を **分布として** 一致させる。
- 平均だけ合わせる（pseudobulk）と、応答の **分散・多峰性・部分集団応答** を捨ててしまう。
  分布マッチングはこれらを保てる。

## 2. モデル（cell-level）

`src/biomodel/model.py: CellLevelResponseModel`:

$$\hat{x}^{\text{pert}}_{i} = x^{\text{ctrl}}_{i} + \text{Decoder}\big(z_{\text{cell}}(x^{\text{ctrl}}_i),\ \Phi(z_{\text{pert}}, z_{\text{indiv}})\big)$$

- $z_{\text{cell}}$ は**細胞ごと**に encoder で計算（細胞の異質性を保つ）。
- $z_{\text{indiv}} = g(\text{genotype}, z_{\text{donor\_baseline}})$ は**ドナー単位**で計算し、
  同ドナー・同摂動の細胞群に broadcast。
- 効果 $\Phi(z_{\text{pert}}, z_{\text{indiv}})$ は `docs/01` 段3 と同じ（FiLM 等）。

## 3. 損失：MMD（`src/biomodel/losses.py`）

多帯域ガウシアンカーネルの最大平均ズレ（MMD$^2$）の不偏推定を最小化:

$$\mathrm{MMD}^2(\hat{P}, P) = \mathbb{E}_{x,x'}[k(x,x')] + \mathbb{E}_{y,y'}[k(y,y')] - 2\,\mathbb{E}_{x,y}[k(x,y)]$$

- $x \sim \hat{P}^{\text{pert}}$（予測処置後細胞）、$y \sim P^{\text{pert}}$（観測処置後細胞）。
- バンド幅は中央値ヒューリスティックで正規化し複数スケールを合算（スケール頑健）。
- 代替: エネルギー距離 / Sinkhorn(最適輸送)。評価には `energy_distance` を用意。

## 4. 学習（`src/biomodel/cell_level.py`）

各 (donor, pert) について control 細胞群と処置後細胞群を **独立に**（unpaired）サンプルし、
予測処置後細胞群との MMD を最小化する。MGM 事前学習（段1）は control 細胞で先に行う。

## 5. 評価（leave-one-donor-out）

`scripts/run_demo_celllevel.py` の出力:
- **pseudobulk 精度**: 予測細胞の平均から delta を集約し、`docs/04` の指標
  （個人差説明率 indiv_R2, ランキング相関）を population-mean と比較。
- **分布レベル精度**: 予測細胞分布 vs 観測細胞分布のエネルギー距離（小さいほど良い）。

### 実測（合成データ, leave-one-donor-out, 80 ドナー・25 epoch）

```
cell-MMD(FiLM+geno)   overall_r≈+0.92  indiv_R2≈+0.74  rank_rho≈+0.83   energy≈3.0
population-mean       overall_r≈+0.67  indiv_R2≈ 0.00  rank_rho≈ 0.00
```

unpaired・分布マッチング学習でも、population-mean を大きく超えて**個人差**を捉えられる
（細胞分布の情報を保ちつつ pseudobulk 精度も高い）。小規模（`--quick`）では indiv_R2 は
下がる（ドナー数・細胞数依存, docs/04 §4.5）。

## 6. 実データへの拡張メモ

- 細胞数がドナー間で不均衡 → ミニバッチで均す / 重み付け。
- batch effect は control・treated 双方に乗る → MMD 前に batch 条件づけ or 補正（docs/02 §4）。
- 大規模化: ミニバッチ MMD はバッチサイズに敏感。Sinkhorn や特徴空間での MMD を検討。
