# 04. 評価プロトコル — 「個人差が当たるか」を測る

最大の落とし穴は「**集団平均（ATE）が当たっているだけ**」を「個人差が当たった」と
誤認すること。評価は最初に設計する。

## 1. データ分割（generalization の軸）

| 分割 | 問い | 難易度 |
|---|---|---|
| **leave-one-donor-out (LODO)** | 未知の人の応答を当てられるか（**本命**） | 高 |
| leave-one-perturbation-out | 未知の薬の応答を当てられるか | 高 |
| leave-one-(donor×pert)-out | 未知の人×未知の薬（二重外挿） | 最高 |
| i.i.d. random split | 基本性能の確認 | 低 |

LODO では、テストドナーの **投与前 scRNA と genotype のみ**を与え、
投与後（各摂動への応答）を予測させる。学習にそのドナーの応答は一切使わない。

## 2. 指標：平均を消して「個人差成分」を測る

予測 $\hat{\Delta}_{d,p}$、真値 $\Delta_{d,p}$。

### 2.1 基本（全体精度）
- **Delta MSE / Pearson $r$**: $\hat{\Delta}$ vs $\Delta$（DEG に絞った $r$ も併記）。
- **分布距離**: unpaired なら MMD / Wasserstein（予測分布 vs 実分布）。

### 2.2 個人差成分（本命）
集団平均 $\bar{\Delta}_p = \mathbb{E}_d[\Delta_{d,p}]$ を引いた残差で評価する:

$$\tilde{\Delta}_{d,p} = \Delta_{d,p} - \bar{\Delta}_p,\qquad
  \hat{\tilde{\Delta}}_{d,p} = \hat{\Delta}_{d,p} - \bar{\hat{\Delta}}_p$$

- **個人差説明率**: $R^2$ of $\hat{\tilde{\Delta}}$ vs $\tilde{\Delta}$。
  これが正でないと「個人差を当てている」とは言えない（平均だけ当てても 0 近辺）。
- **応答ランキング相関**: 薬 $p$ ごとに「どのドナーで最も効くか」の
  Spearman $\rho$（ドナー横断）。個別化医療で本当に効く指標。

### 2.3 較正（calibration）
- 予測の不確実性を出すなら、個人レベルで予測区間の被覆率を測る。

## 3. 必須ベースライン（これらに勝てなければ意味がない）

| ベースライン | 内容 | これに勝つ意味 |
|---|---|---|
| **Population-mean** | 全ドナー平均の応答 $\bar{\Delta}_p$ を全員に適用 | **個人差を全く使わない**。本命の対照 |
| Control-copy | 「効果なし」$\hat{\Delta}=0$ | 自明な下限 |
| Nearest-donor | baseline が最も近い学習ドナーの応答をコピー | 単純な個別化 |
| No-genotype ablation | genotype を抜いた自モデル | genotype の寄与の検証 |

> **核心**: モデルが Population-mean に対して **2.2 の個人差説明率／ランキング相関**で
> 有意に上回って初めて「個人差を予測できた」と言える。本リポジトリのデモは
> まさにこの比較を出力する（`scripts/run_demo.py`）。

## 4. ablation（どの要素が効くか）

- genotype あり / なし
- 相互作用 $\Phi$: FiLM / cross-attn / hypernet / **加法（CPA 相当）**
  → 加法に対する相互作用の優位性が、本設計の主張の検証。
- 事前学習あり / なし（MGM の寄与）
- baseline 集約: pseudobulk / attention pooling

### 4.5 注記: ドナー数・baseline 漏れ込み・線形ベースライン（プロトタイプで観測）

合成データ実験（`scripts/run_demo.py`）から得た、実務でも効く教訓:

1. **ドナー数が個人差学習の律速**。FiLM 等の神経モデルが genotype × 摂動の相互作用を
   学ぶには十分なドナー数が要る。本プロトタイプでは train ~30 ドナーでは未学習
   （個人差 $R^2<0$）だが、~110 ドナーで $R^2\approx0.9$ に跳ね上がる。
   → 「個人差を当てたいなら、まず**多ドナーのデータ**を確保せよ」。
2. **baseline 経由の漏れ込みに注意**。ベースライン状態が genotype を強く反映する設定では、
   genotype 入力を切っても baseline から個人差が部分的に復元され、ablation の差が
   見かけ上小さくなる。genotype の正味の寄与を測るには、baseline と独立に genotype が
   持つ情報で評価する（`genotype_drives_baseline` を下げた条件）。
3. **線形 ridge は強いベースライン**。本合成データの個人差は genotype に線形なので、
   摂動ごとの ridge 回帰（`scripts/run_demo_numpy.py`）が $R^2\approx1.0$ を出す。
   神経モデル＋基盤モデルの真価は、**非線形・摂動間の共有構造・高次元 genotype で
   per-perturbation ridge が過学習する領域**で初めて現れる。実データ評価では
   **線形 ridge を必ずベースラインに加える**こと。

## 5. sanity check（因果的妥当性）

- **null 摂動**で偽の個人差が出ないか（出たら batch を拾っている疑い）。
- **既知 PGx**（例: CYP2D6 PM/EM で代謝が変わる薬）で、予測された個人差が
  既知の方向と一致するか。
- batch を入れ替えた **counterfactual** で予測が不変か（batch 不変性）。

## 6. プロトタイプの評価出力

`src/biomodel/evaluate.py` は LODO で以下を出力:
- Delta Pearson $r$（全体）
- **個人差説明率 $R^2$**（本命）
- **応答ランキング Spearman $\rho$**
- Population-mean ベースラインとの比較表
