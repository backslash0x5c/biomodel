# biomodel — 個人差を予測する摂動応答基盤モデル

scGPT のような単一細胞基盤モデル（single-cell foundation model）を事前学習から構築し、
その上に **「同じ薬でも人によって効果が違う」= 個人差（inter-individual variability）を
予測する摂動応答（perturbation response）モジュール** を載せるための設計とプロトタイプです。

## ブランチ構成（スタック PR・順序関係）

本リポジトリは **3 段のスタック PR** で開発しており、各ブランチは下の段を土台に積み増します
（依存順 `main ← #1 ← #2 ← #3`）。**最新・全部入りは PR #3**。下から順（#1→#2→#3）に
マージすると、各 PR のベースが自動で `main` 側へ繰り上がります。

| 段 | ブランチ | PR | この段で増える内容 |
|---|---|---|---|
| ① 土台 | `claude/nifty-thompson-ih5ove` | #1 | 設計＋本体プロトタイプ（基盤モデル＋個人差モジュール） |
| ② | `claude/clinical-usecases` | #2 | 専門外向け概要（`docs/00`）＋精密医療デモ＋結果図 |
| ③ 先端 | `claude/advanced-extensions` | #3 | 実用化4拡張（不確実性／実データ検証／用量・併用／解釈性, `docs/11`） |

📍 **このREADMEは ① 土台 `claude/nifty-thompson-ih5ove`（PR #1）のものです**（②③ の追加分は未収録）。

中心となる発想は次の一文に集約されます。

> 摂動効果を「個人という文脈に条件づけた異質処置効果（heterogeneous / conditional
> treatment effect, CATE）」として定式化し、事前学習で得たベースライン表現と
> 遺伝型（genotype）の上で個別化された応答を予測する。

```
予測:  x_treated = Decoder( z_cell ,  Φ(z_pert , z_indiv) )

z_cell  : 投与前の細胞 / ドナー状態       （事前学習 encoder）
z_pert  : 薬剤・遺伝子摂動の表現          （化合物 or 遺伝子 embedding）
z_indiv : 個人 embedding                   （genotype + baseline + 共変量）
Φ(...)  : 摂動 × 個人 の相互作用モジュール  ← 個人差の本体（FiLM / cross-attn / hypernet）
```

## このリポジトリの構成

| パス | 内容 |
|---|---|
| `docs/01_design.md` | 全体アーキテクチャ設計（事前学習→摂動→個人条件づけの3段構え） |
| `docs/02_individual_conditioning.md` | 個人 embedding の作り方の深掘り（genotype / eQTL / PGx の入れ方） |
| `docs/03_related_work.md` | scGPT / STATE / CPA / GEARS 等との比較と部品の使い分け |
| `docs/04_evaluation.md` | leave-one-donor-out など「個人差が当たるか」を測る評価プロトコル |
| `docs/05_datasets.md` | データセット候補と「多ドナー×薬剤×single-cell」希少問題への戦略 |
| `docs/06_celllevel_and_mmd.md` | cell-level 拡張と分布マッチング（unpaired・MMD） |
| `docs/07_realdata_pipeline.md` | 実データ前処理パイプライン（OneK1K 等） |
| `docs/08_interaction_benchmark.md` | 相互作用 Φ（FiLM/cross-attn/hypernet）の比較ベンチマーク |
| `docs/09_identifiability_adversarial.md` | identifiability（genotype 条件づけ＋batch 敵対学習） |
| `docs/10_onek1k_acquisition.md` | 実 OneK1K の取得手順（GSE196830）＋ GReX 特徴 |
| `src/biomodel/model.py` | PyTorch 実装（encoder / perturbation / 個人条件づけ / 相互作用 / decoder / GRL） |
| `src/biomodel/simulate.py` | 個人差を持つ合成データ生成器（線形/非線形・処置後細胞・観測マスク, numpy のみ） |
| `src/biomodel/losses.py` | MMD / energy distance（unpaired 分布マッチング） |
| `src/biomodel/cell_level.py` | cell-level + MMD 学習・評価 |
| `src/biomodel/adversarial.py` | batch 敵対学習（identifiability）＋ batch probe 診断 |
| `src/biomodel/data_pipeline.py` | 実データ前処理（生 scRNA+genotype → モデル入力）＋ anndata/GReX/fake ローダ |
| `scripts/run_demo.py` | 「事前学習→摂動学習→leave-one-donor 評価」を一気通貫（`--observed-rate` で疎観測） |
| `scripts/benchmark_interactions.py` | Φ の比較（線形/非線形 × additive/FiLM/cross-attn/hypernet/ridge） |
| `scripts/run_demo_celllevel.py` | cell-level + MMD / Sinkhorn デモ（`--loss sinkhorn`） |
| `scripts/run_demo_adversarial.py` | batch 敵対学習（identifiability）デモ |
| `scripts/preprocess_onek1k.py` | fake OneK1K で前処理→モデル接続（`--genotype-features grex`） |
| `scripts/fetch_onek1k.py` | 実 OneK1K（GSE196830）取得手順＋ローカル検証 |
| `tests/` | 形状・学習・個人差・損失(MMD/Sinkhorn)・前処理・GReX・敵対・anndata のテスト（34 件） |

## クイックスタート

```bash
python -m pip install -r requirements.txt        # numpy（＋あれば torch）
python scripts/run_demo.py                        # 合成データで一気通貫デモ
python scripts/run_demo.py --observed-rate 0.5    # 疎な観測（未観測ペアは学習から除外）
python scripts/benchmark_interactions.py          # Φ の比較ベンチマーク
python scripts/run_demo_celllevel.py --quick --loss sinkhorn   # cell-level + Sinkhorn(OT)
python scripts/run_demo_adversarial.py            # batch 敵対学習（identifiability）デモ
python scripts/preprocess_onek1k.py --genotype-features grex   # 前処理 + GReX 特徴
python scripts/fetch_onek1k.py                    # 実 OneK1K 取得手順（dry-run）
```

torch が無い環境でも、`simulate.py` と numpy 参照実装による
最小デモ（`scripts/run_demo_numpy.py`）が動くようにしてあります。

## デモ結果（参考・leave-one-donor-out）

合成データ（個人差 = genotype × 摂動の相互作用）で、**未知ドナー**の処置効果を予測:

| モデル | overall_r | **indiv_R2**（個人差成分） | rank_rho |
|---|---:|---:|---:|
| **FiLM + genotype（提案）** | +0.975 | **+0.907** | +0.819 |
| population-mean（個人差を使わない対照） | +0.710 | −0.000 | +0.000 |
| additive Φ（ablation: CPA 相当） | +0.717 | −0.055 | +0.295 |
| FiLM, genotype なし（ablation） | +0.736 | −0.027 | +0.317 |

`indiv_R2` は集団平均（ATE）を引いた**個人差成分**の説明率。提案モデルだけが
population-mean を超えて個人差を捉える。加法 Φ や genotype 落としでは個人差が出ない
＝「相互作用」と「genotype」が本質という ablation。

> 重要な注意（`docs/04` §4.5）: 個人差の学習には**十分なドナー数**が要る
> （train ~30 では未学習、~110 で $R^2\approx0.9$）。また線形合成データでは
> per-perturbation の線形 ridge が強力なベースラインになる。実データ評価では
> 線形 ridge を必ず対照に入れること。

## 発展：3 つの拡張と主要知見

### (1) 相互作用 Φ の比較（`docs/08`, `benchmark_interactions.py`）
additive(CPA相当)/FiLM/cross-attention/hypernetwork ＋ 線形 ridge を、線形/非線形の
個人差レジームで leave-one-donor-out 比較。
- **additive は両レジームで失敗**（相互作用なしでは個人差を表現不可）。
- **線形個人差**: ridge が最良（1.0）、hypernet(0.98)・FiLM(0.88) が続く。単純なら深層は不要。
- **非線形個人差 ＋ 十分なドナー**: 神経モジュールが ridge を圧倒（train 900 で
  hypernet 0.90 / FiLM 0.88 vs ridge 0.16）。少データでは全手法が崩れる（データ飢餓）。
- 教訓: 万能の Φ は無い。**ridge を必ずベースラインに**、複雑な個人差には表現力＋ドナー数。

### (2) cell-level + 分布マッチング（`docs/06`, `run_demo_celllevel.py`）
scRNA は破壊的測定で control/treated が **unpaired**。pseudobulk ではなく細胞分布を
予測し、予測細胞群と観測細胞群の **MMD** を最小化。leave-one-donor-out で個人差
（indiv_R2≈0.74）と分布レベル精度（energy distance）を同時評価。

### (3) 実データ前処理パイプライン（`docs/07`, `preprocess_onek1k.py`）
生 scRNA(カウント)＋genotype → 正規化 → HVG → pseudobulk/delta → genotype 特徴
（PCA/eQTL/PGx 差し替え可）→ SimData 互換 → 既存モデルに接続。`load_anndata`（.h5ad）＋
`align_genotype`（donor 整列）で実データに接続でき、`load_fake_onek1k` で実データ無しでも
end-to-end 実行（genotype は機微情報のため実データは扱わない）。

## さらなる拡張（identifiability / 疎観測 / 実データ I/O）

### (4) 疎な観測への対応（`SimConfig.observed_rate`, `run_demo.py --observed-rate`）
実データは (donor × pert) が疎。未観測ペアを学習から自動除外。観測率 0.5 でも
個人差を捕捉（indiv_R2≈0.41）。

### (5) batch 敵対学習で identifiability（`docs/09`, `run_demo_adversarial.py`）
真の個人差と batch effect の分離。GRL による batch 敵対学習で表現を batch 不変に近づける
（batch probe 精度 1.00→0.84）。**ただし完全には落ちず個人差予測とトレードオフ**があり、
**genotype 条件づけが主軸・敵対は補完**（正直な結論は `docs/09`）。

### (6) 実データ I/O（`load_anndata`, `align_genotype`）
AnnData(.h5ad) から `RawScRNA` を構築し、genotype を donor 順に整列。実 anndata での
ラウンドトリップをテスト済み。

### (7) GReX 特徴（PrediXcan 流, `data_pipeline.GReXFeaturizer`）
genotype を「遺伝的に制御された発現」$\text{GReX}_j=\sum_s w_{js}\,\text{dosage}_s$ に変換。
PCA フォールバックより eQTL に忠実。`--genotype-features grex` で前処理に適用。

### (8) Sinkhorn(最適輸送)損失（`losses.sinkhorn_divergence`）
cell-level の分布マッチングを MMD と Sinkhorn から選択（`--loss sinkhorn`）。
debiased・log 安定化・微分可能。合成データで MMD と同程度に個人差を捕捉。

### (9) 実 OneK1K 取得（`scripts/fetch_onek1k.py`, `docs/10`）
OneK1K（GSE196830; 982 ドナー・127万 PBMC・genotype）の取得手順とローカル検証。
無断 DL はせず、dbGaP/EGA の controlled-access パターンと倫理も明記。

## ロードマップ（プロトタイプ → 実データ）

1. **合成データで概念実証**（本リポジトリ）— 個人差を「捕捉できる／できない」を
   population-average ベースラインとの差で定量化。
2. **公開データで事前学習** — CELLxGENE 等の大規模 scRNA でマスク発現予測。
3. **集団アトラス＋genotype** で個人差構造（eQTL）を学習（例: OneK1K）。
4. **細胞株摂動**（Tahoe-100M / sci-Plex / LINCS）で「薬→応答」を学習。
5. **患者由来データ**（ex vivo 薬剤スクリーニング / オルガノイド / PDX）で fine-tune。

詳細は `docs/` を参照してください。
