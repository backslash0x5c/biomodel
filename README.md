# biomodel — 個人差を予測する摂動応答基盤モデル

scGPT のような単一細胞基盤モデル（single-cell foundation model）を事前学習から構築し、
その上に **「同じ薬でも人によって効果が違う」= 個人差（inter-individual variability）を
予測する摂動応答（perturbation response）モジュール** を載せるための設計とプロトタイプです。

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
| `src/biomodel/` | PyTorch 実装（encoder / perturbation / FiLM 個人条件づけ / decoder） |
| `src/biomodel/simulate.py` | 個人差を持つ合成データ生成器（numpy のみで動作） |
| `scripts/run_demo.py` | 合成データで「事前学習→摂動学習→leave-one-donor 評価」を一気通貫で実行 |
| `tests/` | 形状・学習・個人差捕捉の最小テスト |

## クイックスタート

```bash
python -m pip install -r requirements.txt        # numpy（＋あれば torch）
python scripts/run_demo.py --epochs 30           # 合成データで一気通貫デモ
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

## ロードマップ（プロトタイプ → 実データ）

1. **合成データで概念実証**（本リポジトリ）— 個人差を「捕捉できる／できない」を
   population-average ベースラインとの差で定量化。
2. **公開データで事前学習** — CELLxGENE 等の大規模 scRNA でマスク発現予測。
3. **集団アトラス＋genotype** で個人差構造（eQTL）を学習（例: OneK1K）。
4. **細胞株摂動**（Tahoe-100M / sci-Plex / LINCS）で「薬→応答」を学習。
5. **患者由来データ**（ex vivo 薬剤スクリーニング / オルガノイド / PDX）で fine-tune。

詳細は `docs/` を参照してください。
