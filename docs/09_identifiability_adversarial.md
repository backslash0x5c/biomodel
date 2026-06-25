# 09. identifiability — 真の個人差 vs batch effect の分離

ドナー差には **真の生物学的個人差** と **技術的 batch effect**（実験日・試薬ロット・
プラットフォーム）が交絡する。これを分離できないと「個人差を当てた」が実は batch を
当てているだけ、になりうる（docs/02 §4）。ここでは 2 つの相補的な対策を実装・検証する。

## 1. 第一の梃子（推奨）: genotype 条件づけ

genotype は batch と独立。個人差を genotype 由来で説明できれば、それは batch ではない。
本リポジトリの提案モデルは genotype を個人 encoder に入れており、これが最も信頼できる
identifiability の梃子。**まず genotype を入れる**こと。

## 2. 第二の梃子（補完）: batch 敵対学習（DANN）

`src/biomodel/adversarial.py`。勾配反転層（GRL）で encoder の細胞表現が batch を
予測できないように学習し、表現を **batch 不変** にする。

```
delta 損失（個人差予測）  +  adv_weight · CE(Disc(GRL(encode(cell))), batch)
```

- **cell-level** で敵対をかける（probe と整合し、強い batch 信号に十分な勾配を与える）。
- 安定化（DANN は不安定になりやすい）: 判別器は高 lr（`disc_lr_mult`）、モデル側は
  勾配クリップ（`grad_clip`）、`grl_lambda` を学習進行でランプ。

### 診断: batch probe

`batch_probe_accuracy`: 凍結した encoder の細胞表現から batch を当てる **線形 probe** の
検証精度。高いほど表現が batch と絡む（entangled）。chance = 1/n_batches に近いほど不変。

## 3. 実測（`scripts/run_demo_adversarial.py`, leave-one-donor-out）

個人差を genotype 由来にし、baseline に batch effect を載せた設定（batch と個人差が
分離可能）:

```
手法                 batch_probe精度   indiv_R2   rank_rho
標準学習 (adv=0)            1.000       +0.80      +0.81
敵対学習 (adv=1.0)          0.84        +0.36      +0.48
population-mean              -          0.00       0.00
       (batch chance = 0.25)
```

**正直な評価**:
- 敵対学習は batch の線形分離性を **明確に下げる**（1.00 → 0.84）= identifiability の改善。
- ただし **chance(0.25) までは落ちない**。強い加法的 batch shift の完全除去は難しい。
- **個人差予測とのトレードオフがある**（indiv_R2 0.80 → 0.36）。敵対圧は encoder を
  不安定化させ、共有表現を傷める。`adv_weight` で強さを調整（小さくすると probe 低下も
  小さいが indiv 保持）。
- 小スケール・短時間では効果が出ないこともある（DANN の不安定性）。

→ 結論: **genotype 条件づけを主軸**にし、batch 敵対学習は補完的に、トレードオフを見ながら
弱めに使う。batch を当てているだけでないかは probe で常に監視する。

## 4. 設計上の注意

- **実験計画で交絡を壊す**のが最善（同一バッチに複数ドナー、複数バッチに同一ドナー）。
- batch と donor が完全に交絡していると、いかなる手法でも分離不能。
- null 摂動で偽の個人差が出ないか、既知 PGx で方向が合うかを sanity check（docs/04 §5）。
- 予測された個人差と batch の相関を監視（batch leakage の検出）。

## 5. 関連 API

| 関数/クラス | 役割 |
|---|---|
| `grad_reverse`, `BatchDiscriminator` | GRL と batch 判別器（model.py） |
| `train_supervised_adversarial` | delta 回帰 ＋ batch 敵対（adversarial.py） |
| `batch_probe_accuracy` | batch entanglement 診断（線形 probe） |
| `AdvConfig` | `adv_weight` / `grl_lambda` / `disc_lr_mult` / `grad_clip` |
