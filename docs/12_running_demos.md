# 12. 実行ガイド（コマンドと出力例）

## まず大事な点：各デモは「独立した単体実行」

クイックスタートの各行は **1 本の連続パイプラインではありません**。それぞれが
**独立したスタンドアロンのデモ**で、実行するたびに「自前で合成データ生成 → モデル学習 →
結果表示」を最初から最後まで完結します。順序依存はなく、興味のあるものを 1 つずつ実行します。

- 共通の前提は `pip install -r requirements.txt` だけ。
- ほとんどのデモは PyTorch を使います（無い環境向けに numpy 参照実装の最小デモあり）。
- 4 拡張のデモ（§5〜§8）は `--quick` で軽量・短時間に実行できます。
- すべて**合成（架空）データ**の実演です（実在薬の効果を主張するものではありません）。

| コマンド | 何をするか | 目安 | quick | torch |
|---|---|---|---|---|
| `python scripts/run_demo.py` | 中核: 事前学習→摂動学習→leave-one-donor 評価 | ~1–2分 | – | 要 |
| `python scripts/demo_precision_medicine.py` | 🔰 未知患者に薬を推薦・responder 見分け | ~30秒 | – | 要 |
| `python scripts/demo_uncertainty.py` | ① 不確実性＋較正（deep ensemble） | ~2分(quick~10秒) | ✓ | 要 |
| `python scripts/demo_validation.py` | ② 実データ検証指標（top-1/regret/AUROC） | ~25秒(quick~10秒) | ✓ | 要 |
| `python scripts/demo_dose_response.py` | ③ 用量反応・組み合わせ | ~40秒(quick~15秒) | ✓ | 要 |
| `python scripts/demo_interpret.py` | ④ 解釈性・バイオマーカー | ~25秒(quick~10秒) | ✓ | 要 |
| `python scripts/benchmark_interactions.py` | 相互作用 Φ の比較（ridge 等） | ~数分 | – | 要 |
| `python scripts/run_demo_celllevel.py --quick --loss sinkhorn` | cell-level + MMD/Sinkhorn | ~30秒 | ✓ | 要 |
| `python scripts/run_demo_adversarial.py` | batch 敵対学習（identifiability） | ~1分 | – | 要 |
| `python scripts/preprocess_onek1k.py --genotype-features grex` | 実データ前処理＋GReX 特徴 | ~10秒 | – | 不要 |
| `python scripts/fetch_onek1k.py` | 実 OneK1K 取得手順（dry-run） | 即時 | – | 不要 |

> 数値はシード・設定で多少変わります。`--quick` は患者数・エポックを減らした軽量設定で、
> 数値は控えめに出ます（本来の代表値は `docs/11` のフル設定の結果を参照）。

---

## §1. 中核パイプライン `run_demo.py`

事前学習（MGM）→摂動応答学習→未知ドナーで評価。`--observed-rate 0.5` で疎観測も試せます。

```
$ python scripts/run_demo.py
...
結果（leave-one-donor-out, テストドナーで評価）
  FiLM+genotype(提案)      overall_r=+0.975  indiv_R2=+0.907  rank_rho=+0.819
  population-mean(対照)    overall_r=+0.710  indiv_R2=-0.000  rank_rho=+0.000
  additive(ablation)     overall_r=+0.717  indiv_R2=-0.055  rank_rho=+0.295
  FiLM,no-genotype(abl.) overall_r=+0.736  indiv_R2=-0.027  rank_rho=+0.317
=> 提案モデルは集団平均を超えて『個人差』を捉えている ✅
```

**読み方**: `indiv_R2`（集団平均を引いた個人差成分の説明率）が本命。提案だけが正で、
対照や ablation はほぼ 0＝「相互作用」と「genotype」が個人差の本質。

---

## §2. 🔰 精密医療デモ `demo_precision_medicine.py`

未知の患者に「どの薬が効くか」を予測（薬の推薦＋responder 見分け）。

```
$ python scripts/demo_precision_medicine.py
【1】患者ごとの『最も効く薬』の推薦（未知患者・モデル予測）
  集団平均だけで決めると、全員に同じ薬『D11』を薦めることになる。
    患者41   D11  →本当の最適 D11  ○
    患者10   D12  →本当の最適 D3   ×
  推薦が本当の最適薬と一致: 18/20 人
【2】薬『D1』に対する responder（効く人）の見分け
  予測スコア vs 真の効きの順位相関 (Spearman) = +0.99
```

**読み方**: 集団平均は全員同じ薬になるが、モデルは genotype を見て患者ごとに別の薬を推薦
（20人中18人的中）。responder 順位も Spearman +0.99 で当たる。

---

## §3. ① 不確実性＋較正 `demo_uncertainty.py`

「この予測をどれだけ信じてよいか」。複数モデルのばらつきを不確実性とし、較正で補正。

```
$ python scripts/demo_uncertainty.py --quick      # フルは引数なし（5モデル, 約2分）
【較正】予測区間の被覆率（nominal に近いほど信頼度が正しい）  分散スケール=2.73
   区間         生ensemble       較正後
   50%           20.6%     50.5%
   90%           46.6%     80.0%
   平均較正誤差: 生=0.394 -> 較正後=0.070
【患者ごとの信頼度】薬 D1 の予測効果 ± 不確実性:
   自信あり(低std): P87:+2.8±0.2 ...   自信なし(高std): P1:+3.0±1.4 ...
```

**読み方**: 生の分散は過信（50%区間が20%しか当たらない）→較正で nominal に一致。
不確実性が高い患者を「要追加検査」と切り分けられる。フル設定の代表値は `docs/11 §1`。

---

## §4. ② 実データ検証指標 `demo_validation.py`

ex vivo 薬剤感受性（BeatAML 等）を想定した leave-one-patient-out 評価。

```
$ python scripts/demo_validation.py --quick       # フルは引数なし
【検証指標】(top1_acc=推薦が真の最適と一致, regret=最適との差, AUROC=responder見分け)
   提案モデル        : top1_acc=0.600  regret=0.056  responder_AUROC=0.814
   集団平均(対照)    : top1_acc=0.350  regret=0.176  responder_AUROC=0.500
```

**読み方**: 提案が top-1 的中・regret・AUROC で対照（個人差なし）を上回る＝個別化の価値。
フル設定では top1≈0.90 / AUROC≈0.95（`docs/11 §2`）。

---

## §5. ③ 用量反応・組み合わせ `demo_dose_response.py`

yes/no でなく「適量」を当てる。必要量が genotype 依存（ワルファリン用量と同じ発想）。

```
$ python scripts/demo_dose_response.py --quick     # フルは引数なし
【個別化用量】薬 D1 で目標応答 0.5 に必要な用量（未知患者）:
   真の必要量の範囲: 0.32〜5.00 (患者で 15.5 倍の差)
   予測 vs 真の必要量: Pearson r=+0.99, 平均誤差=0.14
【組み合わせ】D1+D2 併用の相乗効果:
   相乗が最大の患者: P72 (相加1.20→併用2.03)
   拮抗（逆効果）の患者: P107 (相加1.39→併用0.71)
```

**読み方**: 必要量は患者間で十数倍の差があり、未知患者でも予測 r≈0.99。併用の相乗/拮抗も
患者別に切り分け。フル設定では必要量 約30倍差・r≈1.00（`docs/11 §3`）。

---

## §6. ④ 解釈性・バイオマーカー `demo_interpret.py`

「なぜ効くか」を genotype 特徴の勾配で抽出し、真の駆動因子を復元できるか検証。

```
$ python scripts/demo_interpret.py --quick         # フルは引数なし
【薬 D1 の応答を駆動する genotype 特徴】（勾配で抽出した重要度）
   特徴   抽出重要度   真の重要度
   g1         1.620       0.678
   g2         0.575       0.343
【検証】全薬で『抽出重要度 vs 真の駆動因子』の順位相関 平均 = +0.78
```

**読み方**: 抽出した重要度が真の駆動因子と高い順位相関（フル設定で ρ≈0.68, `docs/11 §4`）。
実データではこの上位特徴が候補バイオマーカー（コンパニオン診断）になる。

---

各機能の手法・背景は `docs/01`〜`docs/11`、専門外向けの動機は `docs/00` を参照。
