# 03. 関連研究と部品の使い分け

本設計は車輪の再発明ではなく、既存の (a) 細胞基盤モデル、(b) 摂動予測モデル、
(c) 因果推論手法 を組み合わせて「個人差」という軸を足すもの。各部品の役割を整理する。

## A. 単一細胞基盤モデル（段1 encoder の候補）

| モデル | 事前学習の要点 | 本設計での使い所 |
|---|---|---|
| **scGPT** | 遺伝子トークン＋発現 binning、生成的 attention で MGM | 段1 encoder の主候補。摂動 fine-tune の前例あり |
| **Geneformer** | 発現順位（rank-value encoding）で MLM、ネットワーク文脈 | 摂動・制御遺伝子の in-silico 摂動に強い |
| **scFoundation** | 連続値発現、非対称 encoder-decoder、5,000万細胞規模 | 大規模・read-depth 頑健な表現 |
| **UCE (Universal Cell Embedding)** | 種・組織横断のゼロショット埋め込み | ラベルなし統合・転移の土台 |
| **STATE（Arc Institute, 2025）** | 「state embedding」＋「state transition」で摂動応答を**細胞文脈をまたいで**予測 | **最も発想が近い**。文脈条件づけの設計を参照 |

> 段1 では scGPT / Geneformer 流の MGM を採用。STATE は「文脈（context）に条件づけた
> 摂動応答」を扱う点で本設計の段3と問題意識が重なるため、最初に精読すべき先行研究。

## B. 摂動予測モデル（段2＋段3 の候補）

| モデル | アイデア | 個人差への対応 | 本設計との関係 |
|---|---|---|---|
| **CPA** | 潜在空間で摂動を **加法ベクトル** $z+v_p$、用量・共変量を分解 | 共変量 embedding はあるが応答は基本加法 | 段3 を「加法→相互作用」に拡張する出発点 |
| **chemCPA** | CPA に化合物構造 encoder を追加、未知薬へ外挿 | 化合物側のみ | 段2 の化合物 encoder の参考 |
| **GEARS** | 遺伝子–遺伝子グラフ事前知識で **未観測摂動**へ汎化 | 摂動側の汎化 | 段2 の遺伝子摂動 encoder の参考 |
| **biolord** | 既知属性で潜在を **disentangle** | 属性として個人を分離可 | 個人属性の分離設計の参考 |
| **sams-vae** | スパースな加法機構で摂動を構造化 | — | 機構の解釈性 |
| **PerturbNet** | 摂動→応答の分布変換 | — | 分布マッチング損失の参考 |

**本設計の差分**: 上記の多くは摂動効果を「個人に依らず一定（加法）」と暗に仮定。
本設計は $\Phi(z_{\text{pert}}, z_{\text{indiv}})$ で **摂動 × 個人の相互作用**を明示し、
個人差（heterogeneous treatment effect）を一級市民として扱う。

## C. 因果推論（個人差＝異質処置効果の枠組み）

| 手法 | 役割 |
|---|---|
| **CATE 推定（S-/T-/X-learner, causal forest）** | 個人差＝条件付き処置効果。評価指標・反実仮想構成の枠組み |
| **CEVAE / GANITE** | 潜在交絡・反実仮想生成を深層で。unpaired 設定の参考 |
| **CINEMA-OT** | 最適輸送で摂動効果を因果的に分離（single-cell 向け） |
| **MMD / OT 分布マッチング** | 細胞が unpaired なときの損失設計 |

## D. 設計判断の要約

- **段1**: scGPT/Geneformer 流 MGM 事前学習（STATE の文脈条件づけを参照）。
- **段2**: 化合物は chemCPA 流の構造 encoder、遺伝子摂動は GEARS 流のグラフ事前知識。
- **段3（新規性）**: CPA の加法を **FiLM/cross-attn/hypernet による相互作用**へ拡張し、
  CATE の枠組みで個人差を推定・評価する。
- **identifiability**: CINEMA-OT / batch 敵対学習 / genotype 条件づけで
  「真の個人差」と「batch」を分離（`docs/02`）。

## E. ベンチマーク

- **PerturBench** 系のプロトコルで「未知摂動・未知細胞型」への汎化を測る枠組みを流用し、
  本設計では軸を **「未知ドナー（leave-one-donor-out）」** に拡張する（`docs/04`）。
