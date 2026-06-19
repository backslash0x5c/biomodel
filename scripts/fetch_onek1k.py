#!/usr/bin/env python3
"""OneK1K データ取得の手順スクリプト（docs/10）。

OneK1K（Yazar et al., Science 2022; 982 ドナー・約127万 PBMC・14 免疫細胞型）の
scRNA-seq と genotype を取得し、本リポジトリの前処理パイプラインに接続するための
**手順とローカルファイル検証**を提供する。

重要:
  - 本スクリプトは既定で **dry-run**（手順の表示のみ）で、無断ダウンロードは行わない。
  - genotype は個人ゲノム情報（機微）。OneK1K は GEO(GSE196830) で配布されるが、多くの
    集団データは dbGaP/EGA の **アクセス制御**（申請・DAC 承認・IRB）が必要。利用規約と
    倫理審査を必ず遵守すること。
  - データが手元にある場合は --data-dir でファイル存在を検証し、--run で前処理を実行。

使い方:
    python scripts/fetch_onek1k.py                       # 取得手順を表示（dry-run）
    python scripts/fetch_onek1k.py --data-dir ./onek1k   # ローカルファイルを検証
    python scripts/fetch_onek1k.py --data-dir ./onek1k --run  # 前処理→モデル接続
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- OneK1K のデータアクセス情報（2022 論文時点; 最新は各ポータルで要確認）---
ONEK1K = {
    "paper": "Yazar et al., Science 2022 (doi:10.1126/science.abf3041, PMID 35389779)",
    "geo": "GSE196830",  # scRNA-seq ＋ array genotype の両方
    "portals": {
        "GEO (scRNA + genotype)": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196830",
        "CELLxGENE / Human Cell Atlas (processed)": "https://cellxgene.cziscience.com/",
        "OneK1K portal (browse eQTL)": "https://onek1k.org/",
    },
    "scrna": "1,267,758 PBMC / 982 donors / 14 免疫細胞型（10x 3' v2）",
    "genotype": ("Illumina Infinium Global Screening Array, 759,993 markers / 1,104 個体。"
                 "Michigan Imputation Server で 1000G phase III v5 を参照に補完"
                 "（Minimac4 + Eagle v2.4）。cis-eQTL 用に R²≥0.8 かつ MAF>0.05 の "
                 "5,849,361 SNP を保持。"),
}

# 前処理に必要な想定ローカルファイル（命名は取得時に合わせて調整）
EXPECTED_FILES = {
    "scrna_h5ad": "onek1k.h5ad",                 # load_anndata で読む単一細胞発現
    "genotype_dosage": "genotype_dosage.tsv",    # donor × variant のドーズ（0/1/2 or imputed）
    "grex_weights": "predixcan_weights.tsv",     # （任意）PrediXcan 風 GReX 重み
}


def print_plan() -> None:
    print("=" * 78)
    print("OneK1K データ取得手順（dry-run）")
    print("=" * 78)
    print(f"論文     : {ONEK1K['paper']}")
    print(f"scRNA    : {ONEK1K['scrna']}")
    print(f"genotype : {ONEK1K['genotype']}")
    print("\n[アクセス先]")
    for name, url in ONEK1K["portals"].items():
        print(f"  - {name}: {url}")
    print("\n[手順]")
    print("  1. GEO GSE196830 から scRNA-seq（カウント行列）と array genotype を取得。")
    print("     例: NCBI GEO の supplementary files、または HCA/CELLxGENE の .h5ad。")
    print("  2. genotype を QC・imputation（または配布済み imputed を使用）し、")
    print("     donor × variant のドーズ行列（TSV/plink/VCF）に整える。")
    print("  3. （任意）GReX 用に PrediXcan/FUSION の重み（GTEx 等）を用意（docs/02 §2）。")
    print("  4. ファイルを --data-dir に配置（想定名は下記）し、本スクリプトに --run。")
    print("\n[想定ファイル名（--data-dir 配下）]")
    for key, fn in EXPECTED_FILES.items():
        print(f"  - {key:16s}: {fn}")
    print("\n[アクセス制御・倫理]")
    print("  - OneK1K genotype は GEO 配布だが、個人ゲノムは機微情報。利用規約を遵守。")
    print("  - 他の集団データ（多くの臨床コホート）は dbGaP/EGA の controlled access が必要:")
    print("      dbGaP: データ利用申請（DUC）→ DAC 承認 → 暗号化ダウンロード（prefetch/sra）。")
    print("      EGA  : DAC へ申請 → EGA download client / pyega3 で取得。")
    print("  - 機関の IRB / データガバナンスに従うこと。")
    print("=" * 78)
    print("注: 本スクリプトは無断ダウンロードを行いません。取得は各規約に従い手動で。")


def verify_files(data_dir: Path) -> dict:
    print(f"\n[ローカルファイル検証] {data_dir}")
    status = {}
    for key, fn in EXPECTED_FILES.items():
        p = data_dir / fn
        ok = p.exists()
        status[key] = ok
        mark = "OK " if ok else "欠落"
        size = f"{p.stat().st_size/1e6:.1f}MB" if ok else "-"
        print(f"  [{mark}] {key:16s} {fn:28s} {size}")
    return status


def run_pipeline(data_dir: Path) -> None:
    """ローカルの実 OneK1K ファイルから前処理→モデル接続を実行（要 anndata/pandas）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np

    from biomodel.data_pipeline import (
        GenotypeFeaturizer,
        align_genotype,
        build_processed,
        load_anndata,
        processed_to_simdata,
    )

    h5ad = data_dir / EXPECTED_FILES["scrna_h5ad"]
    dosage_tsv = data_dir / EXPECTED_FILES["genotype_dosage"]
    if not h5ad.exists() or not dosage_tsv.exists():
        print("必要ファイルが不足しています。--data-dir の中身を確認してください。")
        return

    import pandas as pd
    print(f"\n[run] load_anndata({h5ad.name}) ...")
    # obs のカラム名は OneK1K の配布形式に合わせて調整すること
    raw, meta = load_anndata(str(h5ad), donor_key="individual", batch_key="pool",
                             perturbation_key="treatment", control_value="control")
    print(f"      cells={raw.counts.shape[0]} genes={raw.counts.shape[1]} "
          f"donors={len(meta['donor_ids'])}")

    # donor × variant のドーズを読み、donor_ids 順に整列
    df = pd.read_csv(dosage_tsv, sep="\t", index_col=0)   # index=donor_id, columns=variant
    dosage_by_donor = {str(d): df.loc[d].to_numpy(np.float32) for d in df.index}
    geno = align_genotype(meta["donor_ids"], dosage_by_donor, list(df.columns))

    proc = build_processed(raw, geno, n_hvg=2000, n_cells=128,
                           featurizer=GenotypeFeaturizer("pca", 32))
    print(f"[run] processed: donors={proc.n_donors} HVG={proc.n_genes} "
          f"perts={proc.n_perts} geno_feat={proc.geno_features.shape[1]}")
    _ = processed_to_simdata(proc)
    print("[run] processed_to_simdata 完了。train.py / cell_level.py で学習可能です。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="ローカルの OneK1K ファイル置き場（検証/実行に使用）")
    ap.add_argument("--run", action="store_true",
                    help="--data-dir のファイルで前処理→モデル接続を実行")
    args = ap.parse_args()

    print_plan()
    if args.data_dir:
        data_dir = Path(args.data_dir)
        status = verify_files(data_dir)
        if args.run:
            if status.get("scrna_h5ad") and status.get("genotype_dosage"):
                run_pipeline(data_dir)
            else:
                print("\n--run には scrna_h5ad と genotype_dosage が必要です。")


if __name__ == "__main__":
    main()
