# HPO と PCF ランキングの関係分析

`analyze_hpo_pcf_relationship.py` は、NewData の 462 症例について、入力された **present HPO** と PCF の正解 OMIM ランキングの関係を集計する。

## 指標

- 入力 HPO 数、HPO 情報量（OMIM 注釈を重複排除した `-log2` 値）の総和・平均・最大値（最大値のHPO ID/ラベルも保存）
- 正解 OMIM の PCF 順位、PCF スコア、Recall@1/3/5/10/30
- 正解候補に付与された matched HPO と入力 HPO の一致数・カバレッジ
- IC 総和・IC 平均の四分位別 Recall
- Spearman 相関と散布図

OMIM の照合は疾患名ではなく正規化した OMIM ID の完全一致で行う。PCF の順位が 30 位以内にない症例は、順位分布では `31 (>30/not returned)` として扱う。全症例を分母に含め、PCF 出力のエラー・欠損も `pcf_status` に残す。

## 実行

プロジェクトルートから次を実行する。

```bash
MPLCONFIGDIR=/tmp/zebraseek-mpl .venv/bin/python -u \
  tryToolsForNewData/hpo_pcf_relationship/analyze_hpo_pcf_relationship.py
```

入力 TSV、PCF JSON、HPO 情報量ファイル、出力先は `--tsv`、`--pcf-dir`、`--ic-file`、`--output-dir` で変更できる。

## 主な出力

- `per_case_metrics.csv/json`: 症例単位の集計
- `candidate_metrics_top30.csv`: PCF 上位 30 候補と入力 HPO の一致
- `recall_by_information_bin.csv`: IC 総和・平均・最大値の四分位別 Recall
- `correlation_summary.csv`: 指標間の Spearman 相関
- `rank_distribution.png`: 正解疾患の順位分布
- `information_vs_pcf.png`: 情報量と順位・PCF スコアの散布図
- `recall_by_information_bin.png`: 四分位別 Recall
- `summary.json`: 全体集計

## 正解候補スコアの追加分析

`analyze_truth_score_distributions.py` は、正解OMIMの全ランキング上の値を抽出する。Top30外の正解についても、全ランキングに存在する場合はスコアを保持する。PCFは正解OMIMが全ランキングに存在しない症例があるため、`truth_found_full` とスコア欠損数を確認する。

```bash
MPLCONFIGDIR=/tmp/zebraseek-mpl .venv/bin/python -u \
  tryToolsForNewData/hpo_pcf_relationship/analyze_truth_score_distributions.py
```

- `truth_tool_metrics.csv/json`: 症例別の正解PCF/GM順位・スコア
- `truth_score_distribution_summary.csv`: 正解候補の分布統計
- `truth_score_by_recall30.csv`: Top30内外の分布統計とMann–Whitney U検定
- `truth_score_distributions.png`: 全体分布
- `truth_score_by_recall30.png`: Top30内外の比較
- `truth_score_summary.json`: 件数と欠損状況

## 動的な確認候補数の分析

`analyze_dynamic_cutoffs.py` は、スコア閾値を候補数のカットオフとして評価する。各閾値について、正解OMIMを含む割合（Recall）、閾値を通過する平均・中央値・95パーセンタイル候補数、候補集合のmicro-precisionを出力し、固定Top-kと比較する。

```bash
MPLCONFIGDIR=/tmp/zebraseek-mpl .venv/bin/python -u \
  tryToolsForNewData/hpo_pcf_relationship/analyze_dynamic_cutoffs.py
```

- `dynamic_cutoff_summary.csv`: 閾値別のRecallと候補数
- `dynamic_cutoff_per_case.csv`: 症例別の閾値通過候補数
- `fixed_topk_summary.csv`: Top1/3/5/10/30との比較
- `dynamic_cutoff_recall_workload.png`: Recallと候補数の曲線
- `dynamic_cutoff_threshold_curves.png`: 閾値・Recall・候補数の曲線
- `dynamic_cutoff_summary.json`: ランキングの単調性など

生成された CSV/JSON/PNG は `.gitignore` で除外し、解析スクリプトだけを追跡対象にしている。
