# 段階別評価

PR #11のv2フローを変更せず、Zero-shotの正規化前後、候補統合、暫定診断の正規化前後、最終診断を独立に保存・採点するためのCLIです。正解ラベルは採点にだけ使い、推論入力には渡しません。

```bash
python -m scripts.evaluation.run_zero_shot \
  --benchmark path/to/cases.json \
  --output-dir local_artifacts/evaluation_results/zero_shot_run

python -m scripts.evaluation.run_tentative \
  --benchmark path/to/cases.json \
  --output-dir local_artifacts/evaluation_results/v2_run \
  --repeats 3 --enable-log
```

`--dry-run`は入力形式と画像パスだけを検証し、外部APIを呼びません。実行結果は`local_artifacts/`や`log/`へ保存し、Gitへ追加しないでください。評価スクリプト自体とテストのみを管理対象にします。

ベンチマークは次の形式です。

```json
{
  "cases": [{
    "case_id": "example_1",
    "input": {
      "present_hpo_list": ["HP:0001250"],
      "absent_hpo_list": [],
      "onset": "Unknown",
      "sex": "unknown"
    },
    "expected_output": {"omim_id": "OMIM:123456"}
  }]
}
```

`evaluate`は保存済み予測だけを再採点します。

```bash
python -m scripts.evaluation.evaluate \
  --cases path/to/cases.json \
  --predictions-dir path/to/repeat_001/predictions \
  --output path/to/report.json
```

## Zero-shot model/reasoning comparison

The current Zero-shot prompt and legacy 74-case inputs can be compared across
both GPT-5 models and three reasoning efforts without running the rest of the
diagnosis graph:

```bash
.venv/bin/python -m scripts.evaluation.run_zeroshot_model_comparison
```

The default conditions are `gpt-5-2/gpt-5-4` × `none/medium/high`. Results are
stored under `testCode/phenopacket_image_case/runs/zeroshot_model_reasoning_comparison_74/`.
All incomplete conditions are placed in one global queue and run in parallel
by default (`--max-parallel-conditions 6`), so a worker immediately starts the
next patient/condition when a faster request finishes. Reduce this value if
the Azure deployment rate-limits concurrent requests.
Each patient/condition has its own `status.json`, `zero_shot.json`,
`reasoned_output.json`, input snapshot, and prompt snapshot. Completed
conditions are skipped on rerun, so interruption or transient API failures can
resume with the same command. Use `--force` to regenerate completed
conditions, `--dry-run` to materialize inputs/prompts without API calls, and
`--max-cases 1` for a smoke test. `comparison_summary.csv` contains each
condition's predictions and gold ranks; `evaluation_summary.csv` contains
exact and MONDO close-match Hit@1/3/5 metrics.
While a condition is running, the terminal prints its elapsed time every 30
seconds. Each condition also writes `timing.json`, and the experiment root
writes `timing_summary.json` after every completed case so timing data survives
interruption.

Hit@kは明示された`rank`（候補統合では`best_rank`）で計算し、MRRは正解順位の逆数の平均です。出力・候補ID・重複・正規化前後の欠落はレポートへ残します。評価結果は推論精度の観測であり、臨床的な正しさを保証しません。

保存済み比較結果のオフライン評価とグラフ生成は、次のコマンドで行います。こちらは保存済みのIDと順位を使い、APIは呼び出しません。

```bash
.venv/bin/python -m scripts.evaluation.report_zeroshot_model_comparison
```

既定では比較結果ディレクトリの`analysis/`に、Markdownレポート、Top-1〜5のMATCHとCLOSE MATCHを積み上げ表示する図、指標表、条件別CSVを出力します。別の保存先は`--run-dir`と`--output-dir`で指定できます。
症例別の`case_comparison.png`と`.svg`、`case_comparison.csv`も出力します。画像は患者情報・正解疾患・HPO数を左側に、6条件の正解順位を色分けして並べる一覧です。CSVは1症例1行で、6条件の各セルにM（MATCH順位）、C（CLOSE MATCHのみの順位）、または`-`を記録します。

候補名からOMIM IDを補い、ZeroShotの除外・連番再採番をZebraSeek runtimeと同じ規則で適用して評価する場合は、次を実行します。これは未キャッシュの病名ごとに、runtimeの`disease_normalize`と同じAzure embedding検索を呼びます。検索結果は解析ディレクトリのキャッシュに保存され、再実行時に再利用されます。結果は既存の`analysis/`を上書きせず、`analysis_zebraseek_resolved/`へ保存します。

```bash
.venv/bin/python -m scripts.evaluation.report_zeroshot_model_comparison \
  --run-dir testCode/phenopacket_image_case/runs/zeroshot_model_reasoning_comparison_74 \
  --resolve-missing-ids
```

`zeroshot_id_resolution_audit.csv`には候補ごとの元ID、補完ID、解決状態、除外有無、再採番後順位を記録します。未解決候補はZebraSeek runtime同様に除外し、残った候補の順位を1から振り直します。MATCH/CLOSE MATCHは`MondoOntologyMatcher`の判定に基づきます。
