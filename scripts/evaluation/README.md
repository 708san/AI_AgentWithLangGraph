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

Hit@kは明示された`rank`（候補統合では`best_rank`）で計算し、MRRは正解順位の逆数の平均です。出力・候補ID・重複・正規化前後の欠落はレポートへ残します。評価結果は推論精度の観測であり、臨床的な正しさを保証しません。
