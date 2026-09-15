# Zero-shot・暫定診断のオフライン評価

推論から保存・採点までを実行する入口と、保存済み出力だけを採点する入口があります。

## ベンチマークから推論・保存・採点する

プロジェクトルートで、依存をインストールしたPython環境を使って実行します。

```bash
# Zero-shotだけを実行（他の検索ツール・Embedding・画像APIは呼ばない）
python -m scripts.evaluation.run_zero_shot \
  --benchmark local_artifacts/five_case_benchmark/data/five_cases.json \
  --model gpt-5-2 --repeats 3

# 入力から暫定診断の正規化までを実行
python -m scripts.evaluation.run_tentative \
  --benchmark local_artifacts/five_case_benchmark/data/five_cases.json \
  --model gpt-5-2 --repeats 3
```

`--dry-run` を付けるとケース・HPO入力と（暫定診断の場合）画像パスを確認するだけで、APIを呼びません。API認証・依存ライブラリ・検索インデックスの動作検証は含みません。指定モデルのAzure環境変数とHPO辞書が必要です。暫定診断にはさらに既存パイプラインのEmbedding設定・FAISSインデックス等が必要です。

- `--ks 1 3 5`: 採点する順位上限。
- `--use-absent-hpo`: 陰性所見を使用（既定は既存処理に合わせて無効）。
- `--output-dir PATH`: 新規実験ディレクトリ。既存ディレクトリは上書きせずエラー。
- `--no-images`: 暫定診断を画像なし条件で実行。
- `--image-root PATH`: ベンチマーク内の相対画像パスの基準。省略時は正解ファイルの親・その親・プロジェクトルートから一意に解決する。今回配布されたフォルダ構成は自動解決可能。

Zero-shot単体では `createZeroshot()` のみを呼び、HPOラベル変換後の入力で推論します。病名正規化も行わず `zeroShotRaw` を採点するため、LLMがOMIM IDを出さなければ病名が正しくてもID一致にはなりません。

暫定診断では本番のノードとエッジを再利用し、PubCaseFinder・任意の顔画像解析・Zero-shot・表現型検索・HPO Web検索・候補統合・暫定診断・病名正規化を実行します。PhenoBrainは本番既定どおり無効です。疾患別Wikipedia/PubMed検索、Reflection、最終診断は実行しません。上流の検索も毎回実行するので、その変動を含む評価です。

各実験の保存構成:

```text
<output-dir>/
  metadata.json                 # モデル、入力・コードのハッシュ、実行条件
  repeat_001/
    predictions/<case_id>.json  # 推論出力、プロンプト、入力、時間、エラー
    node_results/               # 暫定診断用の既存ノード保存結果
    evaluation.json             # 正解との照合と出力スナップショット
  repeat_002/...
```

暫定診断用は `zeroShotRaw` / `zeroShotResult` / `tentativeRaw` / `tentativeDiagnosis` の4段階を採点します。`tentativeRaw`はテキスト解析後・病名正規化前です（解析前のLLM応答テキストではありません）。ノード出力も随時保存し、後続で失敗しても完了した段階を残します。症例のエラー後は次の症例へ進み、失敗症例を含むレポートを保存します。初期化失敗は `initialization_error.json` に記録して終了します。

正解ラベルは採点のみで使用し、推論関数には患者情報をホワイトリストで渡します。`--repeats`は独立した繰り返し実行で、各回を別々に採点します。複数回の平均・分散の集計は現在含みません。

## 保存済み出力だけを採点する

以下の `evaluate` は標準ライブラリのみを使い、外部APIを呼びません。

## 実行

プロジェクトルートから実行してください。

```bash
python -m scripts.evaluation.evaluate \
  --predictions-dir res \
  --ks 1 3 5
```

既定の正解ファイルは `local_artifacts/five_case_benchmark/data/five_cases.json` です。別データは `--cases PATH` で指定できます。結果JSONは `local_artifacts/evaluation_results/` に一意な名前で保存されます。`--output PATH` でも指定できますが、既存ファイルは上書きしません。

Zero-shotだけ、または暫定診断だけを採点する場合:

```bash
python -m scripts.evaluation.evaluate --predictions-dir res --stages zeroShotResult
python -m scripts.evaluation.evaluate --predictions-dir res --stages tentativeDiagnosis
```

変更前後を比較する場合:

```bash
python -m scripts.evaluation.evaluate \
  --predictions-dir local_artifacts/experiments/new/node_results \
  --baseline-dir local_artifacts/experiments/baseline/node_results
```

両方を同一正解セットで再採点し、Hit@kの差、新しくヒットした症例、ヒットしなくなった症例、MRRの差を保存します。欠損・異常件数はそれぞれの `summary.status_counts` で確認してください。

## 出力結果の保存と実行ごとの比較

レポートには点数に加えて、以下のスナップショットを保存します。

- `cases[].stages.<段階>.output`: 採点対象の出力を加工せず保持。候補の配列順、元の順位、病名、OMIM ID、支持理由、参考情報、追加フィールドを含みます。不正な順位などで採点できなかった段階も、その出力を残します。
- `cases[].source_payload`: 読み込めた予測JSON全体。元ファイルに含まれるプロンプト・モデル設定なども残ります。
- `cases[].expected_output`: ベンチマークに記録された正解情報。
- `baseline.cases[]`: 比較対象についても同じ情報を保存します。

これにより、採点後に元の予測ファイルを変更してもレポート内の出力は残ります。ファイル欠損や読み込めないJSONは `source_payload: null` になり、元ファイルの生テキストは保存しません。

出力のブレを観測するには、推定を複数回実行し、それぞれ別の予測ディレクトリに保存して採点してください。点数が同じでも `output` を見れば順位・候補・説明の違いを確認できます。このスクリプト自体は推定を再実行せず、同じ保存済み出力を再採点しても新しい推定結果は得られません。分散などの複数実行の統計集計は行いません。

正規化前のLLM出力やプロンプトは、予測JSONに保存されている場合のみ保持できます。既存の正規化後の結果から、失われた候補や元の文章は復元しません。

## 入力形式

正解ファイルは以下の形式です（架空例）。`case_id` は一意、正解IDは必須です。

```json
{
  "cases": [{
    "case_id": "example_1",
    "patient_id": "1",
    "expected_output": {"omim_id": "OMIM:123456"}
  }]
}
```

予測ディレクトリの `example_1.json` または `1.json` を読みます。両方存在する場合は曖昧として採点を失敗扱いにします。予測内に `case_id` / `patient_id` がある場合は正解ファイルとの一致を検査します。

```json
{
  "zeroShotRaw": {
    "ans": [{"disease_name": "Example disease", "OMIM_id": "123456", "rank": 2}]
  },
  "zeroShotResult": {
    "ans": [{"disease_name": "Example disease", "OMIM_id": "OMIM:123456", "rank": 2}]
  },
  "tentativeDiagnosis": {
    "ans": [{"disease_name": "Example disease", "OMIM_id": "OMIM:123456", "rank": 1, "description": "Example rationale"}]
  }
}
```

- 現行の `res/{patient_id}.json` は `zeroShotResult` と `tentativeDiagnosis` を含む形式です。いずれも後段の正規化で更新された結果である点に注意してください。
- `zeroShotRaw` は評価用に予約した名前です。現行パイプラインは自動保存しません。正規化前に `result.model_dump()` などでコピー・保存した場合に `--stages zeroShotRaw` で採点できます。
- `{ "ans": [...] }` だけのファイルは、`--bare-stage zeroShotRaw` などで段階を明示すると採点できます。この指定は全予測ファイルとbaselineに適用されます。最終診断のファイルをZero-shotとして渡さないでください。
- `actual_output` の最終診断を段階出力として自動認識することはありません。
- 各候補のIDは `OMIM_id` / `omim_id` に対応します。両方が有効かつ異なる場合は異常として扱います。

## 採点規則

- OMIMの6桁番号を完全一致で比較します。`123456` と `OMIM:123456` は同一扱い。疾患名による曖昧一致やMONDO変換はしません。
- `rank` が k 以下に正解があればHit@k。配列の位置ではなく明示的な順位を使い、正規化による順位の穴は詰めません。
- MRRは各症例の最上位正解順位の逆数の平均。未ヒットは0です。Top-kの上限外も含む保存済み全候補が対象です。
- 各症例の正解が1疾患なので、Hit@k率はこのベンチマークにおけるRecall@kと一致します。
- 全症例を分母に含めます。ファイル欠損、段階欠損、解析失敗、空候補も0点です。欠損・異常があればレポートを保存したうえで終了コード2、通常は0になります。
- 順位の欠損・重複・非正整数は段階全体を異常扱いとします。IDなし・不正IDの候補は未一致とし、件数を記録します。
- 同じOMIMが複数出た場合は最小rankで照合し、重複数も記録します。
- 正解ファイルと予測ファイルのSHA-256を記録します。実験結果は段階ごと・モデルやプロンプト変更ごとに別ディレクトリへ保存してください。同じ `res/` の繰り返し利用は過去の結果との混在を招きます。

暫定診断だけの変更を比較するときは入力候補表とWeb結果を固定してください。ここでの指標は疾患の順位の評価であり、説明の正確性・引用の忠実性は評価しません。5症例での改善は探索的な結果として扱ってください。

## 単体テスト

```bash
python -m unittest discover -s tests -p 'test_evaluation*.py' -v
```

## 改善前後を比較するための処理トレース

両方の実行プログラムで、症例・反復ごとに `repeat_001/traces/<case_id>.jsonl` を追加保存します。既存の予測・採点ファイルとは別の観測記録です。実行コマンドの変更は不要です。

- **Zero-shot推論**：入力、プロンプト、構造化された戻り値。これはSDKによる構造化解析後の結果であり、解析前のAPI応答そのものではありません。SDK内部の解析失敗は関数の例外として記録します。
- **暫定推論の正規表現解析**：`parse_diagnosis_text` の `call` に解析前の全文 `data.text` とプロンプトを保存します。`parse_block_check` にブロック本文と各フィールドのマッチ有無、`return` に抽出結果を保存します。ブロック区切り自体の不一致は全文と `case_blocks` から確認できます。`tentativeRaw` は従来どおり「正規表現解析後・正規化前」であり、この全文とは異なります。
- **Zero-shot推論の正規化**：入力候補、加工した検索語、OMIM検索の戻り値、類似度、判定直前の候補と既出ID集合、最終候補を保存します。`normalization_decision_input` で低類似度と重複の条件を確認できます。Zero-shot単体評価は正規化を呼ばないため、これらは暫定推論までの評価で記録されます。
- **暫定推論に渡す候補統合**：各 `_add_candidate` の元の疾患名・ID・情報源と、統合後のキー・候補、統合関数の最終出力を保存します。空の疾患名による除外や同じIDへの統合を追跡できます。
- **暫定推論の候補照合**：`createDiagnosis` の入力候補とプロンプト、解析後の出力を保存します。入力にないIDの出現や候補の欠落を比較する材料です。候補制限や自動修正は行いません。
- **暫定推論の正規化**：処理前後の候補、既存ID、検索を行った場合の検索語・結果・類似度を保存します。既存IDを優先する現行処理も変更しません。

各イベントは `call_id`、スレッドID、関数名、行番号で対応付けます。`exception` は対象関数内で発生した例外で、上位で回復される場合もあります。症例全体の成否は従来の予測ファイルの `status` を確認してください。トレースからの除外理由の集計・候補照合の自動採点はまだ行いません。

実装はevaluation実行中だけPythonのトレースフックで対象関数の値を読み取ります。本番関数・プロンプト・State・正規化条件・戻り値・例外の伝播を変更せず、追加のLLM/API呼び出しも行いません。ただし記録の処理時間とディスク使用量は増えるため、実行時間の比較には向きません。

書き込みに失敗しても推論を続け、警告を表示します。正常終了したトレース末尾の `trace_end.recording_errors` が0か確認してください。デバッガやcoverageの既存フックがある場合は干渉を避けて記録を無効にし、警告します。開始前から存在する別スレッドは観測対象外です。対象関数名や判定行を書き換える改善では、トレース側の対象指定とテストも追従させてください。

記録には症例情報・プロンプト・モデル応答を含みます。既存のローカル評価出力と同じ場所に保存してください。`llm` オブジェクトや環境変数は保存対象に含めません。
