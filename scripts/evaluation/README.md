# Zero-shot・暫定診断の実行評価と保存済み出力の採点

`run_zero_shot`・`run_tentative`は外部APIを呼んで推論・保存・採点します。`evaluate`は保存済み出力だけを採点し、APIを呼びません。両ランナーの`--dry-run`もAPIを呼びません。

コード上の暫定診断（候補の順位付け）は`createDiagnosis` / `tentativeDiagnosis`です。最終診断とは別の段階です。推論の仕様は[agent仕様書](../../docs/agent_specification.md)の5.4・5.9節も参照してください。

## ベンチマークから推論・保存・採点する

プロジェクトルートで、依存をインストールしたPython環境を使って実行します（ローカル検証はPython 3.12）。以下の症例ファイルはGit管理外の例です。cloneしただけでは存在しないため、利用できるベンチマークを用意して`--benchmark`を指定してください。

```bash
# Zero-shotと病名正規化を実行（正規化用Embedding APIを使用）
python -m scripts.evaluation.run_zero_shot \
  --benchmark local_artifacts/five_case_benchmark/data/five_cases.json \
  --model gpt-5-2 --repeats 3

# 入力から暫定診断の正規化までを実行
python -m scripts.evaluation.run_tentative \
  --benchmark local_artifacts/five_case_benchmark/data/five_cases.json \
  --model gpt-5-2 --repeats 3
```

74症例データを配置済みの場合、陰性HPO・画像・選定理由ログを使用する例:

```bash
python -m scripts.evaluation.run_tentative \
  --benchmark local_artifacts/OldData/old_74_cases.json \
  --model gpt-5-2 --use-absent-hpo --enable-log --repeats 3 \
  --output-dir "local_artifacts/evaluation_results/tentative_74_$(date +%Y%m%dT%H%M%S)"
```

`--dry-run` を付けるとケース・HPO入力と（暫定診断の場合）画像パスを確認するだけで、APIを呼びません。API認証・依存ライブラリ・検索インデックスの動作検証は含みません。指定モデルのAzure環境変数とHPO辞書が必要です。暫定診断にはさらに既存パイプラインのEmbedding設定・FAISSインデックス等が必要です。

- `--model`: `gpt-4o` / `gpt-5-1` / `gpt-5-2`。省略時は`gpt-4o`なので、比較時は明示する。
- `--repeats N`: 反復回数。既定は1。
- `--ks 1 3 5`: 採点する順位上限。
- `--use-absent-hpo`: 陰性所見を使用（既定は既存処理に合わせて無効）。
- `--output-dir PATH`: 新規実験ディレクトリ。既存ディレクトリは上書きせずエラー。
- `--no-images`: 暫定診断を画像なし条件で実行。
- `--enable-log`: 暫定診断評価で通常ログ（Zero-shotの選定理由を含む）を `log/<実験ディレクトリ名>/repeat_NNN/<case_id>.log` に保存。評価用JSONL Traceとは別。
- `--image-root PATH`: ベンチマーク内の相対画像パスの基準。省略時は正解ファイルの親・その親・プロジェクトルートから一意に解決する。

`--enable-log`は`run_tentative`専用です。`run_zero_shot`への指定はエラーになります。ログは`--output-dir`の外側に保存されるため、結果を転送する際は対応する`log/`も含めてください。同じ実験ディレクトリ名のログが存在すると、出力先の親パスが異なっても実行を拒否します。

Zero-shot評価ではHPOラベル変換後に `createZeroshot()` を呼び、続けて本番の `normalize_zeroshot_results()` を実行します。正規化前を `zeroShotRaw`、正規化後を `zeroShotResult` として保存し、両方を独立に採点します。正規化は候補を直接変更するため、実行前の出力をコピーしてディスクに保存します。正規化に失敗しても、この出力は残ります。PubCaseFinder・画像API・HPO類似検索・Web検索・暫定診断は呼びません。

Zero-shot評価にも `AZURE_DBCLS_JAPANEAST` と `agent/data/DataForOmimMapping/` の `.bin`・対応JSONが必要です。正規化用Embedding APIが追加で呼ばれます。古いZero-shot評価には `zeroShotRaw` しかないため、以前の結果との直接比較は同じ段階同士で行ってください。古い正規化前結果と新しい正規化後結果を同じ指標として比較しないでください。

暫定診断では本番のノードとエッジを再利用し、PubCaseFinder・任意の顔画像解析・Zero-shot・表現型検索・HPO Web検索・候補統合・暫定診断・病名正規化を実行します。PhenoBrainは本番既定どおり無効です。疾患別Wikipedia/PubMed検索、Reflection、最終診断は実行しません。上流の検索も毎回実行するので、その変動を含む評価です。

各実験の保存構成:

```text
<output-dir>/
  metadata.json                 # モデル、入力・コードのハッシュ、実行条件
  repeat_001/
    predictions/<case_id>.json  # 推論出力、プロンプト、入力、時間、エラー
    traces/<case_id>.jsonl      # 処理前後・生成試行の観測記録
    node_results/               # 暫定診断用の既存ノード保存結果
    evaluation.json             # 正解との照合と出力スナップショット
  repeat_002/...

log/<実験ディレクトリ名>/       # run_tentative --enable-log の場合のみ
  repeat_001/<case_id>.log      # 通常のノードログとZero-shot選定理由
```

`metadata.json`はモデル名、Pythonバージョン、ベンチマークのSHA-256、実行条件、`agent/`配下のPythonコードのハッシュを保存します。Gitコミット、依存バージョン一覧、インデックスのハッシュ、完全なモデル設定、料金は保存しません。厳密に条件を追跡する場合はこれらを別途記録してください。

### 保存する推論段階

- **Zero-shot推論**：`zeroShotRaw`は構造化解析後・正規化前、`zeroShotResult`は正規化後。どちらも候補の`disease_name`・`rank`・`OMIM_id`を持ち、`selection_reason`は含みません。
- **暫定診断**：`tentativeRaw`は最後の生成試行の構造化解析後・正規化前、`tentativeDiagnosis`は正規化後。初回・再生成それぞれの出力はTraceを確認します。

`Raw`はAPIの生テキストという意味ではありません。Zero-shotと暫定診断の正規化アルゴリズムは今回のプロンプト・構造化出力・理由ログの変更では変更していません。LLM出力の候補保持と、正規化による候補除外・OMIM置換は別々に評価します。

### Zero-shot選定理由の生成と保存

`createZeroshot`は`ZeroShotReasonedOutput`で各候補の`selection_reason`（最大2文という指示）を生成し、理由を除いた新しい`ZeroShotOutput`を後段へ返します。理由付き応答の独立スナップショットはログ用の戻り値ラッパーを経由し、State・候補統合・暫定診断の入力には渡しません。

- `run_tentative --enable-log`：通常ログの`Zero-shot Selection Reasons (before normalization)`節に保存します。
- `run_tentative`のログ無効時、および`run_zero_shot`：理由は生成しますが保存しません。理由本文は予測JSON・採点JSON・現行の評価Traceにも保存しません。保存プロンプトには理由を要求する指示が含まれます。
- 本番ノードがキャッシュ済み`zeroShotResult`を再利用するときは、新しい理由を生成しません。

理由追加はプロンプトとLLM出力スキーマも変えるため、単なる観測ログの追加ではありません。理由追加前の評価を、そのまま理由追加後の精度検証として扱わないでください。

### 暫定診断の構造化出力と再生成

従来の実験では`tentativeRaw`に正規表現で解析した結果が保存されています。新旧の`tentativeRaw`はいずれも正規化前ですが、解析方法は異なります。

暫定診断は専用の`TentativeDiagnosisOutput`を`method="json_schema", strict=True, include_raw=True`で指定します。GestaltMatcherあり／なしの両方で同じスキーマを使用します。共通の`DiagnosisOutput`を継承し、暫定診断の各候補だけに必須の`candidate_id`を追加しています。最終診断のスキーマは変更しません。

入力候補の順番に`candidate_0001`などを割り当て、初回・再生成で同じIDを使います（異なる症例や実行間の恒久IDではありません）。出力IDの不足・候補外ID・重複を照合し、不一致なら元のプロンプト＋初回の全出力＋照合結果＋修正指示で全候補を1回だけ再生成します。順位・病名・OMIMは照合条件にせず、病名・OMIMを入力値で上書きしません。再生成後も不一致なら警告を出し、最後の構造化出力をそのまま返します。空リストも保持します。構造化解析失敗・拒否・通信例外は別のエラーであり、旧パーサーへのフォールバックはありません。既存のAPI/content-filter再試行と、候補照合による再生成は別です。

評価トレースの`function=record_diagnosis_attempt, event=call`が各試行の記録です。`data.attempt`は1（初回）または2（再生成）、`input_candidates`にはID・入力病名・入力OMIM、`output`にはその回の全出力、`validation`には`matched`・`missing_ids`・`unexpected_ids`・`duplicate_ids`を保存します。`prompt`と取得できた`llm_response`も記録します。入力と出力の病名・OMIMはここから確認できますが、自動的な正誤判定は行いません。照合サマリーはPythonのloggingへ出力します（初回一致はINFO、不一致はWARNING。表示はlogging設定に依存）。初回・再生成の全文は評価Traceで確認してください。`--enable-log`のノードログとは別です。予測JSONの`tentativeRaw`は最後の試行の結果です。後続の病名正規化は従来どおり実行されるため、そこでの候補除外とは区別してください。

正解ラベルは採点のみで使用し、推論関数には患者情報をホワイトリストで渡します。`--repeats`は独立した繰り返し実行で、各回を別々に採点します。複数回の平均・分散の集計は現在含みません。

### 完了・失敗・中断の確認

ノード出力は随時保存し、後続で失敗しても完了した段階を残します。症例のエラー後は次の症例へ進み、失敗症例を含むレポートを保存します。初期化失敗は`initialization_error.json`に記録して終了します。

- 予測JSONの`status`は`running` → `ok`または`error`です。`ok`は症例処理が例外なく戻ったことを表し、外部ツールの空結果、候補照合不一致、空候補、診断の正しさまで保証しません。
- `evaluation.json`の`status_counts`は各段階の採点可能性です。生成処理の`status`や候補IDの`validation.matched`とは別に確認します。
- `run_*`の終了コードは症例・初期化エラーで1、通常完了で0です。採点の異常件数を終了コードへ反映するものではありません。CLI引数エラー等は2です。
- 各反復の全症例が揃い、`evaluation.json`があり、各Traceの末尾が`trace_end`かつ`recording_errors=0`か確認してください。
- 途中再開・既存出力への追記は未実装です。中断後は新しい出力先に`--repeats 1`などで再実行します。完了済み反復は残し、採用する実験・反復を別途記録して二重集計を防ぎます。

## 保存済み出力だけを採点する

以下の `evaluate` は標準ライブラリのみを使い、外部APIを呼びません。

### 実行

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
  --cases local_artifacts/OldData/old_74_cases.json \
  --predictions-dir local_artifacts/evaluation_results/new/repeat_001/predictions \
  --baseline-dir local_artifacts/evaluation_results/baseline/repeat_001/predictions \
  --stages zeroShotRaw zeroShotResult tentativeRaw tentativeDiagnosis
```

`new`・`baseline`は実際の実験名に置き換えてください。`--predictions-dir`には実験ルートではなく、1反復分の症例JSONが直接入るディレクトリを指定します。`node_results`だけでは正規化前のスナップショットを揃えられないため、4段階の比較には`predictions`を使います。保存済み採点CLIの既定段階は`zeroShotResult`と`tentativeDiagnosis`のみです。

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

推論実行用ベンチマークは以下の形式です（架空例）。`case_id`はファイル名として安全な一意の文字列、正解IDは必須です。`input.present_hpo_list`は非空のHPO ID配列にします。

```json
{
  "cases": [{
    "case_id": "example_1",
    "patient_id": "1",
    "input": {
      "present_hpo_list": ["HP:0001250"],
      "absent_hpo_list": ["HP:0000252"],
      "sex": "female",
      "onset": "Unknown",
      "image_path": "test_images/example.jpg"
    },
    "expected_output": {"omim_id": "OMIM:123456"}
  }]
}
```

画像を使う場合は実在するファイルへ置き換え、画像なしなら`image_path`を省略するかnullにします。`onset`は発症時期であり現在の年齢ではありません。正解疾患や遺伝子情報を患者入力へ混ぜないでください。`patient_id`は省略可能で、実行ランナーは`case_id`を症例の保存IDとして使用します。保存済み出力だけを採点する`evaluate`では`input`は不要です。

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
    "ans": [{"candidate_id": "candidate_0001", "disease_name": "Example disease", "OMIM_id": "OMIM:123456", "rank": 1, "description": "Example rationale"}],
    "reference": null
  }
}
```

- 現行の `res/{patient_id}.json` は `zeroShotResult` と `tentativeDiagnosis` を含む形式です。いずれも後段の正規化で更新された結果である点に注意してください。
- `zeroShotRaw`は評価ランナーが保存するキーです。本番パイプラインの`res/`出力には自動追加されません。独自の実行コードでは正規化前のスナップショットを明示的に保存する必要があります。
- `{ "ans": [...] }` だけのファイルは、`--bare-stage zeroShotRaw` などで段階を明示すると採点できます。この指定は全予測ファイルとbaselineに適用されます。最終診断のファイルをZero-shotとして渡さないでください。
- `actual_output` の最終診断を段階出力として自動認識することはありません。
- 各候補のIDは `OMIM_id` / `omim_id` に対応します。両方が有効かつ異なる場合は異常として扱います。

## 採点規則

- OMIMの6桁番号を完全一致で比較します。`123456` と `OMIM:123456` は同一扱い。疾患名による曖昧一致やMONDO変換はしません。
- `rank` が k 以下に正解があればHit@k。配列の位置ではなく明示的な順位を使い、正規化による順位の穴は詰めません。
- MRRは各症例の最上位正解順位の逆数の平均。未ヒットは0です。Top-kの上限外も含む保存済み全候補が対象です。
- 各症例の正解が1疾患なので、Hit@k率はこのベンチマークにおけるRecall@kと一致します。
- 全症例を分母に含めます。ファイル欠損、段階欠損、解析失敗、空候補も0点です。保存済み採点CLIの`evaluate`は、欠損・異常があればレポート保存後に終了コード2、通常は0です。空候補（`empty`）だけでは終了コード2になりません。
- 順位の欠損・重複・非正整数は段階全体を異常扱いとします。IDなし・不正IDの候補は未一致とし、件数を記録します。
- 同じOMIMが複数出た場合は最小rankで照合し、重複数も記録します。
- 正解ファイルと予測ファイルのSHA-256を記録します。実験結果は段階ごと・モデルやプロンプト変更ごとに別ディレクトリへ保存してください。同じ `res/` の繰り返し利用は過去の結果との混在を招きます。

### 改善を比較する際の観点

- **Zero-shotの精度**：正規化前後を同じ段階同士で比較します。OMIM欠損、正規化による正解候補の喪失・獲得は予測とTraceから追跡できます。
- **暫定診断の安全性**：初回照合一致率、欠落・追加・重複、再生成回数と解消・未解消、解析失敗をTraceから集計します。正規化での脱落は別に集計します。これらの安全性指標は`evaluate`が自動集計するものではありません。
- 改善前の正規表現方式には`candidate_id`がないため、入力と生LLM出力、パーサー出力を病名・OMIMで対応付けます。曖昧な対応は判定不能とし、候補数の一致だけで保持を判定しないでください。
- 全パイプラインの再実行では上流候補・外部APIの状態も変動します。暫定診断単独の効果を厳密に分離するには入力候補表とWeb結果を固定する比較が必要ですが、固定入力を再実行するCLIは未実装です。
- 説明の正確性・引用の忠実性は採点しません。症例数、疾患の偏り、反復を考慮してください。74症例×3反復は222独立症例ではありません。

## 単体テスト

```bash
python -m unittest \
  tests.test_evaluation tests.test_evaluation_runner tests.test_evaluation_trace \
  tests.test_tentative_structured_output tests.test_zero_shot_reason_logging -q
```

## 改善前後を比較するための処理トレース

両方の実行プログラムで、症例・反復ごとに `repeat_001/traces/<case_id>.jsonl` を追加保存します。既存の予測・採点ファイルとは別の観測記録です。実行コマンドの変更は不要です。

- **Zero-shot推論**：入力、プロンプト、構造化された戻り値。これはSDKによる構造化解析後の結果であり、解析前のAPI応答そのものではありません。SDK内部の解析失敗は関数の例外として記録します。
- **暫定診断の構造化出力**：`createDiagnosis`の`return`または`exception`に、取得できた生応答を`data.llm_response`、プロンプトを`data.prompt`として保存します。LangChainが返した解析エラーは`data.parsing_error`に残します。SDK内部で応答返却前に例外となる場合は、生応答が取得できず例外のみが残ることがあります。`tentativeRaw`は構造化解析後・病名正規化前の結果です。旧`parse_diagnosis_text`は保存済みテキスト用に残していますが、新規の暫定診断は呼び出しません。旧形式のトレースでは`parse_block_check`で正規表現の取得失敗を確認できます。
- **Zero-shot推論の正規化**：入力候補、加工した検索語、OMIM検索の戻り値、類似度、判定直前の候補と既出ID集合、最終候補を保存します。`normalization_decision_input` で低類似度と重複の条件を確認できます。Zero-shot評価・暫定推論までの評価の両方で記録されます。
- **暫定診断に渡す候補統合**：各 `_add_candidate` の元の疾患名・ID・情報源と、統合後のキー・候補、統合関数の最終出力を保存します。空の疾患名による除外や同じIDへの統合を追跡できます。
- **暫定診断の候補照合**：`record_diagnosis_attempt`の各試行を保存します。不一致時の再生成は本番の`createDiagnosis`が行い、Trace自体は観測のみです。取得できなかった応答は例外のみの場合があります。
- **暫定診断の正規化**：処理前後の候補、既存ID、検索を行った場合の検索語・結果・類似度を保存します。既存IDを優先する現行処理も変更しません。

各イベントは `call_id`、スレッドID、関数名、行番号で対応付けます。`exception` は対象関数内で発生した例外で、上位で回復される場合もあります。症例全体の成否は従来の予測ファイルの `status` を確認してください。トレースからの除外理由の集計・候補照合の自動採点はまだ行いません。

Trace機構はevaluation実行中だけPythonのトレースフックで対象関数の値を読み取ります。Traceの有効化自体は本番関数・プロンプト・State・正規化条件・戻り値・例外の伝播を変更せず、追加のLLM/API呼び出しも行いません。暫定診断の再生成やZero-shot理由生成は本番実装側の機能です。記録の処理時間とディスク使用量は増えるため、実行時間の比較には向きません。

書き込みに失敗しても推論を続け、警告を表示します。正常終了したトレース末尾の `trace_end.recording_errors` が0か確認してください。デバッガやcoverageの既存フックがある場合は干渉を避けて記録を無効にし、警告します。開始前から存在する別スレッドは観測対象外です。対象関数名や判定行を書き換える改善では、トレース側の対象指定とテストも追従させてください。

記録には症例情報・プロンプト・モデル応答を含みます。既存のローカル評価出力と同じ場所に保存してください。`llm` オブジェクトや環境変数は保存対象に含めません。
