# ZebraSeek TogoMCP 検索・検証機能 要件定義

## 1. 目的

ZebraSeekは、患者の表現型と顔画像から初期候補疾患を作成し、候補疾患ごとにTogoMCPを用いて検証情報を収集する。検索で発見した新規疾患も、初期候補と同じ候補データ構造で検索・検証の対象にする。

検索終了後、候補疾患を根拠付きで評価し、再ランキングした上位疾患を出力する。最終出力には、支持根拠、矛盾根拠、既知の原因候補遺伝子を、それぞれ出典とともに含める。

TogoMCPの接続、ツール一覧、データベース、MIE、クエリ作成、検証手順は、実行時点の公式情報を参照する。

- 公式チュートリアル: https://togomcp.rdfportal.org/tutorial/ja
- 公式ホスト: https://togomcp.rdfportal.org/
- 公式リポジトリ: https://github.com/dbcls/togomcp

## 2. 範囲

### 2.1 対象

- Present HPOとAbsent HPOを用いた候補疾患の作成・検証
- 顔画像を用いたGestaltMatcher検索
- PubCaseFinder、ベクトル検索、PhenoBrain、ZeroShotによる初期候補作成
- TogoMCPによる疾患同一性、表現型、症例報告、文献、反証情報の収集
- 検索結果からの新規疾患候補の発見
- 検索完了後のReflectionによる候補評価
- Reflection結果を使ったLLMリランキング、または単純なツール平均によるリランキング
- 最終疾患に対する既知の原因候補遺伝子の表示
- 全API/MCPレスポンスと全根拠の出典追跡

### 2.2 対象外

- `clinical_text`を入力とする検索
- 患者の遺伝子・変異を入力とする推論
- 遺伝子情報だけを使った新規疾患探索
- 治療、薬剤、パスウェイ、発現解析の検索
- Reflectionによる検索継続判断

## 3. 入力要件

### 3.1 入力項目

| 項目 | 必須 | 内容 |
|---|---:|---|
| `patient_id` | 任意 | 患者を識別するID。未指定時は`unknown` |
| `present_hpo_ids` | 必須 | 患者で観察されたHPO IDのリスト |
| `absent_hpo_ids` | 任意 | 患者で明示的に観察されなかったHPO IDのリスト |
| `sex` | 任意 | 性別。未指定時は`unknown` |
| `onset` | 任意 | 発症時期。未指定時は`unknown` |
| `image_path` | 任意 | 顔画像パス。未指定時は画像検索をスキップ |

`clinical_text`は入力スキーマ、State、プロンプトから削除する。

### 3.2 不明値

`sex`と`onset`は、値がない場合でも検索コンテキストに必ず含める。値は`unknown`とする。ツールが当該項目を受け付けない場合はAPI引数に変換せず、実行コンテキストに保存する。

Absent HPOがない場合は空リストとして扱う。Absent HPOが未収集であることと、患者に存在しないことを混同しない。

## 4. 初期候補生成

### 4.1 初期ツール

以下の5手法を初期候補生成に使用する。

- PubCaseFinder
- GestaltMatcher
- ベクトル検索
- PhenoBrain
- ZeroShot

画像がない場合、画像を必要とするツールは実行しない。

### 4.2 レスポンス保存

各ツールについて、表示・LLM入力用のTop5と、照合・再現用の取得可能な全レスポンスを保存する。

全レスポンスは、新規疾患候補が初期ツール結果に存在するか、順位・スコアが何であったかを確認するために利用する。

APIが返却件数を制限した場合は、`complete`、`truncated`、`not_returned`を区別する。返却されなかったことを疾患否定として扱わない。

### 4.3 初期候補集合

通常の検索対象は、各初期ツールのTop5を統合した候補集合とする。疾患IDを最優先し、同義語・名称を補助的に使用して重複排除する。

全レスポンス中のTop5外の疾患は、順位照合と新規候補探索のために保存するが、初期の検索対象には含めない。

## 5. 固定検索ルート

候補疾患ごとの検索は、LLMが継続可否を判断するループにしない。候補ごとに決められた情報を検索し、検索完了後にReflectionへ渡す。

### 5.1 検索アクション

```text
resolve_identity
check_present_hpo
check_absent_hpo
search_case_reports
search_pubmed
search_contradiction
expand_candidates
```

初期候補生成で直接使用するPubCaseFinderのランキングAPIを、TogoMCPの固定検証ルートで再度呼び出してはならない。現在はTogoMCPの`ncbi_esearch`をMedGen、PubMed、Geneへ固定的に振り分け、疾患同定・表現型照合・症例報告・PubMed・矛盾情報・原因候補遺伝子を取得する。`expand_candidates`はそれらの結果から抽出する。`run_sparql`はMIEと対象グラフを明示した実装を追加するまで使用しない。

### 5.2 実行順序

1. `resolve_identity`で疾患ID・同義語を正規化する。
2. 正規化後、`check_present_hpo`、`check_absent_hpo`、`search_case_reports`、`search_pubmed`を実行する。
3. これらの結果を統合して`search_contradiction`を実行する。
4. 検索結果から`expand_candidates`で新規疾患を抽出する。
5. 対象候補の検索を完了し、Reflectionへ渡す。

疾患ID正規化後の独立した検索は並列実行する。反証検索と新規候補抽出は、前段の検索結果を入力とするため、その後に実行する。

### 5.3 検索対象情報

各候補について次の情報を収集する。

- 疾患ID・名称・同義語
- Present HPOとの疾患側の関係
- Absent HPOに対する明示的な否定または矛盾
- 症例報告
- PubMed文献
- 既存情報に対する反証

患者の`sex`と`onset`は検索コンテキストに含める。`clinical_text`は使用しない。

## 6. 新規疾患候補

### 6.1 発見経路

新規疾患は、固定検索ルートで取得した以下の結果から抽出する。

- 表現型検索結果に記載された疾患
- 症例報告に記載された診断名
- PubMedタイトル・抄録に明示された疾患名・疾患ID
- TogoMCPの疾患検索・横断検索結果

初期ツールの全レスポンスは新規疾患探索の主経路ではなく、新規候補と初期順位を照合するために利用する。

### 6.2 抽出規則

- 出典に記載された疾患名または疾患IDだけを候補にする。
- 可能な限りMONDO、OMIM、Orphanet、MedGen等のIDへ正規化する。
- 既存候補と同じ疾患IDなら、既存候補にEvidenceを追加する。
- 新規疾患なら、初期候補と同じ`CandidateRecord`を作成する。
- 新規疾患には、発見元の`source_id`と`evidence_id`を必ず付与する。

初期候補からの新規探索は1段階だけとする。新規疾患からさらに新規疾患を探索する処理は行わない。

### 6.3 新規候補の検索

新規疾患は、事前の別検証フェーズを持たず、通常候補と同じ固定検索ルートを実行する。ただし、新規疾患では`expand_candidates`を再度実行しない。

## 7. Absent HPO要件

Absent HPOの評価は、次の3値を区別する。

```text
contradicts      疾患側に明示的な否定・矛盾がある
unknown          データから判断できない
not_annotated    疾患側に該当アノテーションがない
```

疾患側に情報がないことを`contradicts`に変換しない。

## 8. Reflection

Reflectionは、検索が完了した時点で実行する。検索の継続、次の検索アクション、候補探索の実行可否はReflectionに判断させない。

### 8.1 入力

- 患者のPresent/Absent HPO
- `sex`、`onset`
- 初期ツールのTop5情報
- 初期ツール全件結果から照合した対象候補の順位・スコア
- TogoMCPのEvidence
- 症例報告とPubMedのEvidence
- 反証Evidence
- 新規疾患を含む全CandidateRecord

### 8.2 出力

候補ごとに次を返す。

```text
correct
incorrect
uncertain
```

同時に、支持Evidence、矛盾Evidence、判断理由、患者表現型との関係を返す。すべての根拠は`evidence_id`で参照する。

## 9. 最終リランキング

### 9.1 LLM方式

デフォルトは、Reflection結果とEvidenceを入力したLLMによるリランキングとする。LLMは候補IDの順序、各候補の判断、支持根拠、矛盾根拠を構造化出力する。

### 9.2 ツール平均方式

実行オプションで、LLMリランキングの代わりに単純なツール平均を選択できる。

ツール間でスコア尺度が異なるため、順位を0から1へ正規化して平均する。

```text
rank_score = 1 - (rank - 1) / (returned_count - 1)
not_returned = 0
tool_average = enabled_toolsのrank_scoreの算術平均
```

この方式では、Reflectionの判断は順位計算に使用せず、根拠付き評価として出力する。

## 10. 原因候補遺伝子

遺伝子検索は最終候補確定後に行う。疾患の順位、Reflection、検索終了、新規疾患探索には使用しない。

最終候補ごとに、既知の原因候補遺伝子を並列取得する。取得した遺伝子には、疾患ID、遺伝子ID、関係、DB、エントリー、URI/URL、Evidence IDを付与する。

## 11. 最終出力

最終出力には、少なくとも以下を含める。

- 患者ID
- 最終順位
- 疾患名
- 疾患ID
- Reflectionの判断
- 支持する根拠
- 矛盾する根拠
- 不明・未注釈の情報
- 既知の原因候補遺伝子
- 各ツールの順位・スコア
- 根拠ごとのDB、エントリーID、URI、URL

## 12. 出典追跡要件

すべての外部情報を、少なくとも次の属性で追跡可能にする。

```text
source_id
access_layer
tool
database
entry_id
uri
url
query
request
retrieved_at
raw_response_ref
```

Evidenceは必ず1つ以上の`source_id`を持つ。Reflectionと最終出力は、事実そのものをコピーせず、Evidence IDを参照する。

## 13. 並列化要件

以下は可能な限り並列実行する。

- 初期5ツール
- 初期候補の全件結果照合
- 疾患ID正規化後のPresent HPO、Absent HPO、症例報告、PubMed検索
- 複数候補の検索ルート
- 新規候補の検索ルート
- 最終候補ごとの原因候補遺伝子取得

候補統合、検索結果統合、Evidence ID付与、Reflection、最終リランキングは、依存データが揃った後に実行する。

## 14. 再現性・非機能要件

- API/MCPへの入力を保存する。
- rawレスポンスを保存する。
- TogoMCPの実ツール名、引数、検索クエリ、DB、エントリーを保存する。
- LLMのモデル名、プロンプト、構造化出力、実行時刻を保存する。
- 同一入力と同一設定で検索履歴を再現できるようにする。
- 並列実行時も、最終的な候補統合順序とEvidence順序を決定的にする。
- APIやMCPの一部失敗は候補全体の失敗にせず、該当SourceRecordにエラーを記録する。
