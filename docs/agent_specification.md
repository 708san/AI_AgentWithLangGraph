# ZebraSeek エージェント仕様書

## 1. 文書情報

| 項目 | 内容 |
|---|---|
| システム | ZebraSeek |
| 対象 | HPO・顔画像を用いた希少疾患候補の検索と検証 |
| 本仕様の位置づけ | TogoMCP統合後の実装仕様 |
| 入力の特徴 | `clinical_text`を持たず、`sex`と`onset`は不明時に`unknown`を使用 |
| 遺伝子の扱い | 最終疾患表示時の既知原因候補遺伝子取得だけに使用 |
| 最終リランキング | デフォルトはLLM。`tool_average`オプションで単純ツール平均を選択可能 |

本書は、[docs/togomcp_zebraseek_requirements.md](./togomcp_zebraseek_requirements.md)で定義した要件を、ノード、データ構造、呼び出し、並列化、保存形式へ落とし込む。

実行器はLangGraphの`StateGraph`を使用する。アクティブグラフは`ZebraSeekInputNode`、`InitialToolsNode`、`DiseaseNormalizeNode`、`CandidateResearchNode`、`ReflectionNode`、`RerankNode`、`GeneAnnotationNode`、`ZebraSeekOutputNode`で構成し、`zebraseek_context`で段階ごとの状態を受け渡す。従来のノード構成は`legacy_graph`に保持する。

アクション別の検索目的、プロンプト、実行条件は[docs/zebraseek_tool_overview.md](./zebraseek_tool_overview.md)にまとめる。

## 2. 設計原則

1. 検索に使用する患者情報を明示する。
2. 初期ツールのTop5を通常候補にし、取得可能な全レスポンスを保存する。
3. 初期候補と新規候補を同じ`CandidateRecord`で保持する。
4. TogoMCP検索は固定アクションのルートで実行し、Reflectionに検索継続を判断させない。
5. 検索完了後のReflectionで、候補ごとに`correct`、`incorrect`、`uncertain`を判定する。
6. 外部情報、正規化結果、Evidence、LLM判断を別層に保存する。
7. 全ての外部事実をDB・エントリー・URI/URL・クエリ・取得日時まで追跡可能にする。
8. データ依存関係のない呼び出しは並列実行し、統合処理は依存データが揃った後に行う。

`DiseaseNormalizeNode`は初期ツールの統合直後に実行する。旧`NormalizePCFNode`・`NormalizeGestaltMatcherNode`と同じく、OMIM IDの数値部分を基準にローカル`omim_mapping.json`の公式病名へ置換し、IDの接頭辞を正規化する。OMIM IDがない候補の名称は保持する。正規化前後と適用方法は`normalization_records`へ保存し、TogoMCPから追加された1段階候補にも同じ処理を適用する。

初期候補生成で直接実行するPubCaseFinderと、TogoMCPの固定検索ルートは重複させない。TogoMCP側のアクション割り当ては明示的な許可リストで行い、`pubcasefinder_rank_by_phenotypes`は検証ルートから除外する。現在の固定実装は、TogoMCPの`ncbi_esearch`をMedGen、PubMed、Geneの各データベースへ固定的に振り分けて疾患情報を取得する。`run_sparql`はMIEと対象グラフを明示したルートを追加するまで選択せず、`expand_candidates`は固定検索結果からローカル抽出する。

## 3. エントリポイント

### 3.1 推奨インターフェース

```python
pipeline.run(
    present_hpo_ids: list[str],
    absent_hpo_ids: list[str] | None = None,
    image_path: str | None = None,
    sex: str | None = None,
    onset: str | None = None,
    patient_id: str | None = None,
    ranking_mode: str = "llm",
)
```

`clinical_text`引数は定義しない。後方互換のために受け取る実装も作らない。

実装上のエントリポイントは`agent.tools.zebraseek_flow.run_zebraseek`であり、
`RareDiseaseDiagnosisPipeline.run`はこの関数を呼び出す互換ラッパーである。

### 3.2 入力正規化

```python
sex = sex or "unknown"
onset = onset or "unknown"
absent_hpo_ids = absent_hpo_ids or []
patient_id = patient_id or "unknown"
```

Present HPOが空の場合は初期候補生成を実行せず、入力エラーとして終了する。Absent HPOが空の場合は、Absent HPO検索を実行し、`unknown`または`not_applicable`を記録する。

## 4. 全体フロー

```text
入力正規化
  ↓
初期5ツールを並列実行
  ↓
Top5統合と全レスポンス保存
  ↓
初期CandidateRecord作成
  ↓
DiseaseNormalizeNode（OMIM ID・病名の正規化）
  ↓
各候補の固定検索
  ├─ 疾患ID正規化
  ├─ Present HPO
  ├─ Absent HPO
  ├─ 症例報告
  ├─ PubMed
  ├─ 反証検索
  └─ 新規疾患抽出
  ↓
新規CandidateRecordを追加
  ↓
新規候補にも固定検索を1段階だけ実行
  ↓
全候補の検索完了
  ↓
Reflectionで候補ごとの正誤評価
  ↓
LLMまたはtool_averageでリランキング
  ↓
最終候補ごとの原因候補遺伝子を並列取得
  ↓
最終出力
```

新規候補からさらに新規候補を探索しない。新規候補にも`resolve_identity`から`search_contradiction`までの固定検索を行うが、`expand_candidates`は実行しない。

## 5. State

### 5.1 State構成

```python
class ZebraSeekState(TypedDict, total=False):
    patient_input: PatientInput
    initial_tool_responses: list[ToolResponseRecord]
    disease_ranking_index: dict[str, DiseaseRankingIndexItem]
    candidate_pool: list[CandidateRecord]
    source_records: list[SourceRecord]
    evidence_records: list[EvidenceRecord]
    search_sessions: list[CandidateSearchSession]
    reflection_assessments: list[ReflectionAssessment]
    final_ranking: list[FinalCandidateResult]
    gene_annotations: list[GeneAnnotationRecord]
    ranking_mode: str
    execution_metadata: ExecutionMetadata
```

### 5.2 患者入力

```python
class PatientInput(TypedDict):
    patient_id: str
    present_hpo_ids: list[str]
    absent_hpo_ids: list[str]
    sex: str
    onset: str
    image_path: str | None
```

`clinical_text`はこの型に含めない。

### 5.3 初期ツールレスポンス

```python
class ToolResponseRecord(TypedDict, total=False):
    run_id: str
    tool: str
    request: dict
    request_context: PatientInput
    response_status: str
    completeness: str       # complete | truncated | unknown
    top5: list[RankedResult]
    all_results: list[RankedResult]
    raw_response_ref: str
    retrieved_at: str
    elapsed_ms: float
    error: str
```

`top5`は表示・LLM入力用の射影であり、`all_results`が保存可能な範囲の正式な取得結果である。

### 5.4 ランキング結果

```python
class RankedResult(TypedDict, total=False):
    disease_name: str
    disease_ids: list[str]
    rank: int
    score: float | None
    source_codes: list[str]
    raw_item: dict
```

### 5.5 疾患ランキングインデックス

```python
class DiseaseRankingIndexItem(TypedDict, total=False):
    disease_key: str
    disease_name: str
    normalized_ids: list[str]
    tool: str
    run_id: str
    rank: int | None
    score: float | None
    status: str       # found | not_returned | truncated | error
```

### 5.6 候補疾患

```python
class CandidateRecord(TypedDict, total=False):
    candidate_id: str
    disease_name: str
    normalized_ids: list[str]
    discovery_source_ids: list[str]
    discovery_evidence_ids: list[str]
    tool_rankings: dict[str, ToolCandidateRanking]
    evidence_ids: list[str]
    search_status: str       # pending | researching | complete | error
```

初期候補と新規候補で型を分けない。`discovery_source_ids`と`discovery_evidence_ids`は出典追跡用の属性であり、候補の評価規則を分けるために使用しない。

```python
class ToolCandidateRanking(TypedDict, total=False):
    tool: str
    rank: int | None
    score: float | None
    status: str              # found | not_returned | truncated | skipped | error
    run_id: str
```

## 6. 初期ツールノード

### 6.1 共通入力

各ノードは`PatientInput`を参照する。`sex`と`onset`は入力コンテキストへ含める。APIが対応しない項目はAPI引数に渡さず、`request_context`に保存する。

### 6.2 並列実行

以下を並列実行する。

```text
PCFNode
GestaltMatcherNode
VectorSearchNode
PhenoBrainNode
ZeroShotNode
```

画像がない場合、GestaltMatcherと画像を必要とするPhenoBrain処理は`skipped`として記録する。

### 6.3 初期結果の保存

各ノードは、次を返す。

```text
ToolResponseRecord
SourceRecord（外部APIの場合）
```

ZeroShotはLLM呼び出しだが、プロンプト、モデル名、構造化出力を`ToolResponseRecord`相当の実行記録として保存する。

## 7. 候補統合ノード

### 7.1 入力

- 各初期ツールのTop5
- 各ツールの`all_results`
- 疾患ID正規化器

### 7.2 処理

1. 疾患IDを優先して候補を識別する。
2. 疾患IDがない場合は、名称・同義語で一時キーを作る。
3. 同一疾患のランキングを1つの`CandidateRecord`へ統合する。
4. 全レスポンスから`DiseaseRankingIndex`を作る。
5. 各候補のツール別順位・スコア・状態を付与する。

### 7.3 出力

```text
candidate_pool
disease_ranking_index
```

## 8. 固定検索ノード

### 8.1 アクション一覧

```text
resolve_identity
check_present_hpo
check_absent_hpo
search_case_reports
search_pubmed
search_contradiction
expand_candidates
```

LLMは検索継続、次の検索、アクションの追加を決定しない。検索制御側がこの順序と実行条件を保持する。

### 8.2 `resolve_identity`

入力:

```text
candidate.disease_name
candidate.normalized_ids
```

出力:

```text
preferred disease ID
preferred label
synonyms
cross-references
identity conflicts
```

TogoMCPの実ツール名は、接続時の`tools/list`とMIE/ツール説明から解決する。取得した実ツール名、引数、DB、エントリーを`ToolCallRecord`に保存する。

### 8.3 `check_present_hpo`

疾患側の表現型アノテーションと、患者のPresent HPOを照合する。

Evidenceの極性:

```text
supports
unknown
not_annotated
```

患者HPOと疾患HPOの単純な文字列一致だけでなく、使用したDB・エントリー・HPO関係を保存する。

### 8.4 `check_absent_hpo`

疾患側に明示的なExcludedまたは否定情報がある場合だけ、矛盾として記録する。

```text
contradicts
unknown
not_annotated
```

疾患側に該当アノテーションがない場合は`contradicts`にしない。

### 8.5 `search_case_reports`

候補疾患ID・名称で症例報告を検索する。患者のPresent/Absent HPO、`sex`、`onset`は検索コンテキストとして保持する。

各症例報告は、PMID、文献タイトル、該当箇所、URL、取得日時を保存する。文献に記載された別の疾患名は`expand_candidates`の入力にする。

### 8.6 `search_pubmed`

候補疾患ID、候補疾患名、表現型に関連する固定クエリを用いてPubMedを検索する。

検索結果は、検索クエリ、DB、PMID、タイトル、抄録、URL、取得日時とともに保存する。

### 8.7 `search_contradiction`

前段の表現型、症例、PubMed結果を入力として、候補疾患と矛盾する情報を検索する。

出力は、支持根拠とは別のEvidenceとして保存する。矛盾が見つからない場合も、実行済み検索として記録する。

### 8.8 `expand_candidates`

前段の検索結果に明示された疾患名・疾患IDを抽出する。

候補抽出時のLLM出力は、自由な鑑別疾患生成ではなく、与えられた検索結果からの構造化抽出とする。

```python
class CandidateMention(TypedDict, total=False):
    disease_name: str
    normalized_ids: list[str]
    source_ids: list[str]
    evidence_ids: list[str]
    mention_context: str
```

新規疾患は、初期候補と同じ`CandidateRecord`に追加する。初期ツール全件結果に存在する場合は順位・スコアを付与し、存在しない場合は`not_returned`を付与する。

新規疾患に対しては`resolve_identity`から`search_contradiction`までを実行する。`expand_candidates`は再実行しない。

## 9. 検索セッション

```python
class CandidateSearchSession(TypedDict, total=False):
    session_id: str
    candidate_id: str
    expansion_depth: int
    planned_actions: list[str]
    completed_actions: list[str]
    action_records: list[str]
    discovered_candidate_ids: list[str]
    status: str              # complete | partial | error
    started_at: str
    finished_at: str
```

`planned_actions`は固定ルートから作成する。LLMの`need_more_search`や自由な検索計画は保存しない。

## 10. TogoMCP呼び出し

### 10.1 抽象アクションと実ツール

アプリケーションは論理アクション名を使用する。実際のMCPツール名は、実行開始時に取得したツールカタログから`ActionBinding`へ解決する。

```python
class ActionBinding(TypedDict, total=False):
    action: str
    tool_name: str
    database: str
    input_mapping: dict
    output_mapping: dict
    availability: str
```

この方式により、TogoMCP側のツール名変更を候補検索ロジックから分離する。

### 10.2 ToolCallRecord

```python
class ToolCallRecord(TypedDict, total=False):
    call_id: str
    action: str
    tool_name: str
    arguments: dict
    database: str
    query: str
    result_ref: str
    source_ids: list[str]
    status: str              # success | empty | error | not_applicable
    started_at: str
    elapsed_ms: float
    error: str
```

TogoMCPのガイド、MIE、実行クエリ、エンドポイントも、呼び出しとともに保存する。

## 11. 情報源とEvidence

### 11.1 SourceRecord

```python
class SourceRecord(TypedDict, total=False):
    source_id: str
    access_layer: str        # direct_api | TogoMCP | local_file | llm
    tool: str
    database: str
    source_type: str         # ranking | phenotype | case_report | literature | ontology | gene
    entry_id: str
    uri: str
    url: str
    endpoint: str
    query: str
    request: dict
    retrieved_at: str
    raw_response_ref: str
    content_hash: str
    status: str
```

出典がローカルファイルの場合は`local_file`としてパス、ファイルハッシュ、該当エントリーを保存する。URLが存在しない場合も空欄にせず、`uri`または`local_path`のいずれかを記録する。

### 11.2 EvidenceRecord

```python
class EvidenceRecord(TypedDict, total=False):
    evidence_id: str
    candidate_id: str
    source_ids: list[str]
    claim: dict
    polarity: str            # supports | contradicts | unknown | not_annotated
    relation: str
    excerpt: str
    structured_value: dict
    extraction_method: str   # api_mapping | deterministic_match | llm_extraction
    created_at: str
```

Evidenceは、必ず1つ以上の`source_ids`を持つ。LLMが作成した要約も、元のEvidence IDを失わない。

## 12. Reflection

### 12.1 実行タイミング

全CandidateRecordの固定検索が完了した後に1回実行する。

新規候補も、通常候補と同じReflection入力へ含める。

### 12.2 入力

- `PatientInput`
- CandidateRecord
- 初期ツールTop5
- 初期ツール全件結果との照合結果
- TogoMCP Evidence
- 症例報告Evidence
- PubMed Evidence
- 反証Evidence
- 全SourceRecordの参照情報

`clinical_text`は含めない。`sex`と`onset`は値または`unknown`として含める。

### 12.3 出力

```python
class ReflectionAssessment(TypedDict, total=False):
    candidate_id: str
    judgment: str           # correct | incorrect | uncertain
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    unknown_evidence_ids: list[str]
    patient_summary: str
    analysis: str
    source_ids: list[str]
    model: str
    prompt_ref: str
    created_at: str
```

`judgment`は、収集した根拠に対する候補の妥当性評価である。根拠のない事実を追加せず、判断に使用したEvidence IDを必ず付与する。

## 13. 最終リランキング

### 13.1 `ranking_mode`

```text
llm            デフォルト。ReflectionとEvidenceを用いてLLMが順位を生成
tool_average   ツール順位の単純平均
```

### 13.2 LLM方式

LLMは、CandidateRecordとReflectionAssessmentを入力として、候補IDの順序を構造化出力する。

```python
class LLMRankingItem(TypedDict, total=False):
    candidate_id: str
    rank: int
    rationale: str
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
```

### 13.3 tool_average方式

各ツールの順位を、候補が返却された順位集合に対して0から1へ正規化する。

```text
rank_score = 1 - (rank - 1) / (returned_count - 1)
not_returned = 0
tool_average = enabled_and_executed_toolsのrank_scoreの算術平均
```

画像がなくスキップされたツールは平均の分母から除外する。APIの結果に存在しない候補は、そのツールについて0とする。

## 14. 最終原因候補遺伝子

最終ランキング確定後、各疾患について既知の原因候補遺伝子を並列検索する。

```python
class GeneAnnotationRecord(TypedDict, total=False):
    annotation_id: str
    disease_candidate_id: str
    disease_id: str
    gene_id: str
    gene_symbol: str
    relation: str
    source_ids: list[str]
    evidence_ids: list[str]
    retrieved_at: str
```

この結果は、候補順位、Reflection、検索ルート、新規疾患抽出に使用しない。

## 15. 最終出力

```python
class FinalCandidateResult(TypedDict, total=False):
    candidate_id: str
    rank: int
    disease_name: str
    normalized_ids: list[str]
    judgment: str
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    unknown_evidence_ids: list[str]
    known_causal_gene_annotation_ids: list[str]
    tool_rankings: dict[str, ToolCandidateRanking]
    rationale: str
```

```python
class ZebraSeekOutput(TypedDict, total=False):
    patient_id: str
    ranked_candidates: list[FinalCandidateResult]
    source_records: list[SourceRecord]
    evidence_records: list[EvidenceRecord]
    tool_response_records: list[ToolResponseRecord]
    reflection_assessments: list[ReflectionAssessment]
    gene_annotations: list[GeneAnnotationRecord]
    execution_metadata: dict
```

通常表示は上位5疾患とする。監査用ファイルには、候補全体、全ツールレスポンス、全Evidence、全SourceRecord、Reflection結果を保存する。

## 16. 並列実行の依存関係

```text
患者入力正規化
  ↓
初期5ツール ─────────────┐
                          ↓
                    候補統合
                          ↓
              候補ごとの疾患ID正規化
                          ↓
  Present HPO ────────────┐
  Absent HPO ─────────────┤
  症例報告 ───────────────┤→ 結果統合
  PubMed ────────────────┘
                          ↓
                    反証検索
                          ↓
                  新規候補抽出
                          ↓
             新規候補の固定検索
                          ↓
                    Reflection
                          ↓
                    リランキング
                          ↓
              原因候補遺伝子検索（並列）
```

同じMCPサーバーへの並列数は設定可能な同時実行数で制限する。並列結果を統合するときは、`action`、`candidate_id`、`source_id`、取得時刻をキーにして決定的な順序を作る。

## 17. エラー・不明値

| 状態 | 意味 | 候補評価 |
|---|---|---|
| `success` | 結果取得済み | Evidenceを作成 |
| `empty` | 検索結果が空 | `unknown`または`not_found`を記録 |
| `not_applicable` | 入力・DBの制約で対象外 | SourceRecordに理由を記録 |
| `truncated` | API返却上限で全件でない | 未返却と区別 |
| `error` | 呼び出し失敗 | エラーSourceRecordを作成 |

`empty`、`not_returned`、`not_annotated`を`contradicts`へ変換しない。

## 18. 既存実装からの移行

### 18.1 削除・変更する項目

- `clinicalText`をStateから削除する。
- `use_absentHPO`によるAbsent HPOの任意利用を終了し、Absent HPOを常に検索コンテキストへ含める。
- `togomcp_search_plans.need_more_search`を検索制御に使用しない。
- `candidate_admission_decisions`による新規疾患専用の候補型を廃止する。
- `check_gene`を候補検索アクションから削除する。
- Reflection後にBeginningへ戻る検索継続分岐を削除する。

### 18.2 追加する項目

- `all_results`と`completeness`を持つToolResponseRecord
- DiseaseRankingIndex
- 固定検索アクションのSearchSession
- SourceRecordのURI、エントリー、エンドポイント、ハッシュ
- EvidenceRecordの支持・矛盾・不明の極性
- ReflectionAssessment
- `ranking_mode`
- 最終GeneAnnotationRecord

## 19. 検証項目

最低限、以下をテストする。

1. `clinical_text`なしで入力を受け付ける。
2. `sex`と`onset`が未指定の場合に`unknown`になる。
3. 各初期ツールのTop5が候補集合へ統合される。
4. Top5外を含む全レスポンスが保存される。
5. 新規疾患が初期全レスポンスと照合される。
6. 新規疾患が通常候補と同じ固定検索を受ける。
7. 新規疾患からの二段階目の候補探索が実行されない。
8. Absent HPOの未注釈が矛盾扱いされない。
9. Reflectionが検索後に一度実行され、3値を返す。
10. `ranking_mode=llm`と`ranking_mode=tool_average`が切り替わる。
11. 最終疾患ごとの原因候補遺伝子取得が並列実行される。
12. 全EvidenceがSourceRecordへ到達できる。
13. すべての並列結果を同一入力で決定的に統合できる。
