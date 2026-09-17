# ZebraSeek ツール概要

## 1. 目的

本書は、ZebraSeekが使用する初期ツール、TogoMCP論理アクション、LLMプロンプト、出典保存、並列実行単位を一覧化する。

実行は`RareDiseaseDiagnosisPipeline.graph`に構築したLangGraphの`StateGraph`で管理する。アクティブグラフは、`ZebraSeekInputNode`、`InitialToolsNode`、`DiseaseNormalizeNode`、`CandidateResearchNode`、`ReflectionNode`、`RerankNode`、`GeneAnnotationNode`、`ZebraSeekOutputNode`の順に実行する。各段階の状態は`zebraseek_context`で受け渡し、`procedure_trace`、正規化記録、既存のノードログへ保存する。旧来のノードグラフは`legacy_graph`として保持する。

実際のTogoMCPツール名は、起動時に取得した`tools/list`とMIEから解決する。本文のアクション名は、TogoMCPの具体的な関数名から分離したZebraSeek側の名前である。

初期候補生成の直接PCF呼び出しと、TogoMCPによる検証検索は分離する。TogoMCPの固定ルートでは`pubcasefinder_rank_by_phenotypes`を使用しない。現在の固定実装は、疾患同定・表現型照合・症例報告・PubMed・反証・原因候補遺伝子を`ncbi_esearch`（MedGen、PubMed、Gene）で取得する。`TogoMCP_Usage_Guide`は起動時の手順確認だけに使い、検索アクションへは割り当てない。`run_sparql`はMIEと対象グラフが明示的に設定された別ルートを追加するまで選択しない。`expand_candidates`は症例報告・PubMedの取得結果からローカルに候補を抽出する。

`DiseaseNormalizeNode`は初期候補の統合後に実行する。OMIM IDを優先して`omim_mapping.json`の公式病名へそろえ、接頭辞を統一する。正規化の前後値は`normalization_records`に残すため、候補名の変換理由を後から追跡できる。TogoMCPの1段階探索で追加された候補にも同じ正規化処理を適用する。

## 2. 共通検索コンテキスト

```json
{
  "patient_id": "272",
  "present_hpo_ids": ["HP:0001263", "HP:0010808"],
  "absent_hpo_ids": ["HP:0001250"],
  "sex": "FEMALE",
  "onset": "childhood",
  "image_path": "/path/to/image.jpg"
}
```

`sex`と`onset`は欠損時に`unknown`を入れる。`clinical_text`は使用しない。

## 3. 初期ツール

| ツール | 主入力 | 通常利用 | 保存 |
|---|---|---|---|
| PubCaseFinder | Present HPO | Top5 | Top5、取得可能な全件、raw、リクエスト |
| GestaltMatcher | 顔画像 | Top5 | Top5、取得可能な全件、raw、リクエスト |
| VectorSearch | Present HPO | Top5 | Top5、取得可能な全件、raw、リクエスト |
| PhenoBrain | Present HPO | Top5 | Top5、取得可能な全件、raw、リクエスト |
| ZeroShot | HPO、sex、onset | 構造化候補 | プロンプト、モデル、全構造化出力 |

初期ツールは独立しているため、可能な限り並列実行する。画像がないツールは`skipped`として記録する。

## 4. 初期レスポンス処理

1. rawレスポンスを保存する。
2. ツール固有形式を`RankedResult`へ正規化する。
3. `top5`を作成する。
4. 取得可能な全結果を`all_results`へ保存する。
5. `DiseaseRankingIndex`へ疾患ID、名称、順位、スコア、ツール、run_idを登録する。
6. Top5の和集合を初期候補として統合する。

新規候補が出たときは、`DiseaseRankingIndex`を参照して初期ツールでの順位・スコアを付与する。初期レスポンスに存在しない場合は`not_returned`と記録する。

## 5. 固定TogoMCPアクション

### 5.1 共通方針

- `tools/list`で使用可能な実ツールを確認する。
- TogoMCPの使用ガイドを確認する。
- SPARQLを使う場合は対象DBのMIEを確認する。
- 検索クエリ、DB、エントリー、エンドポイント、URL、取得日時、rawレスポンスを保存する。
- 検索順序とアクションはアプリケーション側で固定する。
- LLMは検索継続や次のアクションを決めない。

### 5.2 アクション一覧

| 論理アクション | 入力 | 主な検索内容 | 並列化 |
|---|---|---|---|
| `resolve_identity` | 疾患名、既存ID | MONDO、OMIM、Orphanet、MedGen等の同一性・同義語 | 候補間で可 |
| `check_present_hpo` | 正規化疾患ID、Present HPO | 疾患側表現型との関係 | 候補間・他検索と可 |
| `check_absent_hpo` | 正規化疾患ID、Absent HPO | Explicit excluded/否定情報 | 候補間・他検索と可 |
| `search_case_reports` | 疾患ID・名称 | PubCaseFinder症例報告 | 候補間・他検索と可 |
| `search_pubmed` | 疾患ID・名称・HPO | PubMed文献・抄録 | 候補間・他検索と可 |
| `search_contradiction` | 前段Evidence | 表現型・症例・文献の反証 | 前段完了後 |
| `expand_candidates` | 前段の検索結果 | 疾患名・疾患IDの構造化抽出 | 前段完了後 |

### 5.3 固定ルート

```text
resolve_identity
  ↓
check_present_hpo ─┐
check_absent_hpo ──┤
search_case_reports ┤→ 結果統合
search_pubmed ─────┘
  ↓
search_contradiction
  ↓
expand_candidates
```

初期候補から抽出された新規候補には、`resolve_identity`から`search_contradiction`までを実行する。新規候補では`expand_candidates`を実行しない。

## 6. TogoMCP論理アクションの出力

### 6.1 `resolve_identity`

```json
{
  "preferred_id": "MONDO:0000000",
  "preferred_label": "Example syndrome",
  "synonyms": [],
  "cross_references": [],
  "identity_conflicts": [],
  "source_ids": ["S001"]
}
```

### 6.2 `check_present_hpo`

```json
{
  "present_hpo_evaluations": [
    {
      "hpo_id": "HP:0010808",
      "relation": "supports",
      "annotation_status": "annotated",
      "source_ids": ["S002"]
    }
  ]
}
```

### 6.3 `check_absent_hpo`

```json
{
  "absent_hpo_evaluations": [
    {
      "hpo_id": "HP:0001250",
      "relation": "not_annotated",
      "annotation_status": "no_explicit_negative",
      "source_ids": ["S003"]
    }
  ]
}
```

`contradicts`は疾患側の明示的な否定・矛盾に限定する。

### 6.4 `search_case_reports` / `search_pubmed`

各文献結果は、少なくとも次を返す。

```text
entry_id（PMID等）
title
abstractまたは該当抜粋
disease_mentions
patient_relevant_features
source_id
url
```

### 6.5 `search_contradiction`

```json
{
  "contradicting_evidence_ids": ["E010"],
  "searched_questions": ["Absent HPOとの明示的矛盾"],
  "no_contradiction_found": false
}
```

矛盾が見つからない場合も、検索を実行した記録を保存する。

## 7. 新規疾患抽出プロンプト

LLMには、検索結果中の疾患名・疾患IDだけを抽出させる。

```text
あなたは医学文献・データベース結果から疾患エンティティを抽出します。
入力された検索結果に明示された疾患名または疾患IDだけを返してください。
入力に存在しない疾患を追加しないでください。

各候補について、次を返してください。
- disease_name
- normalized_ids
- source_ids
- evidence_ids
- mention_context

既存候補と同一の場合も、出典を返してください。
```

## 8. Reflectionプロンプト

Reflectionは固定検索完了後に一度だけ実行する。

```text
患者情報と候補疾患ごとのEvidenceを確認し、各候補を評価してください。

判断値は次のいずれかです。
- correct
- incorrect
- uncertain

検索の継続や追加ツールの選択は行わないでください。
検索結果にない事実を追加しないでください。
支持根拠と矛盾根拠はEvidence IDで返してください。
sexまたはonsetが不明の場合、その値はunknownです。
```

## 9. 最終リランキングプロンプト

### 9.1 LLM方式

```text
以下の候補とReflection評価を、患者のPresent/Absent HPOとの適合性と
支持・矛盾Evidenceを総合して順位付けしてください。

候補IDを失わず、各順位に次を付けてください。
- candidate_id
- rank
- rationale
- supporting_evidence_ids
- contradicting_evidence_ids
```

### 9.2 ツール平均方式

LLMを呼ばず、保存済みランキングから計算する。

```text
rank_score = 1 - (rank - 1) / (returned_count - 1)
not_returned = 0
tool_average = 実行済みツールのrank_scoreの算術平均
```

画像なしでスキップしたツールは平均から除外する。

## 10. 出典保存

すべての結果は、次の関係で保存する。

```text
ToolCallRecord
  ↓
SourceRecord
  ↓
EvidenceRecord
  ↓
ReflectionAssessment / FinalCandidateResult
```

各Evidenceには、少なくとも次を設定する。

```text
source_ids
database
entry_id
uri
url
query
retrieved_at
raw_response_ref
```

LLMの要約・判断は、必ず元Evidence IDを参照する。

## 11. 並列実行

### 並列化する処理

- 初期5ツール
- 各候補の初期ランキング照合
- `check_present_hpo`、`check_absent_hpo`、`search_case_reports`、`search_pubmed`
- 複数候補の固定検索
- 新規候補の固定検索
- 最終候補ごとの原因候補遺伝子検索

### 依存する処理

- 候補統合は初期ツール結果の後
- 反証検索は前段Evidenceの後
- 新規疾患抽出は前段検索の後
- Reflectionは全検索完了の後
- リランキングはReflectionの後
- 原因候補遺伝子取得は最終ランキングの後

## 12. 実行記録

```text
run_id
action
tool_name
arguments
database
query
source_ids
status
started_at
elapsed_ms
error
```

API/MCPの一部失敗は候補全体の失敗にせず、該当アクションとSourceRecordに記録する。
