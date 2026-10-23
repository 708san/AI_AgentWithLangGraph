# 文献根拠付き・日英併記診断フロー仕様 v2

> **現在の最小仕様**: この文書の旧記述にある `patient_matches`、`limitations`、
> `impact`、`relationship`、`match_level`、`reasoning`、`reference_catalog` は廃止する。
> 現在の正式な公開対応は、診断文の `[n]`、`citation_numbers`、および
> `finalDiagnosis.references[number=n]` のみとする。Referenceにはタイトル、URL、
> 文献主張、原文抜粋を含める。診断ツールは候補順位の補助情報であり、文献Referenceには含めない。
> 実行順は `tool -> tentative -> literature search/Evidence -> reflection -> final` を維持する。

## 1. 目的と適用範囲

本仕様は、希少疾患診断エージェントを次の状態へ移行するための実装契約である。

- 候補生成ツールのrankと医学文献の根拠を分離する。
- PubMed等の文献を候補統合後に検索する。
- 文献の全文要約を最終診断へ直接渡すのではなく、診断に必要な項目をEvidenceとして構造化する。
- 診断理由を英語と日本語で同時に出力する。
- 出典をURL、source_type、識別子、取得内容、Evidenceとの紐付け付きで返す。
- PhenotypeSearchは候補rankには使うが、文献Referenceには含めない。
- OMIM IDをすべて`OMIM:数字`形式に統一する。

本仕様は診断支援を対象とし、分子診断の確定、治療指示、予後判断、医療行為の代替を目的としない。

## 2. 用語と責務

### 2.1 Candidate

各診断候補を表す。Candidateはツール出力と文献Evidenceの共通の紐付け先になる。

### 2.2 CandidateSupport

PCF、GestaltMatcher、ZeroShot、PhenoBrain、PhenotypeSearchなどが候補を支持した情報。rankやscoreを含むが、医学文献Citationではない。

### 2.3 SourceRecord

PubMed、PMC、OMIM、ClinGen、診断ツールなど、取得元そのものの記録。URL、識別子、取得原文を保持する。

### 2.4 EvidenceItem

SourceRecordの内容から、診断に必要な主張を項目別に抽出したもの。患者との一致・不一致、支持・反証を表す。

### 2.5 Reference

公開レスポンスで表示する出典情報。内部的には`SourceRecord`から決定論的に生成する。LLMが自由文でReferenceを生成することは正式経路としない。

## 3. 設計原則

1. 候補生成、文献検索、Evidence抽出、診断文章生成を別工程にする。
2. LLMに渡す医学的根拠は、候補に紐付いたEvidenceだけにする。
3. LLMが原文にない具体的な主張を補完しないよう、Evidence IDを必須にする。
4. 文献不足と医学的矛盾を区別する。
5. 未報告の所見を陰性所見として扱わない。
6. 英語と日本語は同じ主張、同じEvidence ID、同じ確信度に対応させる。
7. OMIM ID、Candidate ID、Source ID、Evidence IDは実行中に不変とする。
8. Reference、Evidence、CandidateSupportの3種類を混ぜない。

## 4. Canonical ID

### 4.1 OMIM ID

OMIM IDの正規形は常に次とする。

```text
OMIM:123456
```

内部State、保存JSON、LLM入力、公開レスポンスのすべてでこの形式を使う。数字のみの値は入力境界で正規化する。

### 4.2 Candidate ID

OMIM IDが存在する場合はOMIM IDをCandidate IDにする。

```text
candidate_id = "OMIM:123456"
```

OMIM IDがない場合は、正規化疾患名から次を作る。

```text
candidate_id = "DISEASE:<normalized-disease-name>"
```

Candidate IDはEvidenceの紐付けに使用し、`candidate_OMIM_id`だけに依存しない。

### 4.3 その他のID

- PubMed: `PMID:12345678`
- DOI: `DOI:<doi>`
- OMIM: `OMIM:123456`
- Evidence: `EVD-000001`
- Source: `SRC-000001`
- ツール出力: `TOOL-<run-id>-<candidate-id>`

## 5. データモデル

以下は実装の正式モデルである。`Literal`、`Optional`、`Union`、`Field`はPydanticモデルに合わせて実装する。

移行期間中は、現行の`DiagnosisFormat`、`DiagnosisOutput`、`ReflectionFormat`、`ReflectionOutput`を直接変更しない。v2モデルを`DiagnosisFormatV2`、`DiagnosisOutputV2`、`ReflectionFormatV2`、`ReflectionOutputV2`として追加し、`evidence_mode`に応じて使用する。以下の`Legacy*`は概念上の別名であり、実装では次の対応を明示する。

```python
LegacyDiagnosisOutput = DiagnosisOutput
LegacyReflectionOutput = ReflectionOutput
LegacyMergedDiseaseCandidate = MergedDiseaseCandidate  # current TypedDict
```

legacyモデルを削除できる段階で、必要ならv2モデルを正式名称へrenameする。

### 5.1 CandidateSupport

```python
class CandidateSupport(BaseModel):
    tool: Literal[
        "PubCaseFinder",
        "GestaltMatcher",
        "ZeroShot",
        "PhenoBrain",
        "PhenotypeSearch",
    ]
    rank: Optional[int] = None
    score: Optional[float] = None
    rank_source: Optional[Literal["api", "agent_enumeration", "model", "search"]] = None
    raw_distance: Optional[float] = None
    matched_hpo_id: Optional[str] = None
    note: Optional[str] = None
    citation_allowed: bool = False
```

`citation_allowed`は診断ツールについて常に`False`とする。ツールのrankは候補支持情報であり、文献Referenceではない。

### 5.2 MergedDiseaseCandidateV2

```python
class MergedDiseaseCandidateV2(BaseModel):
    candidate_id: str
    disease_name: str
    OMIM_id: Optional[str] = None
    gene_symbols: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    phenotype_labels: list[str] = Field(default_factory=list)
    consensus_count: int
    best_rank: int
    tool_supports: list[CandidateSupport] = Field(default_factory=list)
```

`gene_symbols`、`synonyms`、`phenotype_labels`は文献検索クエリ生成用であり、候補統合後に疾患レジストリまたは既存マッピングから補完する。

`best_rank`は候補統合時の最小rankであり、rankを返さない入力では決められたfallback値を使用する。GestaltMatcherのrankは既存APIのrankを優先して保持し、APIにrankがない場合だけレスポンス順の順位を割り当てる。その場合は`CandidateSupport.rank_source="agent_enumeration"`とする。

現行の`agent/state/state_types.py`にある`MergedDiseaseCandidate`はTypedDictであり、既存の`.get()`利用を含むlegacy経路の入力型として残す。v2では`MergedDiseaseCandidateV2`へ変換し、v2ノードはPydantic属性アクセスを使う。v1 TypedDictを同名のPydanticモデルへ直接置換してはならない。

### 5.3 SourceRecord

```python
class SourceRecord(BaseModel):
    source_id: str
    source_type: Literal[
        "literature",
        "database",
        "diagnostic_tool",
        "web_search",
    ]
    source_name: str
    title: str
    url: Optional[str] = None
    identifier: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_year: Optional[str] = None
    candidate_ids: list[str] = Field(default_factory=list)
    disease_name: Optional[str] = None
    OMIM_id: Optional[str] = None
    gene_symbols: list[str] = Field(default_factory=list)
    content_type: Literal[
        "abstract",
        "full_text",
        "database_record",
        "tool_result",
        "search_snippet",
    ]
    raw_content: str
    search_query: Optional[str] = None
    retrieved_at: Optional[str] = None
    # Fail closed. The source policy explicitly enables citation per source.
    citation_allowed: bool = False
```

`raw_content`は取得元のabstract、本文、データベースレコードなどを保存する。LLM生成要約を`raw_content`として保存してはならない。公開レスポンスではこのフィールドを「出典コンテンツ」として返し、全文取得時はサイズ上限・利用条件・個人情報の扱いを実装側で適用する。

Source種別の扱い:

| source_type | 例 | 最終Reference |
|---|---|---|
| `literature` | PubMed、PMC | 主要根拠として使用可 |
| `database` | OMIM、ClinGen | 内部の候補補助情報。番号付きReferenceには含めない |
| `diagnostic_tool` | PCF、GestaltMatcher、PhenotypeSearch | CandidateSupportのみ。Reference不可 |
| `web_search` | DDGS | 補助情報。主要根拠にはしない |

PhenotypeSearchのdefinitionやembeddingスコアをSourceRecordのReferenceとして出してはならない。

`citation_allowed`はデフォルトを`False`とし、取得時に次のポリシーで明示設定する。

- `literature`: 有効なURLまたは識別子を持つ場合は`True`
- `database`: `False`。候補補助情報として保持するが、番号付きReferenceには含めない
- `diagnostic_tool`: 常に`False`
- `web_search`: 原則`False`

### 5.4 EvidenceItem

```python
class EvidenceItem(BaseModel):
    evidence_id: str
    candidate_id: str
    source_id: str
    evidence_type: Literal[
        "disease_identity",
        "gene_and_inheritance",
        "core_phenotype",
        "distinctive_phenotype",
        "negative_finding",
        "phenotype_overlap",
        "phenotype_mismatch",
        "disease_mechanism",
    ]
    claim_en: str
    claim_ja: str
    excerpt: str
    support: Literal["supports", "contradicts", "unknown"]
    confidence: Literal["high", "medium", "low"]
```

`excerpt`はSourceRecordの原文に存在する短い根拠箇所である。根拠が原文で確認できない場合、EvidenceItemを作成しない。したがって`excerpt`は空文字列にせず、必須の非空値とする。

### 5.5 Reflection用Evidence評価

```python
class EvidenceAssessment(BaseModel):
    evidence_id: str
    relevance: Literal["high", "medium", "low"]
    supports_patient: Literal["yes", "no", "unknown"]
    reason_en: str
    reason_ja: str
```

### 5.6 診断理由

```python
class DiagnosisFormatV2(BaseModel):
    candidate_id: str
    disease_name_en: str
    disease_name_ja: str
    OMIM_id: Optional[str] = None
    rank: int
    description_en: str
    description_ja: str
    evidence_ids: list[str] = Field(default_factory=list)
    literature_status: Literal[
        "supported",
        "insufficient",
        "contradictory",
        "not_retrieved",
    ]
```

`source_ids`はLLM出力として持たせない。`evidence_ids`から参照先SourceRecordを決定論的に導出する。

現行の`tool_rankings`は移行期間中の入力aliasとして読み取ってよいが、保存・出力する正式フィールドは`tool_supports`とする。

後方互換のため、既存APIの`disease_name`、`description`、`reference`を一時的に返してもよい。ただし以下の扱いに限定する。

- `disease_name` = `disease_name_en`のalias
- `description` = `description_en`のalias
- `reference` = `references`の表示用legacy rendering
- 診断ロジックは新フィールドだけを参照する

### 5.7 DiagnosisOutputV2

```python
class DiagnosisOutputV2(BaseModel):
    ans: list[DiagnosisFormatV2]
    references: list["ReferenceRecord"] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    validation_status: Literal["passed", "failed", "partial"] = "passed"
    validation_errors: list[str] = Field(default_factory=list)
    reference: Optional[str] = None  # legacy rendering only
```

`references`と`evidence`が正式な出典経路である。`reference`の自由文は診断生成の入力に再利用しない。

### 5.8 ReferenceRecord

`ReferenceRecord`は`SourceRecord`の公開用projectionであり、Evidenceとの関係を表示するための`evidence_ids`を持つ。内部の原本は常に`SourceRecord`である。

```python
class ReferenceRecord(SourceRecord):
    evidence_ids: list[str] = Field(default_factory=list)
```

### 5.9 ReflectionFormatV2

```python
class ReflectionFormatV2(BaseModel):
    candidate_id: str
    disease_name_en: str
    disease_name_ja: str
    correctness: bool
    patient_summary_en: str
    patient_summary_ja: str
    diagnosis_analysis_en: str
    diagnosis_analysis_ja: str
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    literature_status: Literal[
        "supported",
        "insufficient",
        "contradictory",
        "not_retrieved",
    ]

```

```python
class ReflectionOutputV2(BaseModel):
    ans: list[ReflectionFormatV2]
```

旧Reflectionの`Correctness`、`PatientSummary`、`DiagnosisAnalysis`、`references`は、Pydanticの正式フィールドaliasにはしない。移行adapterで新フィールドへ変換し、内部ロジックは新フィールドだけを参照する。

`DiagnosisOutputV2`は`ReferenceRecord`を前方参照するため、Pydantic v2では全モデル定義後に`DiagnosisOutputV2.model_rebuild()`を実行する。実装では`ReferenceRecord`を先に定義して前方参照を避けてもよい。

## 6. State契約

既存フィールドは段階移行中も保持する。新しく正式なフィールドを追加する。

```python
class State(TypedDict):
    # existing input and candidate fields
    depth: int
    evidence_mode: Literal["legacy", "v2"]
    hpoList: List[str]
    hpoDict: dict[str, str]
    absentHpoList: List[str]
    absentHpoDict: dict[str, str]
    imagePath: Optional[str]
    use_absentHPO: bool
    use_phenobrain: bool
    filter_impotance: bool
    onset: Optional[str]
    sex: Optional[str]
    patient_id: Optional[str]
    llm: Optional[AzureOpenAIWrapper]

    pubCaseFinder: List[PCFres]
    GestaltMatcher: List[GestaltMatcherFormat]
    zeroShotResult: Optional[ZeroShotOutput]
    phenoBrain: List[PhenoBrainResult]
    phenotypeSearchResult: Optional[List[PhenotypeSearchFormat]]
    mergedDiseaseCandidates: List[Union[LegacyMergedDiseaseCandidate, MergedDiseaseCandidateV2]]

    # CandidateSupport is nested in each mergedDiseaseCandidates item.
    sources: List[SourceRecord]
    evidence: List[EvidenceItem]

    # optional auxiliary web results; not primary literature evidence
    webresources: List[webresource]

    tentativeDiagnosis: Optional[Union[LegacyDiagnosisOutput, DiagnosisOutputV2]]
    reflection: Optional[Union[LegacyReflectionOutput, ReflectionOutputV2]]
    finalDiagnosis: Optional[Union[LegacyDiagnosisOutput, DiagnosisOutputV2]]
    validation_status: Literal["not_run", "passed", "failed", "partial"]
    validation_errors: List[str]
    flow_iteration: int
    max_flow_iterations: int

    # compatibility during migration; not authoritative for new diagnosis logic
    memory: List[InformationItem]
    clinicalText: Optional[str]
```

`CandidateSupport`の正式な保存場所はv2の`mergedDiseaseCandidates[*].tool_supports`である。トップレベルに重複した`candidateSupport`配列は持たせない。legacy TypedDictでは移行期間中のみ`tool_rankings`を読み取り、v2境界で`tool_supports`へ変換する。

`memory`は旧経路との互換用に残すが、新しい`createDiagnosisNode`、`reflectionNode`、`finalDiagnosisNode`は`evidence`と`sources`を使う。

初期Stateでは、公開実行のデフォルトとして`evidence_mode="v2"`、`sources=[]`、`evidence=[]`、`flow_iteration=0`、`max_flow_iterations=1`、`validation_status="not_run"`として初期化する。旧ルートは`evidence_mode="legacy"`を明示した場合だけ選択する。

## 7. 目標フロー

```text
入力HPO・画像・発症時期・性別
        |
        +--> HPO辞書作成
        +--> PCF --> normalize
        +--> GestaltMatcher --> normalize/rank
        +--> ZeroShot --> normalize
        +--> PhenoBrain（有効時）
        +--> PhenotypeSearch
        +--> HPO Web検索（補助、任意）
        |
        v
候補統合・Candidate ID付与
        |
        v
候補メタデータ補完
  - OMIM
  - gene symbols
  - synonyms
  - phenotype labels
        |
        v
文献検索
  - CandidateごとにPubMed/PMCを検索
        |
        v
Evidence抽出・候補紐付け
        |
        v
文献根拠付き暫定診断
        |
        v
Reflection
        |
        v
文献根拠付き最終診断
        |
        v
最終正規化・出力検証
```

### 7.1 既存ノードとの対応

| 役割 | 現行ノード | v2方針 |
|---|---|---|
| PCF | `PCFnode` | 維持。rankとOMIMを正規化 |
| GestaltMatcher | `GestaltMatcherNode` | rank、rank_source、raw_distanceを保持 |
| ZeroShot | `createZeroShotNode` | 候補生成として維持 |
| PhenoBrain | `PhenoBrainNode` | 有効時のみ維持 |
| PhenotypeSearch | `DiseaseSearchWithHPONode` | 候補rankのみ。Reference不可 |
| HPO Web | `HPOwebSearchNode` | 補助検索。主要文献根拠にしない |
| 候補統合 | `mergeCandidateResultsNode` | Candidate IDと候補メタデータを追加 |
| 文献検索 | 旧`diseaseSearchNode` | `literatureSearchNode`へ置換。候補統合後に実行 |
| Evidence抽出 | なし | `evidenceExtractionNode`を新設 |
| 暫定診断 | `createDiagnosisNode` | `sources/evidence`のみを医学根拠として使用 |
| Reflection | `reflectionNode` | Evidence IDを評価。日英出力 |
| 最終診断 | `finalDiagnosisNode` | Evidence ID必須。日英出力 |
| 出力検証 | なし | `diagnosisOutputValidationNode`を新設 |

旧`diseaseSearchNode`は、暫定診断後に`memory`を検索する経路としては廃止する。互換期間中は残してもよいが、新グラフからは呼び出さない。

## 8. 文献検索仕様

### 8.1 検索入力

各Candidateから次を生成する。

- 正規化疾患名
- OMIM ID
- gene symbols
- synonyms
- 上位の特徴的HPOラベル
- 疾患名と特徴的HPOの組み合わせ

OMIM、遺伝子、別名が取得できない場合は、疾患名と患者HPOだけで検索し、`literature_status="insufficient"`の可能性を保持する。

### 8.2 クエリプラン

Candidateごとに、以下を別クエリとして生成する。1本の長大なクエリにすべてを詰め込まない。

```text
Q1: "disease name"
Q2: "disease name" AND (phenotype OR clinical features OR syndrome)
Q3: "disease name" AND gene_symbol
Q4: "OMIM:123456" OR "123456"
Q5: "synonym" AND distinctive_phenotype
Q6: gene_symbol AND hpo_feature_1 AND hpo_feature_2
```

### 8.3 検索ソースの優先順位

1. PubMed/PMCで疾患名・OMIM・遺伝子が一致する文献
2. PubMed/PMCで疾患名と特徴的表現型が一致する文献
3. GeneReviews、ClinGen、OMIM等の専門データベース
4. 一般Web検索

WikipediaやDDGSは背景情報・追加候補探索には利用できるが、最終診断の主要なliterature Evidenceにはしない。

### 8.4 取得データ

PubMed/PMCから少なくとも次を保存する。

- PMID
- DOI（取得できる場合）
- title
- authors（取得できる場合）
- publication year
- journal
- abstractまたは取得本文
- PubMed/PMC URL
- 使用した検索クエリ
- candidate_id

LLMによる要約を原文の代替として保存してはならない。

PMCIDがPubMed XMLから取得できる場合は、[NCBI E-utilitiesのPMC
EFetch](https://www.ncbi.nlm.nih.gov/entrez/query/static/efetch_help.html)の
`db=pmc`でPMC全文XMLを取得する。PMCの自動取得経路としては、[PMC公式の
Developer向け案内](https://pmc.ncbi.nlm.nih.gov/tools/developers/)に従う。
本文の取得に成功したSourceRecordは、
`source_name="PMC"`、`content_type="full_text"`、
`identifier="PMCID:PMC..."`、PMC記事URLを持つ。本文が取得できない場合、
PMCIDがない場合、または本文が空の場合は、同じPMIDのPubMed抄録を
`source_name="PubMed"`、`content_type="abstract"`として保存する。
いずれの場合も取得元の原文を保存し、LLM要約で置き換えない。全文は
環境変数で上限を設定し、取得失敗は診断フロー全体の失敗にせず抄録へ
フォールバックする。

同一文献の重複判定にはPMID、PMCID、DOIを使う。PMC全文が利用できる
場合は抄録レコードとの二重登録を避け、Evidenceは全文SourceRecordに
紐付ける。

### 8.5 Entity matching

検索結果が別疾患、別遺伝子、別サブタイプを扱っていないか確認する。疾患同一性が確認できない文献は、候補のEvidenceとして採用しない。

Entity matchingの判定結果自体はEvidenceではなく検索メタデータとして保存する。

## 9. Evidence抽出仕様

Evidence抽出では、文献全文を短く要約するのではなく、以下の項目を個別に抽出する。

- 疾患同一性
- 責任遺伝子
- 遺伝形式
- コア表現型
- 特徴的・鑑別上有用な表現型
- 明示された陰性所見
- 病態機序
- 患者HPOとの一致
- 患者HPOとの不一致

各Evidenceは、原文抜粋、候補ID、Source ID、support状態、confidenceを持つ。

病態機序は、文献Evidenceに存在する場合だけ診断理由に出す。文献にない場合は「文献で直接確認できない」とする。

## 10. 診断・Reflection仕様

### 10.1 診断文章の必須内容

英語・日本語の両方に以下を含める。

- 患者の主要HPO所見
- 疾患側の特徴的表現型
- 一致点
- 不一致点または未報告で判断できない点
- CandidateSupportのrank/score
- Evidence IDに対応する文献根拠
- literature_status
- 確定診断ではないこと

英日でEvidence IDと確信度を一致させる。日本語訳で新しい医学的主張を追加してはならない。

### 10.2 文献ステータス

```text
supported
  1つ以上のliterature Evidenceが候補と患者所見を支持する

insufficient
  文献が少ない、または候補同一性を確認できないが、ツール支持や表現型一致により候補を残す

contradictory
  文献または明示的陰性所見が疾患の中核と明確に矛盾する

not_retrieved
  文献検索が実行されなかった、または検索失敗した
```

`insufficient`は候補維持を許可するが、文献支持済みとして扱ってはならない。

最終診断の「支持されている」という表現は、少なくとも1つの`literature` SourceRecord（PubMed/PMC）に紐付くEvidenceItemがある候補に限定する。OMIMなどの`database`は内部の疾患名・ID・遺伝子同定の補助には使えるが、番号付きReferenceや臨床的な最終支持を単独で与えない。`insufficient`や`not_retrieved`の候補を返す場合は、文献根拠が不足していることを英語・日本語の両方で明示する。

### 10.3 Reflection

- 文献不足だけで`correctness=False`にしない。
- ツール支持による候補維持と文献支持済みを分離する。
- `contradictory`の場合は最終候補から除外または強く低順位化する。
- `insufficient`の場合は診断文章に文献不足を明示する。
- 明示的陰性所見と未報告を区別する。

既存の「文献不足ならツールrankでCorrectnessをTrueへoverrideする」処理は廃止する。代わりに`correctness`と`literature_status`を別々に判定する。

## 11. 出力Reference仕様

`DiagnosisOutputV2.references`を正式なReference一覧とする。

```json
{
  "source_id": "SRC-000001",
  "source_type": "literature",
  "source_name": "PubMed",
  "identifier": "PMID:12345678",
  "title": "Article title",
  "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
  "content_type": "abstract",
  "raw_content": "Retrieved abstract or source content",
  "candidate_ids": ["OMIM:123456"],
  "citation_allowed": true,
  "evidence_ids": ["EVD-000001", "EVD-000002"]
}
```

`evidence_ids`は`ReferenceRecord`でのみ公開する。正規の関係は`EvidenceItem.source_id`から決定論的に導出し、SourceRecordに同じ配列を保存して二重管理しない。

PhenotypeSearch、PCF、GestaltMatcherなどのCandidateSupportはこの一覧に入れない。

## 12. 出力検証

`diagnosisOutputValidationNode`で以下を検証する。

1. OMIM IDがすべて`OMIM:数字`形式である。
2. Candidate IDがすべて一意である。
3. rankが1始まりで重複していない。
4. 英語と日本語の診断理由が両方存在する。
5. 英語と日本語のEvidence ID集合が一致する。
6. 英語と日本語のEvidence confidence/supportが一致する。
7. Evidence IDが実在する。
8. Source IDが実在する。
9. `EvidenceItem.source_id`がSourceRecordに対応する。
10. `citation_allowed=False`のSourceがReferenceに含まれない。
11. PhenotypeSearchがReference一覧に含まれない。
12. `literature_status="supported"`の候補にはliterature Evidenceがある。
13. `literature_status="insufficient"`の候補には不足状態の説明がある。
14. `literature_status="contradictory"`の候補が最終上位に残っていない。
15. 最終候補のCandidate IDが`mergedDiseaseCandidates`に存在する。
16. ReferenceのURL、識別子、Source typeがSourceRecordと一致する。
17. `supported`の最終候補が、同一Candidate IDに紐付く`literature` Evidenceを少なくとも1つ持つ。
18. 最終診断の各主張が`evidence_ids`からReferenceへ追跡でき、ツールrankだけを根拠にしていない。

失敗時は黙って修正せず、`validation_status="failed"`と`validation_errors`を返す。候補を除外する場合も、除外理由を検証エラーまたは診断メタデータに残す。

## 13. LangGraph実装変更

### 13.1 新規ノード

```text
candidateMetadataNode
literatureSearchNode
evidenceExtractionNode
diagnosisOutputValidationNode
```

### 13.2 推奨エッジ

```text
mergeCandidateResultsNode
  -> candidateMetadataNode
  -> literatureSearchNode
  -> evidenceExtractionNode
  -> createDiagnosisNode
  -> diseaseNormalizeNode
  -> reflectionNode
  -> finalDiagnosisNode
  -> diseaseNormalizeForFinalNode
  -> diagnosisOutputValidationNode
  -> END
```

`HPOwebSearchNode`は候補生成と並列に実行してもよいが、`sources/evidence`の主要literature経路には含めない。

### 13.3 Reflection後の反復

現行の`depth > 0`判定は初回から最終診断へ進み、反復条件を実質的に無効化する。v2では`depth`を検索深度と反復制御に兼用しない。

以下を追加する。

```text
flow_iteration: int
max_flow_iterations: int
```

Reflection後は次の条件分岐で判定する。

- 有効な候補があり、Evidenceが十分: finalDiagnosisNode
- Evidence不足だが再検索可能: `flow_iteration`を1増加させてliteratureSearchNodeへ一度だけ再試行
- 明確な矛盾のみ: 候補を除外してfinalDiagnosisNode
- 最大反復回数到達: finalDiagnosisNodeへ進み、状態を明示

再検索時はCandidate metadataを再取得せず、既存`SourceRecord`をidentifierまたはURLで重複排除して追加する。再検索後は`evidenceExtractionNode`を再実行し、同一`evidence_id`を上書きしない。

再検索の遷移は次の通りとする。

```text
reflectionNode
  -> literatureSearchNode
  -> evidenceExtractionNode
  -> createDiagnosisNode
  -> diseaseNormalizeNode
  -> reflectionNode
```

再検索後に`finalDiagnosisNode`へ直接進んではならない。新しいEvidenceを使って暫定診断とReflectionを更新してから最終診断へ進む。

`diseaseNormalizeNode`と`diseaseNormalizeForFinalNode`は、v2モデルの`candidate_id`、`evidence_ids`、`literature_status`、日英文章を削除・再生成せず、疾患名とOMIM形式だけを正規化する。

## 14. 後方互換方針

Phase 1では既存フィールドを読み取り可能にするが、新ロジックの正式入力は次とする。

```text
sources
evidence
mergedDiseaseCandidates
```

以下は移行期間のみ維持する。

```text
memory
clinicalText
DiagnosisOutput.reference
DiagnosisFormat.description
ReflectionFormat.Correctness
ReflectionFormat.PatientSummary
ReflectionFormat.DiagnosisAnalysis
ReflectionFormat.references
```

旧フィールドと新フィールドの両方が存在する場合、新フィールドを優先する。

`evidence_mode="legacy"`では現行ノードの診断ロジックと旧フィールドを維持し、`evidence_mode="v2"`で新しい`sources`、`evidence`、日英診断、Reference検証を有効にする。ただしOMIMの公開フィールドは移行モードに関係なく`OMIM:数字`へ統一する。数値だけを必要とする既存利用者向けの互換値が必要な場合は別名フィールドで提供し、`OMIM_id`自体を数値へ戻してはならない。

## 15. 実装フェーズと完了条件

### Phase 1: 型と正規化

- `CandidateSupport`、`MergedDiseaseCandidateV2`を追加し、legacy TypedDict adapterを実装
- `SourceRecord`、`EvidenceItem`を追加
- Stateに`sources`、`evidence`を追加
- OMIM IDの入力・出力正規化
- GestaltMatcherのrank、rank_source、raw_distanceを追加

### Phase 2: 検索

- `candidateMetadataNode`を追加
- Candidateごとの疾患名、OMIM、遺伝子、別名、HPOからクエリ生成
- PubMed/PMCのmetadataと原文をSourceRecordへ保存
- 旧`diseaseSearchNode`を新グラフから外す

### Phase 3: Evidence抽出

- `evidenceExtractionNode`を追加
- 文献から項目別Evidenceを抽出
- 原文抜粋とEvidence IDを保持
- EvidenceをCandidate IDへ紐付け

### Phase 4: 診断生成

- 暫定診断、Reflection、最終診断をEvidence入力へ変更
- 英語・日本語を同時出力
- `evidence_ids`を必須化
- 文献不足・矛盾を出力

### Phase 5: 検証とテスト

- `diagnosisOutputValidationNode`を追加
- OMIM形式テスト
- Candidate ID紐付けテスト
- PubMed PMID/URL/raw_content保持テスト
- 日英Evidence ID一致テスト
- PhenotypeSearch非Citationテスト
- 文献不足・疾患identity mismatchテスト
- Reflection反復制御テスト

完了条件は次のすべてを満たすこととする。

1. 最終診断が候補統合後に取得したEvidenceを入力として生成される。
2. 各診断理由が英語・日本語で返る。
3. ReferenceにURL、source_type、識別子、取得内容、Evidenceとの紐付けがある。
4. PhenotypeSearchはrankには使われるがReferenceには含まれない。
5. GestaltMatcherのrankが常に返る。
6. OMIM IDが常に`OMIM:数字`形式で返る。
7. 文献にない断定的な根拠文章が最終出力に残らない。
8. 既存API利用者向けのlegacy aliasが新フィールドと矛盾しない。
