# agent ディレクトリ仕様書

## 1. 概要

`agent/` は、HPO ID、患者顔画像、発症時期、性別などを入力として希少疾患の鑑別診断を支援する LangGraph ベースのエージェントである。

主な処理は次の外部・内部情報源を統合して診断候補を作成し、文献検索と自己評価を経て最終診断を出力する。

- HPO ID から HPO ラベルへの変換
- PubCaseFinder API による OMIM 疾患候補検索
- GestaltMatcher API による顔画像ベース疾患候補検索
- Azure OpenAI による zero-shot 診断、統合診断、reflection、最終診断
- DDGS、Wikipedia、PubMed による外部知識検索
- Azure OpenAI Embedding と FAISS による疾患名正規化、表現型類似疾患検索
- ノード実行結果のログ・JSON 保存

## 2. パイプライン外部仕様

### 2.1 エントリポイント

対象ファイル: `agent/agent_pipeline.py`

主要クラス:

```python
RareDiseaseDiagnosisPipeline(model_name="gpt-4o", enable_log=False, log_filename=None)
```

実行メソッド:

```python
run(
    hpo_list,
    image_path=None,
    verbose=False,
    absent_hpo_list=None,
    onset=None,
    sex=None,
    patient_id=None,
    use_absentHPO=False,
    filter_impotance=False,
)
```

### 2.2 入力

| 引数 | 型 | 必須 | 内容 |
|---|---:|---:|---|
| `hpo_list` | `List[str]` | 必須 | 患者に存在する HPO ID のリスト。例: `["HP:0001263"]` |
| `image_path` | `Optional[str]` | 任意 | 顔画像ファイルのパス。未指定時は GestaltMatcher をスキップ |
| `verbose` | `bool` | 任意 | `True` の場合、reflection と finalDiagnosis を標準出力に整形表示 |
| `absent_hpo_list` | `Optional[List[str]]` | 任意 | 明示的に観察されなかった HPO ID のリスト |
| `onset` | `Optional[str]` | 任意 | 発症時期。未指定時は `"Unknown"` |
| `sex` | `Optional[str]` | 任意 | 性別。未指定時は `"Unknown"` |
| `patient_id` | `Optional[str]` | 任意 | 結果保存ファイル名に使用。未指定時は `"unknown"` |
| `use_absentHPO` | `bool` | 任意 | `True` の場合のみ、明示的に観察されなかった HPO 所見を LLM プロンプトに含める。既定値は `False` |
| `filter_impotance` | `bool` | 任意 | `True` の場合、present HPO と absent HPO を関連疾患数が少ない上位 15 件に絞ってから実行する。既定値は `False` |

### 2.3 出力

`run()` は LangGraph 実行後の `State` 辞書を返す。

主な出力キー:

| キー | 型 | 内容 |
|---|---|---|
| `hpoDict` | `dict[str, str]` | 入力 HPO ID から HPO ラベルへの辞書 |
| `absentHpoDict` | `dict[str, str]` | 明示的に観察されなかった HPO ID から HPO ラベルへの辞書 |
| `pubCaseFinder` | `List[dict]` | PubCaseFinder の候補疾患。正規化後は `disease_name` を含む |
| `GestaltMatcher` | `List[dict]` | GestaltMatcher の候補疾患。画像なしでは空リスト |
| `zeroShotResult` | `ZeroShotOutput` | LLM が HPO から直接推定した候補疾患 |
| `phenotypeSearchResult` | `List[PhenotypeSearchFormat]` | HPO ラベルを embedding 検索した類似疾患 |
| `mergedDiseaseCandidates` | `List[MergedDiseaseCandidate]` | 診断前に各ツール候補を疾患単位で統合した候補表 |
| `webresources` | `List[webresource]` | HPO 由来の Web 検索結果要約 |
| `tentativeDiagnosis` | `DiagnosisOutput` | 各ツール結果を統合した暫定診断 |
| `memory` | `List[InformationItem]` | 暫定診断疾患に関する Wikipedia/PubMed 知識 |
| `reflection` | `ReflectionOutput` | 暫定診断の妥当性評価 |
| `finalDiagnosis` | `DiagnosisOutput` | 最終診断 |

### 2.4 副作用

| 条件 | 出力先 | 内容 |
|---|---|---|
| `enable_log=True` | `log/agent_log_YYYYMMDD_HHMMSS.log` または指定ファイル | グラフ構造、ノード結果、LLM プロンプト |
| 一部ノード実行時 | `res/{patient_id}.json` | ノード結果を JSON で逐次マージ保存 |
| 常時 | 標準出力 | ノード名、進捗、エラー、プロファイル時間 |

## 3. State 仕様

対象ファイル: `agent/state/state_types.py`

### 3.1 State

`State` は LangGraph の共有状態である。

| キー | 型 | 内容 |
|---|---|---|
| `depth` | `int` | フロー反復深度。初期値 0、`BeginningOfFlowNode` で加算 |
| `imagePath` | `Optional[str]` | 顔画像パス |
| `clinicalText` | `Optional[str]` | 類似症例などの追加臨床テキスト。現行初期値は `None` |
| `hpoList` | `List[str]` | 入力 HPO ID |
| `hpoDict` | `dict[str, str]` | HPO ID とラベルの対応 |
| `absentHpoList` | `List[str]` | 明示的に観察されなかった HPO ID |
| `absentHpoDict` | `dict[str, str]` | 明示的に観察されなかった HPO ID とラベルの対応 |
| `use_absentHPO` | `bool` | `absentHpoDict` を LLM プロンプトに含めるかどうか |
| `filter_impotance` | `bool` | HPO 重要度フィルタを適用したかどうか |
| `pubCaseFinder` | `List[PCFres]` | PubCaseFinder 結果 |
| `GestaltMatcher` | `List[GestaltMatcherFormat]` | GestaltMatcher 結果 |
| `phenotypeSearchResult` | `Optional[List[PhenotypeSearchFormat]]` | 表現型 embedding 検索結果 |
| `mergedDiseaseCandidates` | `List[MergedDiseaseCandidate]` | 各ツールの疾患候補と順位情報を統合した診断入力 |
| `webresources` | `List[webresource]` | HPO Web 検索結果 |
| `memory` | `List[InformationItem]` | 疾患知識検索結果 |
| `zeroShotResult` | `Optional[ZeroShotOutput]` | zero-shot 診断 |
| `tentativeDiagnosis` | `Optional[DiagnosisOutput]` | 暫定診断 |
| `reflection` | `Optional[ReflectionOutput]` | 診断評価 |
| `finalDiagnosis` | `Optional[DiagnosisOutput]` | 最終診断 |
| `onset` | `Optional[str]` | 発症時期 |
| `sex` | `Optional[str]` | 性別 |
| `patient_id` | `Optional[str]` | 保存用患者 ID |
| `llm` | `Optional[AzureOpenAIWrapper]` | LLM ラッパー |

### 3.2 Pydantic モデル

| モデル | 内容 |
|---|---|
| `ZeroShotFormat` | `disease_name`, `rank`, `OMIM_id` |
| `ZeroShotOutput` | `ans: List[ZeroShotFormat]` |
| `DiagnosisFormat` | `disease_name`, `OMIM_id`, `description`, `rank` |
| `DiagnosisOutput` | `ans: List[DiagnosisFormat]`, `reference` |
| `ReflectionFormat` | `disease_name`, `Correctness`, `PatientSummary`, `DiagnosisAnalysis`, `references` |
| `ReflectionOutput` | `ans: List[ReflectionFormat]` |
| `GestaltMatcherFormat` | `subject_id`, `syndrome_name`, `omim_id`, `image_id`, `score` |
| `ToolRankingItem` | `tool`, `rank`, `score`, `matched_hpo_id`, `note` |
| `MergedDiseaseCandidate` | `disease_name`, `OMIM_id`, `consensus_count`, `best_rank`, `tool_rankings` |
| `OMIMEntry` | `OMIM_id`, `disease_name`, `synonym`, `definition`, `phenotype` |
| `PhenotypeSearchFormat` | `disease_info: OMIMEntry`, `similarity_score` |

## 4. グラフ処理仕様

対象ファイル: `agent/agent_pipeline.py`, `agent/nodes.py`

### 4.1 ノード一覧

| ノード | 入力 | 出力 | 処理 |
|---|---|---|---|
| `BeginningOfFlowNode` | `depth` | `depth`, `tentativeDiagnosis=None`, `reflection=None` | フロー開始。`depth` を 1 加算し診断・評価をリセット |
| `PCFnode` | `hpoList`, `depth` | `pubCaseFinder` | PubCaseFinder API を呼び上位 5 件取得 |
| `NormalizePCFNode` | `pubCaseFinder` | `pubCaseFinder` | OMIM ID に基づき疾患名を正式名へ正規化 |
| `GestaltMatcherNode` | `imagePath`, `depth` | `GestaltMatcher` | 画像があれば GestaltMatcher API を呼び候補疾患を取得 |
| `NormalizeGestaltMatcherNode` | `GestaltMatcher` | `GestaltMatcher` | OMIM ID に基づき `syndrome_name` を正規化 |
| `createHPODictNode` | `hpoList` | `hpoDict` | `phenotype_mapping.json` で HPO ID をラベル化 |
| `createAbsentHPODictNode` | `absentHpoList` | `absentHpoDict` | 明示的に観察されなかった HPO ID をラベル化 |
| `createZeroShotNode` | `hpoDict`, `absentHpoDict`, `use_absentHPO`, `onset`, `sex`, `llm` | `zeroShotResult`, `prompt` | LLM 構造化出力で候補疾患を生成。`use_absentHPO=True` の場合のみ absent HPO を使用 |
| `NormalizeZeroShotNode` | `zeroShotResult` | `zeroShotResult` | 疾患名を embedding 正規化し、類似度 0.70 未満と重複 OMIM を除外 |
| `HPOwebSearchNode` | `hpoDict`, `llm` | `webresources` | HPO から DDGS 検索クエリを生成し、検索結果スニペットを要約 |
| `DiseaseSearchWithHPONode` | `hpoDict`, `depth` | `phenotypeSearchResult` | HPO ラベルを embedding 化し、FAISS で類似 OMIM 疾患を検索 |
| `mergeCandidateResultsNode` | `pubCaseFinder`, `zeroShotResult`, `GestaltMatcher`, `phenotypeSearchResult` | `mergedDiseaseCandidates` | 診断前に各ツールの順位付き疾患候補を疾患単位で統合し、どのツールで何位だったかを保持 |
| `createDiagnosisNode` | `mergedDiseaseCandidates`, HPO, Web, LLM | `tentativeDiagnosis`, `prompt` | 統合済み候補表と患者情報・Web 結果を使って暫定診断を生成。`use_absentHPO=True` の場合のみ absent HPO を使用 |
| `diseaseNormalizeNode` | `tentativeDiagnosis` | `tentativeDiagnosis` | 暫定診断疾患名を embedding 正規化し、類似度 0.75 未満を除外 |
| `diseaseSearchNode` | `tentativeDiagnosis`, `depth`, `llm`, `memory` | `memory` | 各暫定疾患について Wikipedia/PubMed を並列検索し要約 |
| `reflectionNode` | `tentativeDiagnosis`, `hpoDict`, `absentHpoDict`, `use_absentHPO`, `memory`, `llm` | `reflection`, `prompt` | 各暫定疾患を LLM で妥当性評価。最大 10 スレッドで並列実行。`use_absentHPO=True` の場合のみ absent HPO を使用 |
| `finalDiagnosisNode` | `tentativeDiagnosis`, `reflection`, HPO, `llm` | `finalDiagnosis`, `prompt` | reflection までの情報を統合して最終診断を生成。`use_absentHPO=True` の場合のみ absent HPO を使用 |
| `diseaseNormalizeForFinalNode` | `finalDiagnosis` | `finalDiagnosis` | 最終診断疾患名を embedding 正規化し、類似度 0.75 未満を除外 |

### 4.2 エッジ

現行グラフの主な流れ:

1. `START`
2. `BeginningOfFlowNode`
3. 並列的に以下を開始
   - `PCFnode` -> `NormalizePCFNode`
   - `GestaltMatcherNode` -> `NormalizeGestaltMatcherNode`
   - `createHPODictNode`
   - `createAbsentHPODictNode`
4. `createHPODictNode` と `createAbsentHPODictNode` の完了後、`createZeroShotNode` -> `NormalizeZeroShotNode`
5. `createHPODictNode` の完了後、`HPOwebSearchNode` と `DiseaseSearchWithHPONode`
6. `NormalizeZeroShotNode`, `NormalizePCFNode`, `NormalizeGestaltMatcherNode`, `DiseaseSearchWithHPONode` の完了後、`mergeCandidateResultsNode`
7. `mergeCandidateResultsNode` と `HPOwebSearchNode` の完了後、`createDiagnosisNode`
8. `diseaseNormalizeNode`
9. `diseaseSearchNode`
10. `reflectionNode`
11. 条件分岐
    - `ProceedToFinalDiagnosisNode` -> `finalDiagnosisNode`
    - `ReturnToBeginningNode` -> `BeginningOfFlowNode`
12. `finalDiagnosisNode`
13. `diseaseNormalizeForFinalNode`
14. `END`

### 4.4 HPO 重要度フィルタ

対象ファイル: `agent/utils/hpo_importance_filter.py`

`filter_impotance=True` の場合、`RareDiseaseDiagnosisPipeline.run()` の初期 State 作成前に `hpo_list` と `absent_hpo_list` をそれぞれ重要度順に絞る。

重要度は `HPO_importance/HPO_importance.json` の `related_disease_num` を用いる。関連疾患数が少ない HPO ほど、その表現型はより特異的で重要とみなす。

仕様:

- 上限件数は `TOP_HPO_IMPORTANCE_LIMIT = 15` として `agent/utils/hpo_importance_filter.py` の先頭付近に定義する。
- present HPO と absent HPO は独立に処理する。
- 各リストの件数が 15 件以上の場合、`related_disease_num` が小さい順に上位 15 件へ絞る。
- `HPO_importance.json` に存在しない HPO ID は重要度不明として末尾側に並べる。
- 同じ関連疾患数の場合は入力順を維持する。
- `filter_impotance=False` の場合は従来どおり入力リストをそのまま使用する。

### 4.3 reflection 後の条件分岐

`after_reflection_edge()` は次の条件で遷移する。

| 条件 | 遷移 |
|---|---|
| `depth > 0` | `finalDiagnosisNode` |
| `reflection` が空、または `reflection.ans` が空 | `BeginningOfFlowNode` |
| `reflection.ans[*].Correctness` に `True` が 1 件以上ある | `finalDiagnosisNode` |
| それ以外 | `BeginningOfFlowNode` |

注意: 初期 `depth=0` は `BeginningOfFlowNode` で `1` になるため、現行実装では初回 `reflectionNode` 後に `depth > 0` が成立し、原則として再ループせず最終診断へ進む。

## 5. ツール別仕様

### 5.1 PubCaseFinder

対象ファイル: `agent/tools/pcf_api.py`

入力:

- `hpo_list: List[str]`
- `depth: int`
- `max_retries: int = 3`

処理:

- `hpo_list` をカンマ区切りにする。
- `https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list` に GET リクエストする。
- `target=omim`, `format=json`, `hpo_id=<HPO IDs>` を付与する。
- レスポンス上位 5 件を抽出する。
- 失敗時は指数バックオフで最大 3 回リトライする。

出力:

```python
[
    {
        "omim_disease_name_en": str,
        "description": str,
        "score": Optional[float],
        "omim_id": str,
    }
]
```

### 5.2 GestaltMatcher

対象ファイル: `agent/tools/gestaltMathcher.py`

入力:

- `image_path: str`
- `depth: int`
- `max_retries: int = 3`

環境変数:

- `GESTALT_API_USER`
- `GESTALT_API_PASS`

処理:

- 画像を base64 エンコードする。
- `https://dev-pubcasefinder.dbcls.jp/gm_endpoint/predict` に Basic 認証つき POST を送る。
- `suggested_syndromes_list` から上位 `depth + 4` 件を取得する。
- `distance` または `gestalt_score` を `score = (1.3 - distance) / 1.3` に変換する。
- 失敗時は指数バックオフで最大 3 回リトライする。

出力:

```python
[
    {
        "subject_id": str,
        "syndrome_name": str,
        "omim_id": str,
        "image_id": str,
        "score": float,
    }
]
```

### 5.3 HPO 辞書作成

対象ファイル: `agent/tools/make_HPOdic.py`

入力:

- `hpo_list: List[str]`
- `mapping_path: str`

処理:

- `mapping_path` 引数は現行実装では使用しない。
- `agent/data/phenotype_mapping.json` を読み込む。
- HPO ID ごとにラベルを取得する。

出力:

```python
{ "HP:0001263": "Global developmental delay" }
```

未登録 HPO ID は空文字 `""` になる。

### 5.4 Zero-Shot 診断

対象ファイル: `agent/tools/ZeroShot.py`

入力:

- `hpoDict`
- `absentHpoDict`
- `use_absentHPO`
- `onset`
- `sex`
- `llm`

処理:

- present HPO ラベル、発症時期、性別を `zero-shot-diagnosis-prompt` に埋め込む。
- `use_absentHPO=True` の場合のみ、明示的に観察されなかった HPO ラベルも absent HPO として埋め込む。
- `AzureOpenAIWrapper.get_structured_llm(ZeroShotOutput)` により構造化出力を要求する。

出力:

- `ZeroShotOutput`
- 実行プロンプト文字列

### 5.5 疾患名正規化

対象ファイル: `agent/tools/diseaseNormalize.py`

入力:

- 疾患名文字列、または `State` 内の `pubCaseFinder`, `GestaltMatcher`, `zeroShotResult`, `DiagnosisOutput`

環境変数:

- `AZURE_DBCLS_JAPANEAST`

ローカルデータ:

- `agent/data/DataForOmimMapping/DataForOmimMapping.bin`
- `agent/data/DataForOmimMapping/DataForOmimMapping.json`
- `agent/data/DataForOmimMapping/omim_mapping.json`

処理:

- Azure OpenAI `text-embedding-3-large` で疾患名を embedding 化する。
- FAISS インデックスで最近傍 OMIM ラベルを検索する。
- `omim_mapping.json` にある正式病名へ置換する。
- `zeroShotResult` は類似度 0.70 以上のみ採用し、OMIM ID 重複を除去する。
- `DiagnosisOutput` は類似度 0.75 以上のみ採用する。

出力:

- 正規化済み候補リスト、または正規化済み `ZeroShotOutput` / `DiagnosisOutput`

### 5.6 HPO 表現型 embedding 検索

対象ファイル: `agent/tools/embeddingSearchWithHPO.py`

入力:

- `State.hpoDict`
- `State.depth`

環境変数:

- `AZURE_DBCLS_JAPANEAST`

ローカルデータ:

- `agent/data/DataForDiseaseSearchFromHPO/phenotype_index.bin`
- `agent/data/DataForDiseaseSearchFromHPO/phenotype_index.json`

処理:

- HPO ラベルをカンマ区切りにして検索文を作る。
- Azure OpenAI `text-embedding-3-large` で embedding 化する。
- ベクトルを L2 正規化する。
- FAISS で `k = 5 * depth` 件検索する。
- 検索結果を `PhenotypeSearchFormat` に変換する。

出力:

```python
List[PhenotypeSearchFormat]
```

### 5.7 HPO Web 検索

対象ファイル: `agent/tools/HPOwebReserch.py`

入力:

- `State.hpoDict`
- `State.llm`
- 既存 `State.webresources`

処理:

- HPO ラベルを LLM に渡して DDGS 用検索クエリを 2 件生成する。
- 各クエリで DDGS テキスト検索を最大 2 件実行する。
- 検索結果スニペットを LLM で鑑別診断向けに要約する。
- 医学関連でない要約は除外する。
- URL 重複を除外する。

出力:

```python
[
    {
        "title": str,
        "url": str,
        "snippet": str,
    }
]
```

注意: `createDiagnosis()` は Web 検索結果の本文として `content` があればそれを使用し、なければ `snippet` を使用する。

### 5.8 疾患候補マージ

対象ファイル: `agent/tools/rankingMerge.py`

入力:

- `pubCaseFinder`
- `zeroShotResult`
- `GestaltMatcher`
- `phenotypeSearchResult`

処理:

- PubCaseFinder、ZeroShot、GestaltMatcher、PhenotypeSearch の順位付き疾患候補を読み取る。
- OMIM ID がある場合は `OMIM:<数字>` に正規化し、同一 OMIM ID の候補を同一疾患として統合する。
- OMIM ID がない場合は疾患名を大文字化・空白正規化したキーで統合する。
- 各疾患候補に、どのツールで何位だったか、score、matched HPO、補足情報を `tool_rankings` として保持する。
- `consensus_count` は候補を支持したツール数、`best_rank` は各ツール順位の最小値として算出する。
- 出力は `consensus_count` 降順、`best_rank` 昇順、疾患名昇順で並べる。

出力:

```python
[
    {
        "disease_name": str,
        "OMIM_id": Optional[str],
        "consensus_count": int,
        "best_rank": int,
        "tool_rankings": [
            {
                "tool": str,
                "rank": int,
                "score": Optional[float],
                "matched_hpo_id": str,
                "note": str,
            }
        ],
    }
]
```

### 5.9 暫定診断生成

対象ファイル: `agent/tools/diagnosis.py`

入力:

- `hpoDict`
- `absentHpoDict`
- `use_absentHPO`
- `onset`
- `sex`
- `mergedDiseaseCandidates`
- `webresources`
- `llm`

処理:

- `mergedDiseaseCandidates` を、疾患名、OMIM ID、支持ツール数、最良順位、各ツールの順位・score・matched HPO を含むテキスト表へ整形する。
- `use_absentHPO=True` の場合のみ、明示的に観察されなかった HPO ラベルを診断プロンプトへ含める。
- GestaltMatcher 結果の有無により `diagnosis_prompt` または `diagnosis_prompt_no_gestalt` を選択する。
- プロンプトでは、個別ツールの生リストではなく統合済み候補表を authoritative candidate list として扱う。
- LLM に通常テキスト出力を要求する。
- `===CASE_START===` / `===CASE_END===` と `KEY::VALUE` 形式を正規表現でパースする。
- references セクションを `DiagnosisOutput.reference` に格納する。

出力:

- `DiagnosisOutput`
- 実行プロンプト文字列

### 5.10 疾患知識検索

対象ファイル: `agent/tools/diseaseSearch.py`

入力:

- `tentativeDiagnosis`
- `depth`
- `llm`
- `memory`

処理:

- 暫定診断の各疾患名を抽出する。
- 疾患ごとに Wikipedia と PubMed を並列検索する。
- Wikipedia は `top_k_results = depth * 1`, `doc_content_chars_max = 2000`。
- PubMed は `top_k_results = depth * 3`, `doc_content_chars_max = 3000`。
- 検索本文を LLM で鑑別診断向けに要約する。
- URL 重複を除外して `memory` に追加する。
- PubMed の 429 エラーは最大 3 回リトライする。

出力:

```python
{
    "memory": [
        {
            "title": str,
            "url": str,
            "content": str,
            "disease_name": str,
        }
    ]
}
```

### 5.11 Reflection

対象ファイル: `agent/tools/reflection.py`

入力:

- `State`
- 評価対象の `DiagnosisFormat`

処理:

- `memory` から評価対象疾患名と一致する知識のみ抽出する。
- 患者 present HPO、発症時期、性別、暫定診断説明、疾患知識を `reflection_prompt` に埋め込む。
- `use_absentHPO=True` の場合のみ、明示的に観察されなかった HPO ラベルも absent HPO として埋め込む。
- `ReflectionFormat` の構造化出力として LLM を呼ぶ。
- `max_completion_tokens` を 25000, 35000, 50000 と増やしながら再試行する。
- 長さ制限または例外時は `Correctness=False` の fallback 結果を返す。

出力:

- `ReflectionFormat`
- 実行プロンプト文字列

### 5.12 最終診断

対象ファイル: `agent/tools/finalDiagnosis.py`

入力:

- `hpoDict`
- `absentHpoDict`
- `use_absentHPO`
- `clinicalText`
- `tentativeDiagnosis`
- `reflection`
- `onset`
- `sex`
- `llm`

処理:

- 暫定診断と reflection をテキストに整形する。
- `use_absentHPO=True` の場合のみ、明示的に観察されなかった HPO ラベルを最終診断プロンプトへ含める。
- `final_diagnosis_prompt` に埋め込む。
- `DiagnosisOutput` の構造化出力として LLM を呼ぶ。

出力:

- `DiagnosisOutput`
- 実行プロンプト文字列

## 6. LLM 仕様

対象ファイル: `agent/llm/azure_llm_instance.py`, `agent/llm/llm_wrapper.py`, `agent/llm/prompt.py`

### 6.1 モデル設定

サポートされる `model_name`:

| `model_name` | 環境変数 prefix |
|---|---|
| `gpt-4o` | `AZURE_OPENAI_4o` |
| `gpt-5-1` | `AZURE_OPENAI_5-1` |
| `gpt-5-2` | `AZURE_OPENAI_5-2` |

必要な環境変数:

- `{PREFIX}_ENDPOINT`
- `{PREFIX}_API_KEY`
- `{PREFIX}_DEPLOYMENT_NAME`
- `{PREFIX}_API_VERSION`

### 6.2 AzureOpenAIWrapper

| メソッド | 入力 | 出力 | 内容 |
|---|---|---|---|
| `_create_llm(max_completion_tokens)` | token 上限 | `AzureChatOpenAI` | Azure Chat LLM を生成 |
| `get_temp_llm_with_max_tokens(max_completion_tokens)` | token 上限 | `AzureChatOpenAI` | 一時 LLM を生成 |
| `get_structured_llm(output_schema)` | Pydantic schema | structured LLM | 構造化出力用 LLM |
| `generate(prompt)` | `str` | LLM 応答 | 通常テキスト生成 |

`gpt-4o` は `temperature=0.0` と `max_tokens` を設定する。`gpt-5-1`, `gpt-5-2` は `model_kwargs.extra_body` に `max_completion_tokens`, `verbosity`, `reasoning_effort` を渡す。

### 6.3 プロンプト

| キー | 用途 |
|---|---|
| `diagnosis_prompt_no_gestalt` | 顔画像なしの統合暫定診断 |
| `diagnosis_prompt` | 顔画像ありの統合暫定診断 |
| `zero-shot-diagnosis-prompt` | HPO のみからの zero-shot 診断 |
| `reflection_prompt` | 暫定診断の医学的妥当性評価 |
| `final_diagnosis_prompt` | 最終診断生成 |

## 7. データファイル仕様

| ファイル | 内容 | 主な利用箇所 |
|---|---|---|
| `agent/data/phenotype_mapping.json` | HPO ID から HPO ラベルへの辞書。約 19,726 件 | `make_HPOdic.py` |
| `agent/data/DataForOmimMapping/DataForOmimMapping.bin` | 疾患名正規化用 FAISS インデックス。約 328 MB | `diseaseNormalize.py` |
| `agent/data/DataForOmimMapping/DataForOmimMapping.json` | 正規化インデックスの `labels`, `omim_ids` | `diseaseNormalize.py` |
| `agent/data/DataForOmimMapping/omim_mapping.json` | OMIM ID から正式疾患名への辞書。約 27,957 件 | `diseaseNormalize.py` |
| `agent/data/DataForDiseaseSearchFromHPO/phenotype_index.bin` | HPO 表現型類似検索用 FAISS インデックス。約 98 MB | `embeddingSearchWithHPO.py` |
| `agent/data/DataForDiseaseSearchFromHPO/phenotype_index.json` | OMIM 疾患情報と表現型リスト | `embeddingSearchWithHPO.py` |
| `agent/data/DataForDiseaseSearchFromHPO/omim_database.json` | OMIM 疾患情報データ | 現行 agent コードからの直接参照はなし |
| `HPO_importance/HPO_importance.json` | HPO ID、ラベル、関連疾患数の配列。関連疾患数が少ないほど重要 | `hpo_importance_filter.py` |

## 8. ログ・保存仕様

### 8.1 実行ログ

対象ファイル: `agent/utils/logger.py`

`enable_log=True` の場合、各ノードの結果を `log/` 配下に追記する。結果が Pydantic モデルの場合は JSON 形式で整形する。ノード結果に `prompt` が含まれる場合は、プロンプトも記録する。

### 8.2 結果 JSON 保存

対象ファイル: `agent/utils/result_saver.py`

`@save_result(node_name)` が付与されたノードは、戻り値を `res/{patient_id}.json` に保存する。

処理仕様:

- ファイルがなければ `{}` で作成する。
- `fcntl.flock` により排他ロックする。
- 既存 JSON を読み込む。
- ノード結果を `update()` でマージする。
- Pydantic オブジェクトは再帰的に dict 化する。
- 保存に失敗してもパイプライン実行は継続する。

### 8.3 プロファイル

対象ファイル: `agent/utils/profiler.py`

`@profile_node` が付与されたノードは実行時間を計測し、標準出力に `[Profile] <node>: <秒>` を表示する。`NodeProfiler.get_summary()` により集計テキストを取得できる。

## 9. 外部依存・環境変数

### 9.1 外部サービス

| サービス | 用途 |
|---|---|
| Azure OpenAI Chat | zero-shot、暫定診断、reflection、最終診断、要約 |
| Azure OpenAI Embeddings | 疾患名正規化、HPO 表現型検索 |
| PubCaseFinder API | HPO ベース候補疾患検索 |
| GestaltMatcher API | 顔画像ベース候補疾患検索 |
| DDGS | HPO Web 検索 |
| WikipediaRetriever | 疾患知識検索 |
| PubMedRetriever | 疾患知識検索 |

### 9.2 必須または条件付き環境変数

| 環境変数 | 必要条件 | 用途 |
|---|---|---|
| `AZURE_OPENAI_4o_ENDPOINT` | `model_name="gpt-4o"` | Chat LLM |
| `AZURE_OPENAI_4o_API_KEY` | `model_name="gpt-4o"` | Chat LLM |
| `AZURE_OPENAI_4o_DEPLOYMENT_NAME` | `model_name="gpt-4o"` | Chat LLM |
| `AZURE_OPENAI_4o_API_VERSION` | `model_name="gpt-4o"` | Chat LLM |
| `AZURE_OPENAI_5-1_*` | `model_name="gpt-5-1"` | Chat LLM |
| `AZURE_OPENAI_5-2_*` | `model_name="gpt-5-2"` | Chat LLM |
| `AZURE_DBCLS_JAPANEAST` | 正規化・embedding 検索使用時 | Azure OpenAI Embedding |
| `GESTALT_API_USER` | 画像診断使用時 | GestaltMatcher Basic 認証 |
| `GESTALT_API_PASS` | 画像診断使用時 | GestaltMatcher Basic 認証 |

## 10. 例外・スキップ仕様

| 箇所 | 条件 | 挙動 |
|---|---|---|
| `PCFnode` | `hpoList` が空 | `pubCaseFinder=[]` |
| PubCaseFinder API | リクエスト失敗 | 最大 3 回リトライ後、空リスト |
| `GestaltMatcherNode` | `imagePath` なし | `GestaltMatcher=[]` |
| GestaltMatcher API | 認証情報なし | 例外をノードで捕捉し `GestaltMatcher=[]` |
| `createZeroShotNode` | `hpoDict` なし、または LLM なし | `zeroShotResult=None` |
| `embedding_search_with_hpo` | index、mapping、client の初期化失敗 | `None` |
| `createDiagnosis` | LLM なし、またはパース結果なし | `None, None` |
| `diseaseSearchForDiagnosis` | LLM または暫定診断なし | 既存 `memory` を返す |
| `reflection` | LLM 長さ制限・例外 | `Correctness=False` の fallback |
| `save_result` | 保存失敗 | エラー表示のみで実行継続 |

## 11. 現行実装上の注意点

- `agent/llm/azure_llm_instance.py` はモジュール末尾で `azure_llm = get_llm_instance("gpt-4o")` を実行するため、import 時点で `gpt-4o` 用環境変数が不足していると失敗する。
- `agent/tools/diseaseNormalize.py` は import 時点で `AZURE_DBCLS_JAPANEAST` を確認し、FAISS インデックスも読み込む。環境変数または `.bin` ファイルが不足していると import に失敗する。
- `HPOwebSearchNode` の出力キーは `snippet` だが、`createDiagnosis()` は Web 検索結果の本文として `content` を参照している。
- `State` では `webresources` が必須扱いだが、初期 state には明示的に含まれていない。各処理は `state.get("webresources", [])` で補完している。
- `after_reflection_edge()` は `depth > 0` で最終診断へ進むため、現行初期値では reflection 後の再探索ループは実質的に動作しない。
- `agent/tools/MCP/MCP_client.py` は `mcp_endpoints = {"pcf": "hogehoge"}` の仮 URL で MCPClient を作る実験的コードであり、現行グラフからは利用されていない。
