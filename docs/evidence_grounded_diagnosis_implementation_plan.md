# 文献根拠付き・日英併記診断フロー実装計画

> **現行実装の簡素化方針**: ノードの流れは `tool -> tentative -> literature
> search/Evidence -> reflection -> final` を維持する。`patient_matches`、
> `limitations`、`impact`、`relationship`、`reasoning`、`reference_catalog` などの
> 二重管理は行わず、診断文とEvidenceの原文抜粋、番号付きReferenceだけを正式データとする。
> 候補統合後は上位5件へ切り詰めず、統合された全候補を文献検索・Evidence抽出・Reflection・tentativeDiagnosisへ渡す。公開する最終鑑別のみ最大5件とする。

## 1. 前提

実装対象は、[文献根拠付き・日英併記診断フロー仕様 v2](evidence_grounded_diagnosis_specification.md)で定義したモデルとフローである。

現行コードはv1の`memory`、自由文Reference、暫定診断後の文献検索を使用している。既存APIを急に壊さないため、段階移行中はv1フィールドを読み取り可能に残し、新しい診断ロジックだけをv2フィールドへ切り替える。

## 2. 最終的な実装フロー

```text
HPO・画像・発症時期・性別
        |
        +--> PCF / GestaltMatcher / ZeroShot / PhenoBrain / PhenotypeSearch
        +--> HPO Web検索（補助）
        |
        v
候補統合・Candidate ID付与
        |
        v
候補メタデータ補完
        |
        v
CandidateごとのPubMed/PMC検索
        |
        v
Evidence抽出・原文抜粋検証
        |
        v
日英併記の暫定診断
        |
        v
日英併記のReflection
        |
        v
日英併記の最終診断
        |
        v
出力検証・公開レスポンス整形
```

## 3. 実装フェーズ

### Phase 0: ベースライン固定と移行ガード

目的は、v2実装前の既存動作を再現可能にすること。

作業:

- 現行テストを実行し、ベースライン結果を保存する。
- `docs/agent_specification.md`はv1資料として保持し、v2仕様へのリンクを維持する。
- `evidence_mode`または同等のfeature flagを導入する。
- `legacy`モードでは現行ノードの診断ロジックと旧フィールドを維持する。ただし公開`OMIM_id`はlegacy/v2を問わず`OMIM:数字`へ統一し、数値のみの互換値が必要な場合は別名フィールドで提供する。
- `v2`モードでは新しい`sources/evidence`経路を使用する。
- `RareDiseaseDiagnosisPipeline.run(..., evidence_mode="v2")`を公開デフォルトとし、旧ルートは`evidence_mode="legacy"`を明示した場合だけ選択する。
- 初期Stateにも同じ`evidence_mode`を保存し、ノード内で環境変数や暗黙の判定を使わない。

完了条件:

- 既存テストが現状のまま通る。
- v2を無効にした場合、OMIMのcanonical prefix化を除き、既存ノードのロジックと出力構造に意図しない差分が発生しない。

対象ファイル候補:

- `agent/agent_pipeline.py`
- `tests/`

### Phase 1: データモデルと正規化

目的は、以後のノードが共有する型を先に固定すること。

作業:

- `CandidateSupport`を追加する。
- `MergedDiseaseCandidateV2`を追加し、現行TypedDictの`MergedDiseaseCandidate`をv1 adapterとして残す。v2境界でcandidate modelへ変換し、既存の`.get()`利用をPydanticモデルへ漏らさない。
- `candidate_id`を追加する。
- `tool_rankings`を`tool_supports`へ移行するadapterを追加する。
- GestaltMatcherのAPI rankを優先して保持し、rank欠損時だけレスポンス順をfallbackとして付与する。fallbackには`rank_source="agent_enumeration"`を設定する。
- `SourceRecord`、`ReferenceRecord`、`EvidenceItem`を追加する。
- 現行の`DiagnosisFormat`、`ReflectionFormat`、`DiagnosisOutput`を直接変更せず、`DiagnosisFormatV2`、`ReflectionFormatV2`、`DiagnosisOutputV2`、`ReflectionOutputV2`を追加する。
- v2モデルに`description_en`、`description_ja`、`evidence_ids`、`literature_status`、Evidence評価を持たせる。
- `DiagnosisOutputV2.references`、`DiagnosisOutputV2.evidence`、`validation_status`を追加する。
- `State`に`sources`、`evidence`、`flow_iteration`、`max_flow_iterations`、validation情報を追加する。
- State初期値に`evidence_mode="v2"`、`sources=[]`、`evidence=[]`を追加する。legacyは明示指定時だけ使用する。
- OMIM正規化関数を共通化し、すべての入力境界で`OMIM:数字`にする。
- Zero-shotは理由付き一時モデルをログ専用で使用し、理由を除いた候補だけをStateへ渡す。正規化前後と、match/normalized/mismatch/unconfirmed/duplicate/excludedの判定を評価用に保存する。
- 暫定診断は全統合候補を候補IDごとに一度ずつ含め、候補ID・OMIM・病名・順位を検証する。不整合や候補欠落時は1回だけ再生成し、再生成後も不正なら全候補を含む安全なフォールバックと検証メタデータを残す。

重要な実装方針:

- `CandidateSupport`はv2の`mergedDiseaseCandidates[*].tool_supports`だけに保存する。legacyでは`tool_rankings`を読み取り専用のadapter入力として扱う。
- `source_ids`はLLMに生成させず、`evidence_ids`から決定論的に導出する。
- `reference`、`description`、`Correctness`などの旧フィールドは移行adapterでのみ扱う。

テスト:

- OMIM IDの入力形式別テスト
- Candidate ID生成テスト
- `tool_rankings`から`tool_supports`への変換テスト
- Source/EvidenceのID整合性テスト
- 日英フィールドの必須性テスト

完了条件:

- モデルのimportとPydantic validationが通る。
- v1形式をv2形式へ変換できる。
- 既存テストを壊さない。

### Phase 2: Candidateメタデータ補完

目的は、リッチな文献検索に必要な疾患情報をCandidateへ付与すること。

作業:

- `candidateMetadataNode`を追加する。
- OMIM IDから正式疾患名を取得する。
- 利用可能なデータからgene symbols、synonyms、phenotype labelsを補完する。
- 情報が見つからない場合は空配列とし、検索不能として落とさない。
- 補完元を内部メタデータとして記録する。

候補メタデータの優先順位:

1. Candidateにすでに付与されたOMIM ID
2. OMIM mappingまたは疾患データベース
3. Phenotype indexのsynonym・phenotype情報
4. 正規化済み疾患名

テスト:

- OMIMあり候補のメタデータ補完
- OMIMなし候補のCandidate ID生成
- metadata欠損時のfallback
- 疾患名・gene・synonymの重複除去

完了条件:

- 文献検索ノードがCandidateから検索語を作れる。
- メタデータがなくても候補が消えない。

### Phase 3: PubMed/PMC検索アダプター

目的は、疾患名だけでなくOMIM、遺伝子、別名、HPOを使って文献を取得すること。

作業:

- `agent/tools/literature_search.py`を追加する。
- Candidateごとに複数の検索クエリを生成する。
- PubMed検索結果からPMID、title、authors、year、journal、abstract、URLを取得する。
- PMCまたは取得可能なfull textがある場合は、source metadataを区別して保存する。
- クエリごとの重複PMIDを排除する。
- Candidate IDと使用クエリをSourceRecordへ保存する。
- 候補統合後の全Candidateを検索対象とし、検索前に公開鑑別の件数へ切り詰めない。
- 429、timeout、空結果、metadata欠損を個別に扱う。
- 一般Web検索は補助経路として残すが、PubMed/PMCと混ぜない。

クエリ生成:

- 疾患名
- 疾患名 + phenotype/clinical features/syndrome
- 疾患名 + gene symbol
- OMIM IDまたは数字
- synonym + distinctive phenotype
- gene symbol + 上位特徴的HPO 2件

検索結果の採用ルール:

- 疾患identityが一致するものを優先する。
- 別疾患、別遺伝子、別サブタイプは候補Evidenceに紐付けない。
- 原文abstractまたは取得本文を保存し、LLM要約をraw_contentにしない。

テスト:

- クエリ生成のfixtureテスト
- PubMed APIのmockテスト
- PMCID検出とPMC全文XML取得のmockテスト
- PMC取得失敗時のPubMed抄録フォールバックテスト
- PMID重複除去テスト
- source_type/content_type/URL保存テスト
- 429とtimeoutのリトライテスト
- entity mismatchの除外テスト

完了条件:

- 候補統合後に文献検索が実行される。
- Candidateごとに検索結果とクエリが追跡できる。
- PMCIDがある文献は取得可能な場合に`content_type="full_text"`で追跡でき、取得不能時も抄録経路が維持される。
- `sources`に原文と識別子が保存される。

### Phase 4: Evidence抽出

目的は、文献から診断に必要な情報だけを構造化すること。

作業:

- `evidenceExtractionNode`を追加する。
- SourceRecordとCandidate identityをLLMへ渡す。
- `EvidenceItem`をstructured outputで生成する。
- disease identity、gene/inheritance、core phenotype、distinctive phenotype、negative finding、mechanism、phenotype overlap/mismatchを抽出する。
- `claim_en`と`claim_ja`を同時生成する。
- `excerpt`がraw_content内に存在するか、正規化文字列で検証する。
- 原文で確認できないEvidenceは破棄または`validation_errors`に記録する。
- 同じSourceRecordから同一Evidenceを重複作成しない。

テスト:

- structured outputのfixtureテスト
- excerptの原文包含テスト
- 文章に存在しない主張の拒否テスト
- EvidenceのCandidate/Source紐付けテスト
- 英日claimのEvidence ID整合テスト

完了条件:

- `evidence`がCandidateとSourceに紐付いている。
- 最終診断が自由文の文献要約ではなくEvidenceを参照できる。

### Phase 5: LangGraphのフロー変更

目的は、文献検索とEvidence抽出を暫定診断より前に置くこと。

作業:

- `literatureSearchNode`を追加する。
- `evidenceExtractionNode`を追加する。
- `diagnosisOutputValidationNode`を追加する。
- `mergeCandidateResultsNode`の後に`candidateMetadataNode`を接続する。
- `candidateMetadataNode`の後に`literatureSearchNode`を接続する。
- `literatureSearchNode`の後に`evidenceExtractionNode`を接続する。
- `evidenceExtractionNode`の後に`createDiagnosisNode`を接続する。
- 旧`diseaseSearchNode`を新v2グラフから外す。
- `HPOwebSearchNode`は補助検索として候補生成側に残す。
- `depth`を検索深度と反復制御に兼用しない。
- `flow_iteration`と`max_flow_iterations`でReflection後の再検索回数を制御する。
- Reflection後のconditional edgeを追加する。Evidence不足かつ`flow_iteration < max_flow_iterations`のときだけ`literatureSearchNode`へ戻し、それ以外は`finalDiagnosisNode`へ進む。
- 再検索時はCandidate metadataを再取得せず、SourceのPMID/DOI/URLで重複排除する。
- 再検索後は`evidenceExtractionNode`を実行し、既存EvidenceをIDで上書きせず追加・統合する。

推奨されるv2エッジ:

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

テスト:

- グラフのnode/edge構造テスト
- 文献検索が暫定診断より前に実行されるテスト
- evidenceがcreateDiagnosisへ渡るテスト
- Reflection後の再検索上限テスト
- `literatureSearchNode`失敗時のpartial statusテスト
- 統合候補全件が文献検索・Evidence抽出・Reflectionへ渡るテスト

完了条件:

- v2グラフに暫定診断後の旧文献検索経路が残っていない。
- Evidenceなしで最終診断へ進んだ場合、その状態が明示される。
- 文献検索とReflectionは全統合候補を扱い、公開最終鑑別だけが最大5件に制限される。
- tentativeDiagnosisは全統合候補を候補IDごとに一度ずつ含め、公開最終鑑別はその中から最大5件を選ぶ。

実装上の注意:

- v2では`reflectionNode`から`BeginningOfFlowNode`へ戻さない。再検索が必要な場合は`literatureSearchNode`へ戻り、同じ候補集合を使ってEvidence、暫定診断、Reflectionを順に再生成する。
- legacyの`diseaseSearchNode`、`attach_diagnosis_references`、`memory`経路はlegacyモード専用とし、v2の最終診断入力へ混入させない。
- `HPOwebSearchNode`とPhenotypeSearchの結果は候補探索・補助情報に限定し、`SourceRecord`、`EvidenceItem`、`ReferenceRecord`へ変換しない。

新規ノードのState契約:

```text
candidateMetadataNode
  input: mergedDiseaseCandidates
  output: mergedDiseaseCandidates with candidate metadata

literatureSearchNode
  input: mergedDiseaseCandidates, hpoDict, sources, flow_iteration
  output: sources

evidenceExtractionNode
  input: mergedDiseaseCandidates, sources, hpoDict, absentHpoDict
  output: evidence

diagnosisOutputValidationNode
  input: finalDiagnosis, mergedDiseaseCandidates, sources, evidence
  output: finalDiagnosis with validation_status and validation_errors
```

v2ノードは`evidence_mode="v2"`の場合だけ実行する。legacyモードでは既存ノードと既存State更新を使用する。

### Phase 6: 日英併記の診断生成

目的は、暫定診断、Reflection、最終診断を新しいEvidenceモデルに合わせること。

作業:

- `createDiagnosisNode`をstructured `DiagnosisOutputV2`生成へ移行する。
- 旧`===CASE_START===`自由文パーサーはlegacy fallbackとして残す。
- 暫定診断PromptへCandidateSupport、患者HPO、Evidenceを渡す。
- `description_en`、`description_ja`、`evidence_ids`、`literature_status`を必須出力にする。
- Reflectionを`EvidenceAssessment`ベースに変更する。
- 既存のtool-supported reflection overrideを削除する。
- `finalDiagnosisNode`へraw `memory`ではなく、候補別EvidenceとReflection評価を渡す。
- 日本語は英語の意味、Evidence ID、確信度を変更しない。
- 病態機序はEvidenceにある場合だけ出力する。

テスト:

- 日英フィールド出力テスト
- 日英Evidence ID一致テスト
- literature_status別の診断文章テスト
- Evidenceにない病態機序を出力しないテスト
- PhenotypeSearchが文献Referenceへ入らないテスト
- legacy aliasの整合性テスト
- Zero-shot selection_reasonがState・Evidence・公開レスポンスへ混入しないテスト
- Zero-shotの正規化前後とidentity statusの保存テスト
- 暫定診断の候補外ID・重複・OMIM不一致・順位不整合に対する1回再生成テスト

完了条件:

- 暫定診断と最終診断が同じEvidence契約を使う。
- 英語と日本語が常に同時に返る。

### Phase 7: Reference・OMIM・公開レスポンス

目的は、ユーザーが出典と診断根拠を追跡できる公開形式にすること。

作業:

- `DiagnosisOutputV2.references`をSourceRecordから生成する。
- Evidence IDからReferenceRecordを決定論的に作る。
- URL、source_type、identifier、title、content_type、raw_contentを返す。
- `citation_allowed=False`のSourceをReferenceから除外する。
- `attach_diagnosis_references`はlegacyモードだけで使用し、v2ではEvidence→Sourceから決定論的にReferenceを生成する。
- OMIM IDを公開レスポンスの`OMIM_id`自体へ`OMIM:数字`で返す。
- `_curie`追加方式からcanonical field方式へ移行する。
- `serialize_public_state`をv2モデルに対応させる。
- legacy `reference`は表示用にのみ生成する。

テスト:

- 公開レスポンスのOMIM形式テスト
- Reference URL/source_type/identifierテスト
- ReferenceとEvidenceの関係テスト
- PhenotypeSearch除外テスト
- PydanticモデルのJSON serializationテスト

完了条件:

- 公開レスポンスから、各診断理由をどの文献Evidenceが支えるか追跡できる。
- `OMIM_id`がすべて`OMIM:数字`形式になる。

### Phase 8: 最終検証・評価

目的は、実データでv1より根拠追跡性が向上したことを検証すること。

作業:

- `diagnosisOutputValidationNode`を追加する。
- Candidate ID、OMIM ID、rank、Evidence ID、Source IDを検証する。
- `supported`候補にliterature Evidenceがあるか確認する。
- `supported`の最終候補が同一Candidate IDのliterature Evidenceへ追跡できるか確認する。databaseやツールrankだけでsupportedになった候補は失敗とする。
- `insufficient`候補には文献不足説明があるか確認する。
- `contradictory`候補が最終上位に残っていないか確認する。
- Evidence excerptがraw_contentに存在するか確認する。
- 英日文章のEvidence ID集合と確信度を比較する。
- 既存74ケース等を使い、v1/v2の候補、Reference、失敗率を比較する。
- `scripts/evaluation/`でZero-shot正規化前後、候補統合、暫定診断前後、最終診断を実行ごとに保存し、Hit@k・MRR・候補欠落・候補外追加・重複・identity errorを比較する。

評価指標:

- PubMed/PMC SourceRecord保持率
- Evidence IDのSource紐付け成功率
- excerpt原文一致率
- OMIM canonical形式率
- 英日出力完全率
- PhenotypeSearchのReference混入率
- entity mismatch率
- 検索失敗率、429率、平均実行時間

完了条件:

- validatorの失敗を黙って通過させない。
- v2で全診断候補の根拠追跡が可能である。
- v1互換モードからv2へ切り替える判断材料が揃う。

## 4. 実装順とコミット単位

推奨コミット単位:

1. `docs: define evidence-grounded bilingual diagnosis contract`
2. `feat: add canonical candidate source and evidence models`
3. `feat: enrich candidates with literature search metadata`
4. `feat: add PubMed PMC source retrieval`
5. `feat: extract and validate structured evidence`
6. `feat: reorder graph for evidence-first diagnosis`
7. `feat: add bilingual structured diagnosis and reflection`
8. `feat: validate references and canonical OMIM output`
9. `test: add end-to-end evidence-grounded fixtures`

各コミット後に、対象Phaseのテストと既存テストを実行する。

## 5. リスクと対策

| リスク | 対策 |
|---|---|
| 既存APIのフィールド破壊 | legacy aliasと`evidence_mode`で段階移行 |
| PubMedの検索結果が別疾患になる | Candidate identity matchingとOMIM/gene併用 |
| LLMが原文にないEvidenceを生成する | excerpt原文一致検証とEvidence ID必須化 |
| 文献が少ない希少疾患が落ちる | `insufficient`で候補維持し、文献支持済みとは表示しない |
| Referenceにツール結果が混入する | CandidateSupportとSourceRecordを型・validatorで分離 |
| EvidenceでPromptが長くなる | Candidateごとに上限、Evidence種別ごとに選択、raw_contentはPromptへ直接渡さない |
| Reflectionの再検索が無限ループする | `flow_iteration`と`max_flow_iterations`を明示 |
| 実行時間・API制限が増える | Candidate上限、query重複排除、キャッシュ、指数バックオフ |
| 日本語訳が英語に新情報を追加する | Evidence ID・confidence一致validatorと構造化出力 |

## 6. 実装開始条件

次の条件を満たしたらコード変更を開始する。

- v2仕様書に対するレビューが完了している。
- Phase 1のPydanticモデルとlegacy adapterの方針が確定している。
- v1 baselineテスト結果が保存されている。
- PubMed/PMC取得方式と利用可能な認証・レート制限の扱いが確定している。

最初に実装するファイルは次の順とする。

1. `agent/state/state_types.py`
2. `agent/utils/response_serializer.py`
3. `agent/tools/candidate_metadata.py`
4. `agent/tools/literature_search.py`
5. `agent/tools/evidence_extraction.py`
6. `agent/tools/evidence_validation.py`
7. `agent/nodes.py`
8. `agent/agent_pipeline.py`
9. `agent/tools/diagnosis.py`
10. `agent/tools/reflection.py`
11. `agent/tools/finalDiagnosis.py`
12. `agent/tools/diseaseNormalize.py`
13. `agent/tools/rankingMerge.py`
14. `agent/llm/prompt.py`
15. `agent/utils/logger.py`
16. `agent/utils/result_saver.py`
17. `tests/`

新規ファイルは`agent/tools/candidate_metadata.py`、`agent/tools/literature_search.py`、`agent/tools/evidence_extraction.py`とする。`agent/tools/diseaseSearch.py`と`agent/tools/HPOwebReserch.py`は削除せず、前者はlegacy経路、後者は補助Web検索としてv2のReference経路から切り離す。

## 7. CDLS1 Evidenceの改善実装状況

上記計画のうち、今回の「一般的な疾患説明に留まり、患者所見との関係が見えない」問題については、次の実装を完了した。

- Evidenceの原文抜粋だけを対象に、患者HPOとの一致を決定的に補完する。LLMが`patient_matches`を空で返しても、発達遅滞、成長遅延、顔貌、四肢などの対応が追跡できる。
- 患者の欠如HPOは、文献に記載された特徴と対応した場合に`explicitly_absent`として`limitations`へ分離する。未報告は欠如として扱わない。
- 疾患サブタイプの識別を追加し、CDLS1候補にCDLS4など別サブタイプの文献が紐付かないようにする。OMIM番号、疾患名、疾患ファミリー名を識別に使用する。
- 文献Evidenceを患者所見と結び付けられない場合、診断状態を自動的に`insufficient`へ下げる。`supported`は文献支持と患者所見の一致の両方を満たす場合に限る。
- 暫定診断、Reflection、最終診断の説明にも、一致した患者所見と制限事項を日英で反映する。
- 長いPMC本文はセクション・チャンクに分割し、各候補について疾患同定情報を含む先頭チャンクと、患者HPO・遺伝学・表現型に関連するチャンクを上限付きで選択する。全文をプロンプトへ渡さず、本文後半のResultsやGenotype情報も対象にする。
- Evidenceに任意の`source_chunk_id`、`source_section`、`source_start`、`source_end`を付与し、既存の`source_id`・`excerpt`・Reference番号との追跡関係を維持する。
- 上記を回帰テストへ追加し、Evidence抽出、サブタイプ除外、汎用文献の誤支持防止を検証する。

決定的テストは完了しており、全27テストが成功している。画像付き実症例の再実行は、実行環境から外部API（PubCaseFinder、GestaltMatcher、Azure OpenAI）への接続が拒否されたため、未完了である。
