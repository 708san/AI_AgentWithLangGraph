# public_response.json 仕様書（日本語）

## 1. この文書について

この文書は、ZebraSeek の新版フロー（evidence_mode が v2）がフロントエンドへ渡す
public_response.json の形式を説明するものです。

基準にしたサンプルは次の実行結果です。

[public_response.json](/Users/yoshikuwa-n/Downloads/WorkForBioHackathon/AI_AgentWithLangGraph/testCode/phenopacket_image_case/runs/patient_276_20261016_162445_553292/results/public_response.json)

JSONには次の情報が含まれます。

1. 患者情報と入力HPO
2. 各診断ツールの結果
3. ツール結果を統合した疾患候補
4. 文献検索で取得した文献
5. 文献から抽出したEvidence
6. 暫定診断、検証結果、最終診断

LLMのプロンプト、文献本文全体、検索クエリ、取得時刻などは、フロント向けJSONには含めません。

## 1.1 旧版JSONとの関係

添付された旧版JSONでは、各結果や推論に根拠を付ける設計がすでに存在していました。
ただし、根拠の形式がノードごとに異なり、ツール結果と文献が同じ文字列配列に混在していました。

| 旧版の場所 | 旧版での根拠の形式 | 新版での扱い |
|---|---|---|
| tentativeDiagnosis.reference | PubCaseFinder、ZeroShot、GestaltMatcherなどのツール名を並べた文字列 | 暫定診断は文献検索前なので、通常はreferencesが空。ツール根拠は候補やtool_rankingsから確認する。 |
| reflection.ans[].references | PubMed、Wikipedia、PhenotypeSearch、GestaltMatcherなどを含む自由記述の配列 | 文献はEvidenceとReferenceに分離する。ツールの順位やスコアは文献Referenceにしない。 |
| finalDiagnosis.reference | [1]、[2]と本文を対応させた長い文字列 | referencesを構造化し、ans[].citation_numbersから対応付ける。referenceは表示用の補助文字列として残す。 |

旧版のように、ツール名だけを文献Referenceとして扱ったり、
存在しない外部リンクをReferenceに含めたりしないことが新版の方針です。

## 1.2 ノードごとの根拠の扱い

新版では、根拠を次の2種類に分けます。

| 根拠の種類 | 内容 | 主な保存場所 |
|---|---|---|
| ツール根拠 | ツールが候補を何位にしたか、どのスコアだったか、どのHPOや注記が対応したか | mergedDiseaseCandidates[].tool_rankings |
| 文献Evidence | 文献に実際に書かれている主張、原文抜粋、文献URL | evidence、finalDiagnosis.references[].evidence |

ツール根拠は候補を作った理由として表示できますが、それだけで医学的な文献根拠とは扱いません。
文献に基づく推論では、Evidenceのclaimとexcerptを使い、ReferenceのURLを表示します。

処理の流れと根拠の対応は次の通りです。

~~~text
ツール結果
  → mergedDiseaseCandidates
     └─ ツール名、順位、スコア、注記
  → tentativeDiagnosis
     └─ ツール結果に基づく暫定候補
        ※この時点では文献検索前
  → sources
     └─ 今回検索して取得した文献
  → evidence
     └─ 文献から抽出した主張と原文抜粋
  → reflection
     └─ 患者所見とEvidenceを照合した検討
  → finalDiagnosis
     └─ Evidenceに基づく最終説明とReference
~~~

## 1.3 暫定診断時点のReference

tentativeDiagnosisは文献検索より前に作られます。
そのため、現在の新版フローでは、暫定診断の時点で文献Referenceはありません。

通常は次の状態です。

~~~json
{
  "tentativeDiagnosis": {
    "ans": [
      {
        "rank": 1,
        "OMIM_id": "OMIM:...",
        "disease_name": "候補疾患名",
        "description": "ツール結果に基づく暫定説明"
      }
    ],
    "reference": null
  }
}
~~~

新版の公開形式では整形後にreferencesも保持されますが、通常は空配列です。

~~~json
{
  "tentativeDiagnosis": {
    "references": [],
    "reference": null
  }
}
~~~

暫定診断のdescriptionにツール結果を踏まえた説明があっても、
それは文献Citationではありません。文献番号の[1]を付けてはいけません。

## 1.4 フロント側で主に使うキー

JSONの形式は変更せず、用途だけを次のように整理します。

### 最終診断画面で使うキー

| JSONパス | 用途 |
|---|---|
| finalDiagnosis.ans | 最終的な疾患候補の一覧。 |
| finalDiagnosis.ans[].rank | 表示順位。 |
| finalDiagnosis.ans[].disease_name_en | 英語疾患名。 |
| finalDiagnosis.ans[].disease_name_ja | 日本語疾患名。 |
| finalDiagnosis.ans[].description_en | 英語の説明。 |
| finalDiagnosis.ans[].description_ja | 日本語の説明。 |
| finalDiagnosis.ans[].literature_status | 文献根拠の状態。 |
| finalDiagnosis.ans[].citation_numbers | その候補が引用したReference番号。 |
| finalDiagnosis.references | 最終診断で使用された文献一覧。 |
| finalDiagnosis.references[].number | 引用番号。 |
| finalDiagnosis.references[].title | 論文タイトル。 |
| finalDiagnosis.references[].identifier | PMIDまたはPMCID。 |
| finalDiagnosis.references[].url | 論文へのリンク。 |
| finalDiagnosis.references[].evidence | その文献を根拠にした主張と原文抜粋。 |
| finalDiagnosis.validation_status | 最終診断の検証状態。 |

最終診断の文章にある[1]は、同じfinalDiagnosis内の
finalDiagnosis.references[].numberが1のReferenceに対応します。

### 症例情報として使うキー

| JSONパス | 用途 |
|---|---|
| patient_id | 症例の識別。 |
| imagePath | 顔貌画像などの表示。 |
| hpoDict | 患者に認められた所見の表示。 |
| absentHpoDict | 明示的に認められない所見の表示。 |
| sex | 性別。 |
| onset | 発症時期。 |
| clinicalText | 自由記載の臨床情報。 |

### 詳細ページで必要に応じて使うキー

| JSONパス | 用途 |
|---|---|
| tentativeDiagnosis | 文献検索前の暫定候補。 |
| reflection | 患者所見とEvidenceの関係を検討した結果。 |
| mergedDiseaseCandidates | ツールを統合した候補一覧。 |
| mergedDiseaseCandidates[].tool_rankings | ツールごとの順位やスコア。 |
| sources | 検索で取得した全ての文献。最終診断で未使用の文献も含む。 |
| evidence | 全候補から抽出されたEvidence。最終診断で未使用のものも含む。 |

### 通常の診断表示では直接使わないキー

| JSONパス | 理由 |
|---|---|
| pubCaseFinder | ツール詳細を表示する場合だけ使用。 |
| GestaltMatcher | ツール詳細を表示する場合だけ使用。 |
| phenotypeSearchResult | ツール詳細を表示する場合だけ使用。 |
| zeroShotResult | ツール詳細を表示する場合だけ使用。 |
| phenoBrain | ツールを使用した場合だけ存在する補助情報。 |
| webresources | 外部検索結果の補助情報。 |
| memory | 内部処理または過去情報の補助情報。 |
| validation_errors | エラー表示が必要な場合だけ使用。 |
| flow_iteration / max_flow_iterations | 実行状態の確認用。 |

最終診断の出典表示には、トップレベルのsourcesやevidenceを直接使わず、
finalDiagnosis.referencesを使うことを推奨します。

## 2. データ型

| 表記 | 意味 |
|---|---|
| string | 文字列 |
| integer | 整数 |
| number | 小数を含む数値 |
| boolean | true または false |
| object | キーと値の組み合わせ |
| array<T> | 型Tの配列 |
| null | 値がない状態 |
| T \| null | 型Tまたはnull |

## 3. 全体構造

~~~text
public_response
├─ 患者情報
│  ├─ depth / imagePath / clinicalText
│  ├─ hpoList / hpoDict
│  ├─ absentHpoList / absentHpoDict
│  └─ onset / sex / patient_id
├─ ツール結果
│  ├─ pubCaseFinder
│  ├─ phenoBrain
│  ├─ GestaltMatcher
│  ├─ phenotypeSearchResult
│  └─ zeroShotResult
├─ mergedDiseaseCandidates
├─ 文献とEvidence
│  ├─ sources
│  └─ evidence
├─ tentativeDiagnosis
├─ reflection
├─ finalDiagnosis
└─ 実行・検証状態
   ├─ evidence_mode
   ├─ validation_status / validation_errors
   └─ flow_iteration / max_flow_iterations
~~~

## 4. トップレベルのキー

### 4.1 患者情報

| キー | 型 | 内容 |
|---|---|---|
| depth | integer | 処理の深さ。通常は1。 |
| imagePath | string | 入力画像のパス。 |
| clinicalText | string \| null | 自由記載の臨床情報。入力がない場合はnull。 |
| hpoList | array<string> | 患者に認められたHPO IDの一覧。 |
| hpoDict | object<string, string> | HPO IDからHPO名への対応。 |
| absentHpoList | array<string> | 明示的に認められないと入力されたHPO IDの一覧。 |
| absentHpoDict | object<string, string> | 明示的に認められないHPO IDからHPO名への対応。 |
| onset | string | 発症時期。未知の場合はUnknownなど。 |
| sex | string | 性別。 |
| patient_id | string | 症例ID。 |

absentHpoListにない所見は、「ない」とは限りません。未入力・未報告の可能性があります。

### 4.2 実行設定・検証状態

| キー | 型 | 内容 |
|---|---|---|
| evidence_mode | string | 現在の出力モード。新版はv2。 |
| use_absentHPO | boolean | absent HPOを処理に使用したか。 |
| use_phenobrain | boolean | PhenoBrainを使用したか。 |
| filter_impotance | boolean | 入力時の重要度フィルタ設定。既存仕様のキー名を維持。 |
| flow_iteration | integer | 文献検索・検証フローの反復回数。 |
| max_flow_iterations | integer | 許可された最大反復回数。 |
| validation_status | string | 全体の検証状態。passed、partial、failed。 |
| validation_errors | array<string> | 構造やIDの検証エラー。問題がなければ空配列。 |

validation_statusの意味は次の通りです。

| 値 | 意味 |
|---|---|
| passed | 構造上の問題がなく、診断候補の文献状態も解決している。 |
| partial | 構造上のエラーはないが、文献Evidenceが付いていない候補などが残っている。 |
| failed | 候補ID、Evidence ID、順位、OMIM IDなどに検証エラーがある。 |

partialでもvalidation_errorsが空の場合があります。これは「エラーはないが、全候補を文献で確認できていない」という意味です。

## 5. 各ツールの結果

### 5.1 pubCaseFinder

型：array<object>

| キー | 型 | 内容 |
|---|---|---|
| omim_disease_name_en | string | ツールが返した英語疾患名。 |
| description | string | ツールが返した疾患説明。 |
| score | number | ツールの類似度・スコア。 |
| omim_id | string | OMIM ID。OMIM:数字形式。 |
| rank | integer | PubCaseFinder内での順位。 |
| disease_name | string | ツール由来の疾患名。 |

### 5.2 GestaltMatcher

型：array<object>

| キー | 型 | 内容 |
|---|---|---|
| subject_id | string | 類似画像または症例のID。 |
| syndrome_name | string | 症候群名。 |
| omim_id | string | OMIM ID。ない場合は空文字列の場合がある。 |
| image_id | string | 類似画像のID。 |
| score | number | 表示用に整えた類似度スコア。 |
| rank | integer | GestaltMatcher内での順位。 |
| rank_source | string | 順位の生成元。 |
| raw_distance | number \| null | ツールの元の距離。 |

### 5.3 phenotypeSearchResult

型：array<object>

| キー | 型 | 内容 |
|---|---|---|
| disease_info | object | 疾患情報。OMIM_id、disease_name、synonym、definition、phenotypeなどを持つ。 |
| similarity_score | number | 表現型類似度。 |
| rank | integer | Phenotype Search内での順位。 |

disease_info.phenotypeはarray<string>、disease_info.definitionはstringまたはnullです。

### 5.4 zeroShotResult

型：object

~~~json
{
  "ans": [
    {
      "disease_name": "CORNELIA DE LANGE SYNDROME 1; CDLS1",
      "rank": 1,
      "OMIM_id": "OMIM:122470"
    }
  ]
}
~~~

zeroShotResult.ansはarray<object>です。

| キー | 型 | 内容 |
|---|---|---|
| disease_name | string | Zero-shotが生成した疾患名。 |
| rank | integer | Zero-shot内での順位。 |
| OMIM_id | string | OMIM ID。 |

### 5.5 phenoBrain、webresources、memory

いずれも現在は配列として渡されます。

~~~text
phenoBrain: array<object>
webresources: array<object>
memory: array<object>
~~~

ツール未使用または結果がない場合は空配列です。ツール固有のキーはツール仕様に依存します。

## 6. mergedDiseaseCandidates

型：array<object>

複数ツールの結果をOMIM ID単位で統合した疾患候補です。

| キー | 型 | 内容 |
|---|---|---|
| candidate_id | string | 内部で候補を識別するID。現在はOMIM:数字。 |
| disease_name | string | 統合後の疾患名。 |
| OMIM_id | string | OMIM ID。 |
| gene_symbols | array<string> | 関連遺伝子記号。 |
| synonyms | array<string> | 疾患の別名。 |
| phenotype_labels | array<string> | 疾患に関連する表現型名。 |
| consensus_count | integer | 複数ツールで候補になった数。 |
| best_rank | integer | 統合時の代表順位。 |
| tool_supports | array<object> | 各ツールが候補を支持した情報。 |
| tool_rankings | array<object> | フロント表示用に整えたツール別順位。 |

tool_supportsとtool_rankingsの要素は次の構造です。

| キー | 型 | 内容 |
|---|---|---|
| tool | string | ツール名。 |
| rank | integer | そのツール内での順位。 |
| score | number \| null | スコア。取得できない場合はnull。 |
| rank_source | string | 順位の生成方法。 |
| raw_distance | number \| null | 元の距離。 |
| matched_hpo_id | string \| null | 対応するHPO ID。 |
| note | string | 補足情報。 |
| citation_allowed | boolean | 文献出典として扱えるか。通常、ツール結果はfalse。 |

ツールの順位は診断候補を作るための補助情報であり、文献の出典ではありません。

## 7. 文献情報 sources

型：array<object>

検索で取得し、出典として利用可能と判定された文献です。

| キー | 型 | 内容 |
|---|---|---|
| source_id | string | 文献を内部で識別するID。例：SRC-aba7bcdc74。 |
| source_type | string | 現在は主にliterature。 |
| source_name | string | 取得元。PubMedまたはPMC。 |
| title | string | 論文タイトル。 |
| url | string | フロントから開くURL。 |
| identifier | string | PMID:数字またはPMCID:PMC数字。 |
| pmid | string \| null | PubMed ID。 |
| pmcid | string \| null | PubMed Central ID。 |
| doi | string \| null | DOI。 |
| authors | array<string> | 著者名。 |
| journal | string \| null | 掲載誌。 |
| publication_year | string \| null | 出版年。 |
| candidate_ids | array<string> | この文献が検索された候補ID。 |
| disease_name | string \| null | 検索時に対応付けられた疾患名。 |
| OMIM_id | string \| null | 検索時に対応付けられたOMIM ID。 |
| gene_symbols | array<string> | 検索対象候補の遺伝子。 |
| content_type | string | abstractまたはfull_text。 |
| citation_allowed | boolean | 最終診断の出典として利用可能か。 |

フロント向けsourcesには文献本文全体、検索クエリ、取得時刻は含めません。

## 8. Evidence evidence

型：array<object>

文献から、実際に本文または抄録に記載されている内容を抽出したものです。
ツールのスコアやLLMの内部知識ではありません。

| キー | 型 | 内容 |
|---|---|---|
| candidate_id | string | このEvidenceが関係する疾患候補。 |
| claim_en | string | 文献から判断できる主張の英語要約。 |
| claim_ja | string | 同じ主張の日本語。 |
| excerpt | string | 文献から実際に抜き出した原文。 |
| reference_number | integer \| null | 対応するReference番号。 |

Evidenceと文献の対応はreference_numberを介して確認します。

~~~text
evidence.reference_number = 1
        ↓
finalDiagnosis.references[].number = 1
        ↓
finalDiagnosis.references[].url / title / evidence
~~~

最終診断の表示では、トップレベルのevidenceよりも、
finalDiagnosis.references[].evidenceを使う方が文献との対応を追いやすくなります。

## 9. tentativeDiagnosis と finalDiagnosis

両者の外側の構造は同じですが、現在の新版フローでは役割が異なります。

新版フローの順番は次の通りです。

~~~text
ツール結果
  → tentativeDiagnosis
  → 文献検索
  → Evidence抽出
  → reflection
  → finalDiagnosis
~~~

そのため、tentativeDiagnosisは文献検索より前に作られ、通常は文献Referenceを持ちません。
文献Referenceを使って最終候補を説明するのはfinalDiagnosisです。

~~~text
diagnosis
├─ ans: array<diagnosis item>
├─ references: array<reference>
├─ validation_status: string
├─ validation_errors: array<string>
└─ reference: string | null
~~~

### 9.1 ans

型：array<object>

| キー | 型 | 内容 |
|---|---|---|
| candidate_id | string | mergedDiseaseCandidatesのcandidate_idと対応する候補ID。 |
| disease_name_en | string | 英語の疾患名。 |
| disease_name_ja | string | 日本語の疾患名。 |
| OMIM_id | string | OMIM ID。 |
| rank | integer | この診断リスト内での順位。最終診断では1から連続する。 |
| description_en | string | 英語の診断説明。必要に応じて[1]形式の引用番号を含む。 |
| description_ja | string | 日本語の診断説明。必要に応じて[1]形式の引用番号を含む。 |
| literature_status | string | 文献Evidenceの状態。 |
| disease_name | string | フロント互換用の代表疾患名。通常は英語名。 |
| description | string | フロント互換用の代表説明。通常は英語説明。 |
| citation_numbers | array<integer> | この候補の説明で使用したReference番号。 |

### 9.2 literature_status

| 値 | 意味 |
|---|---|
| supported | 文献Evidenceがあり、候補を支持する内容として扱える。 |
| insufficient | 文献はあるが、診断を支持するには不十分。 |
| contradictory | 文献内容が候補と矛盾する方向。 |
| not_retrieved | 最終診断の候補に文献Evidenceが紐づいていない。検索結果全体に文献がないことと必ずしも同義ではない。 |

### 9.3 references

型：array<object>

この診断ブロックで実際に引用された文献だけが入ります。

ただし、現在の新版フローではtentativeDiagnosisが文献検索より前に作られるため、
通常は次の状態になります。

~~~json
{
  "references": [],
  "reference": null
}
~~~

tentativeDiagnosisにReferenceが入るのは、将来フローを変更した場合、検索済みStateを再利用した場合、
または旧版・別経路の結果を格納した場合です。

| キー | 型 | 内容 |
|---|---|---|
| number | integer | この診断ブロック内のReference番号。 |
| source_name | string | PubMedまたはPMC。 |
| title | string | 論文タイトル。 |
| identifier | string | PMIDまたはPMCID。 |
| url | string | 論文ページへのURL。 |
| authors | array<string> | 著者名。 |
| journal | string \| null | 掲載誌。 |
| publication_year | string \| null | 出版年。 |
| evidence | array<object> | この文献から抽出され、診断で使用されたEvidence。 |

references[].evidenceの各要素は次の形式です。

| キー | 型 | 内容 |
|---|---|---|
| claim_en | string | 英語の文献主張。 |
| claim_ja | string | 日本語の文献主張。 |
| excerpt | string | 文献からの原文抜粋。 |

### 9.4 reference

型：string \| null

表示用の簡易な引用文字列です。

~~~text
[1] PubMed: 論文タイトル https://pubmed.ncbi.nlm.nih.gov/...
~~~

構造化された処理やURL取得にはreferencesを使い、referenceは単純なテキスト表示用に使います。

### 9.5 reflection

型：object

reflectionは、暫定診断候補について、患者所見と文献Evidenceの関係を検討した結果です。

~~~text
reflection
└─ ans: array<object>
~~~

| キー | 型 | 内容 |
|---|---|---|
| candidate_id | string | 検討対象の疾患候補ID。 |
| disease_name_en | string | 英語の疾患名。 |
| disease_name_ja | string | 日本語の疾患名。 |
| correctness | boolean | 暫定診断が妥当と判断されたか。 |
| patient_summary_en | string | 患者所見の英語要約。 |
| patient_summary_ja | string | 患者所見の日本語要約。 |
| diagnosis_analysis_en | string | 患者所見と文献Evidenceの関係を説明する英語文。 |
| diagnosis_analysis_ja | string | 患者所見と文献Evidenceの関係を説明する日本語文。 |
| literature_status | string | 文献Evidenceの状態。 |
| citation_numbers | array<integer> | 分析文で参照したReference番号。 |

reflection自体にはreferences配列がありません。表示上のURLや論文情報が必要な場合は、
最終診断ではfinalDiagnosis.references、全体ではsourcesとevidenceの対応を使います。

ただし、現在の公開形式ではreflectionのcitation_numbersだけから、
reflection専用のReference URLを一意に取得する構造にはなっていません。
そのため、フロントでクリック可能な出典を表示する場合は、
最終診断の出典表示をfinalDiagnosis.referencesから行うことを推奨します。
reflectionの推論文自体は、患者所見とEvidenceを検討した説明として表示します。

旧版のreflection.ans[].referencesのように、候補ごとに文献やツールの説明を自由記述で持たせる方式は、
文献とツールの区別や番号対応が難しいため、新版では採用していません。

## 10. Reference番号の付け方と対応関係

### 10.1 最終診断での対応

最終診断では、次の関係になります。

~~~text
finalDiagnosis.ans[0].description_en
    "... [1]"

finalDiagnosis.ans[0].citation_numbers
    [1]
        ↓
finalDiagnosis.references[0].number
    1
        ↓
finalDiagnosis.references[0]
    ├─ title
    ├─ identifier
    ├─ url
    └─ evidence
       ├─ claim_en / claim_ja
       └─ excerpt
~~~

フロント側では次の手順で表示します。

1. finalDiagnosis.ansから診断候補を表示する。
2. citation_numbersを取得する。
3. その番号と一致するfinalDiagnosis.references[].numberを探す。
4. 該当Referenceのtitle、evidence、urlを表示する。
5. [1]のクリック先としてReferenceのurlを使う。

### 10.2 具体例

~~~json
{
  "candidate_id": "OMIM:300882",
  "description_en": "The patient's features align with CdLS5. [1]",
  "citation_numbers": [1]
}
~~~

この場合、同じfinalDiagnosis内でnumberが1のReferenceを探します。

~~~json
{
  "number": 1,
  "title": "A Classic Cornelia De Lange Syndrome Type 5 ...",
  "url": "https://pubmed.ncbi.nlm.nih.gov/38910710/"
}
~~~

[1]はタイトルの順番や検索順位ではなく、
finalDiagnosis.references[].number == 1で対応付けます。

### 10.3 tentativeDiagnosisとfinalDiagnosisの番号は別管理

現在の実装では、tentativeDiagnosis.referencesとfinalDiagnosis.referencesはそれぞれ独立して番号を振ります。

そのため、次の2つの[1]が同じ文献とは限りません。

~~~text
tentativeDiagnosisの[1]
    → tentativeDiagnosis.references[].number == 1

finalDiagnosisの[1]
    → finalDiagnosis.references[].number == 1
~~~

最終結果画面では、finalDiagnosisだけを表示対象にすると番号の混乱を避けられます。

### 10.4 トップレベルevidenceのreference_number

トップレベルのevidenceにもreference_numberがあります。これは検索・Evidence全体を対象にした対応番号です。

一方、finalDiagnosis.references[].numberは最終診断ブロック内で使用した文献だけを対象にした番号です。

| 場所 | 番号の範囲 | 主な用途 |
|---|---|---|
| evidence[].reference_number | 全Evidenceを対象 | 全体のEvidence確認 |
| finalDiagnosis.references[].number | 最終診断で使用した文献のみ | フロントの最終診断表示 |

フロントで最終診断の[1]を表示する場合は、
トップレベルevidence[].reference_numberではなく、
finalDiagnosis.references[].numberを使ってください。

## 11. ReferenceとEvidenceの役割の違い

| データ | 役割 |
|---|---|
| sources | 検索で取得した文献の情報。タイトル、ID、URLなど。 |
| evidence | 文献本文から抽出した具体的な主張と原文抜粋。 |
| finalDiagnosis.ans[].citation_numbers | 診断説明がどのReferenceを使ったか。 |
| finalDiagnosis.references | 診断説明に実際に使われた文献と、そのEvidence。 |
| finalDiagnosis.reference | Referenceの簡易な文字列表示。 |

ツールの順位・スコアは診断候補を作るための補助情報です。文献の出典として表示する場合は、referencesの文献を使います。

## 12. フロント実装時の推奨表示方法

最終診断を表示する場合は、次の構造を利用します。

~~~text
finalDiagnosis
├─ ans[]
│  ├─ disease_name_en / disease_name_ja
│  ├─ rank
│  ├─ description_en / description_ja
│  ├─ literature_status
│  └─ citation_numbers
└─ references[]
   ├─ number
   ├─ title
   ├─ identifier
   ├─ url
   └─ evidence[]
      ├─ claim_en / claim_ja
      └─ excerpt
~~~

概念的な対応処理は次の通りです。

~~~text
for diagnosis in finalDiagnosis.ans:
    for number in diagnosis.citation_numbers:
        reference = finalDiagnosis.references.find(
            reference.number == number
        )
        表示(reference.title, reference.evidence, reference.url)
~~~

citation_numbersが空の場合、その候補には現在の最終診断説明から直接参照された文献がありません。
この場合はReferenceを無理に表示せず、literature_statusに応じて表示します。

## 13. 現時点での注意点

1. finalDiagnosis.referencesは、検索で取得した全論文ではなく、最終診断がEvidenceを引用した論文だけです。
2. 検索済みの全論文を確認する場合はトップレベルのsourcesを見ます。
3. not_retrievedは「PubMedに文献が存在しない」という意味ではなく、最終診断候補にEvidenceが紐づいていないという意味になる場合があります。
4. tentativeDiagnosisとfinalDiagnosisのReference番号は独立しています。
5. ツール結果は診断候補の補助情報であり、文献Referenceではありません。
6. source_idやevidence_idは内部追跡用です。現在のフロント向けEvidenceでは、Reference番号を中心に文献と対応付けます。

