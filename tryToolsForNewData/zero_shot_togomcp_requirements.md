# Zero-shot / TogoMCP 実験 要件定義

## 目的

NewData の各症例について、画像を使わず、Present HPO、明示的な Absent HPO、性別、発症情報から GPT-5.2 が最大30疾患をStructured Outputで順位付けする。

MCPあり条件では、固定したPubMed検索結果を与えるのではなく、TogoMCPのUsage Guideに従い、GPTが必要な外部情報とツール引数を選択する。候補疾患の初期ランキング用の `pubcasefinder_rank_by_phenotypes` は使用しない。

## 入力契約

- Present HPO: IDとラベル
- Absent HPO: IDとラベル
- 性別
- 発症情報
- 画像パスはメタデータとして保存するが、LLMメッセージには含めない
- 正解疾患はPromptには含めない。評価用の保存データだけに残す

## LLM条件

- Model: Azure GPT-5.2
- `reasoning_effort=medium`
- 画像なし、テキスト入力のみ
- 出力: `rank`, `disease_name`, `omim_id`
- 最大30件
- OMIM IDを根拠なく生成しない。不明なら `null`
- 説明文は出力しない
- 出力前に、表現型整理、特異性評価、候補生成、Absent HPO照合、順位付けを順番に内部推論する

## MCPなし条件

HPO、Absent HPO、性別、発症情報だけを使ってZero-shot推論する。外部検索は禁止する。

## MCPあり条件

### 事前ゲート

1. `TogoMCP_Usage_Guide`を最初に呼ぶ
2. `tools/list`の結果から実験用allow-listを作る
3. `pubcasefinder_rank_by_phenotypes`および同等のPubCaseFinder表現型ランキングをallow-listから除外する

### GPT主導の検索

GPTに次のツールのスキーマを渡し、必要なツール呼び出しをtool callとして選ばせる。

ツール選択前には、表現型の要約、未解決の情報ギャップ、必要なデータベース、最小の呼び出し列を内部推論する。検索結果ごとにランキングへの影響を再評価し、追加情報が不要なら停止する。

- `ncbi_esearch`
- `ncbi_esummary`
- `ncbi_efetch`
- `get_MIE_file`
- `run_sparql`
- `get_sparql_endpoints`
- `get_graph_list`
- `get_workflow`
- `search_mesh_descriptor`
- `pubcasefinder_get_case_reports`
- TogoIDのID変換・関係確認ツール

GPTは、必要がなければ検索せずに終了できる。

### チュートリアル遵守

- `run_sparql`の前に対象DBの`get_MIE_file`を取得する
- MIE未取得で`run_sparql`が要求された場合は、ゲート側でMIEを自動取得する
- 検索結果は次の推論ターンに渡す前に短く要約する
- 完全なMCPレスポンスはトレースに保存する

### 計算量制限

- 最大3回のGPT計画ターン
- MCPツール呼び出し最大6回
- PubMed Abstract等の次ターン入力は文字数を制限する
- 制限に到達した場合は、得られた情報だけで最終Zero-shotを実行する

## 出力契約

各症例について以下を保存する。

- `LLM/{image_id}.json`
- `LLMwithMCP/{image_id}.json`

各JSONは `schema_version: "zero_shot_ranking.v2"` とし、ランキングは重複保存しない。

- `ranking`: 評価に使う唯一の正規化済みランキング（1〜30位）
- `trace`: 監査・デバッグ用情報を格納する領域

- 症例入力と正解情報（評価用）
- 最終Prompt（`trace.prompt`）
- MCP plannerのPromptと各ターンの応答（`trace.mcp_planner`）
- MCPツール名、引数、完全レスポンス、実行時間（`trace.mcp_calls`）
- Structured Outputの生レスポンス（`trace.llm.raw_response`）
- reasoning effortと入力モダリティ

## 実装ファイル

実行スクリプトは `run_zero_shot_for_new_data.py` とする。既定ではMCPなし（`--variant llm`）で実行する。MCPありを実行する場合は`--variant mcp`、両方は`--variant both`を指定する。出力フォルダと`LLM/`、`LLMwithMCP/`は存在しなければ自動作成する。既定の最大並列数は5（`--workers`で変更可能）。各ジョブには既定600秒のタイムアウト（`--job-timeout`で変更可能）を設定し、タイムアウトした症例はエラーJSONを保存して他症例を継続する。既存JSONはデフォルトでスキップし、`--overwrite`で再実行する。
