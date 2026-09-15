# ResultAnalyze

`analyze_failed_cases.py` re-runs the 15 difficult cases from the 74-case
PhenoPacket set:

- all tools previously missed: `11702 11706 11710 11715 11718 11721 12293 12300`
- only some tools previously matched: `12289 12292 12299 12291 11700 11708 11714`

The script uses the original HPO IDs from
`local_artifacts/evaluation/sampleData/ValidationDataWithoutDupli_newest.tsv`
and images from
`local_artifacts/evaluation/sampleData/PhenoPacketStore_25072025/`.

It calls GestaltMatcher, PubCaseFinder, the repository FAISS vector search,
PhenoBrain, and the repository's existing GPT Zero-Shot prompt. Each case
stores the input, request metadata, returned candidates, match method, rank,
elapsed time, and any error under `ResultAnalyze/cases/`.

The historical agent exposed five candidates. The default `--rank-probe-k 100`
expands retrieval for rank checking while retaining the original depth of 5 in
the output metadata. `--include-absent-hpo` makes the GPT request include only
explicitly absent HPO terms; by default it matches the current pipeline's
default (`use_absentHPO=False`). Secret values are never written.

```bash
python ResultAnalyze/analyze_failed_cases.py --dry-run
python ResultAnalyze/analyze_failed_cases.py --model gpt-5-2
```

Use `--force` to replace existing case output. Service failures are recorded per
tool so other results remain usable.

## Verified interfaces

The implementation was checked against the repository clients before the calls were made:

| Tool | Request used | Settings |
|---|---|---|
| GestaltMatcher | `POST https://pubcasefinder.dbcls.jp/gm_endpoint/predict` with the original image as base64 JSON `img` | Basic auth from `GESTALT_API_USER` / `GESTALT_API_PASS`; original depth 5, probe depth 100 |
| PubCaseFinder | `GET https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list` with `target=omim`, `format=json`, and comma-separated present HPO IDs | No auth; original depth 5, probe depth 100 |
| VectorSimilarity | Azure OpenAI embedding followed by the repository FAISS index | `AZURE_DBCLS_JAPANEAST`, endpoint `https://dbcls-japaneast.openai.azure.com/`, API `2024-05-01-preview`, deployment `japaneast-text-embedding-3-large` |
| PhenoBrain | `GET /predict` → poll `GET /query-predict-result` → `POST /disease-list-detail` | `Ensemble`, original depth 5, probe depth 100; base URL defaults to `https://www.phenobrain.cs.tsinghua.edu.cn` |
| GPT Zero-Shot | Existing `agent.tools.ZeroShot.createZeroshot` prompt and Azure deployment wrapper | `gpt-5-2` by default; `use_absentHPO=False` by default, matching the current pipeline |

Azure GPT credentials are read using the repository's model-specific variables:
`AZURE_OPENAI_5-2_ENDPOINT`, `AZURE_OPENAI_5-2_API_KEY`,
`AZURE_OPENAI_5-2_API_VERSION`, and `AZURE_OPENAI_5-2_DEPLOYMENT_NAME`.
The script also accepts `gpt-4o` and `gpt-5-1`, which select their corresponding
repository variable prefix. Secret values are never written to the result files.

The current run is stored in `summary.csv`, `summary.json`, and `cases/*.json`.
`rank` is the 1-based rank in the fresh response; `>100` in an analysis table
means the target was not present in the 100-candidate probe, not that its exact
rank is known. The current PubCaseFinder service did not return within either
the initial 30-second retries or a separate 120-second retry; its error is
preserved in the case JSON and summary.

To retry only PubCaseFinder while retaining other saved tool results:

```bash
RESULT_ANALYZE_HTTP_TIMEOUT=120 RESULT_ANALYZE_HTTP_RETRIES=1 \
python ResultAnalyze/analyze_failed_cases.py --tools PubCaseFinder --force
```

To rebuild the aggregate files without making API calls:

```bash
python ResultAnalyze/analyze_failed_cases.py --rebuild-summary
```

The separate Absent HPO experiment is in `gpt_with_absent_hpo/`. It was run with:

```bash
python ResultAnalyze/analyze_failed_cases.py \
  --output-dir ResultAnalyze/gpt_with_absent_hpo \
  --tools GPTZeroShot --include-absent-hpo --model gpt-5-2
```

Its interpretation is documented in `gpt_with_absent_hpo/analysis.md`.

The Top 10 comparison was run under `gpt_top10_present_only/` and
`gpt_top10_with_absent_hpo/`; see `gpt_top10_comparison.md` for the comparison.
The script's `--gpt-top-k 10` option changes the prompt itself, so the model is
asked to produce ten diagnoses rather than producing five and truncating them.
