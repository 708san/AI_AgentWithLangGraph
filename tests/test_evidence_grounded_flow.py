import inspect
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
from agent.state.state_types import (
    CandidateSupport,
    DiagnosisFormatV2,
    DiagnosisOutputV2,
    EvidenceDraft,
    EvidenceExtractionOutput,
    EvidenceItem,
    FinalDiagnosisGenerationOutput,
    MergedDiseaseCandidateV2,
    ReflectionFormatV2,
    ReflectionOutputV2,
    SourceRecord,
    ZeroShotReasonedOutput,
)
from agent.tools.ZeroShot import createZeroshot
from agent.tools.candidate_metadata import candidate_id_for, to_v2_candidate
from agent.tools.evidence_extraction import (
    _select_relevant_chunks,
    _source_chunks,
    extract_evidence_for_candidates,
)
from agent.tools.evidence_validation import validate_v2_output
from agent.tools.literature_search import (
    build_literature_queries,
    matches_candidate_identity,
    search_candidate_literature,
    search_literature_for_candidates,
)
from agent.tools.reflection import create_reflection_v2
from agent.tools.diagnosis import validate_v2_candidate_output
from agent.tools.finalDiagnosis import _apply_evidence_contract, createFinalDiagnosisV2
from agent.nodes import evidenceExtractionNode
from agent.utils.response_serializer import (
    _add_citations,
    serialize_external_compatibility,
    serialize_node_result,
    serialize_public_state,
)


class TestEvidenceGroundedFlow(unittest.TestCase):
    def test_flow_log_diagram_matches_selected_evidence_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = RareDiseaseDiagnosisPipeline.__new__(RareDiseaseDiagnosisPipeline)
            pipeline.logfile_path = str(Path(tmp) / "agent.log")
            pipeline.graph = Mock()
            pipeline.graph.get_graph.return_value.draw_ascii.return_value = "LEGACY GRAPH"
            pipeline.v2_graph = Mock()
            pipeline.v2_graph.get_graph.return_value.draw_ascii.return_value = "V2 GRAPH"

            pipeline._write_graph_ascii_to_log("legacy")
            legacy_log = Path(pipeline.logfile_path).read_text(encoding="utf-8")
            self.assertIn("evidence_mode=legacy", legacy_log)
            self.assertIn("LEGACY GRAPH", legacy_log)
            self.assertNotIn("V2 GRAPH", legacy_log)

            pipeline._write_graph_ascii_to_log("v2")
            v2_log = Path(pipeline.logfile_path).read_text(encoding="utf-8")
            self.assertIn("evidence_mode=v2", v2_log)
            self.assertIn("V2 GRAPH", v2_log)

    def setUp(self):
        self.candidate = MergedDiseaseCandidateV2(
            candidate_id="OMIM:123456",
            disease_name="Example syndrome",
            OMIM_id="OMIM:123456",
            gene_symbols=["EXG1"],
            synonyms=["Example disease"],
            phenotype_labels=["Hypotonia"],
            consensus_count=2,
            best_rank=1,
        )
        self.source = SourceRecord(
            source_id="SRC-1",
            source_type="literature",
            source_name="PubMed",
            title="Example syndrome and EXG1",
            identifier="PMID:12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            candidate_ids=["OMIM:123456"],
            content_type="abstract",
            raw_content="Example syndrome is associated with hypotonia and EXG1 variants.",
            citation_allowed=True,
        )
        self.evidence = EvidenceItem(
            evidence_id="EVD-1",
            candidate_id="OMIM:123456",
            source_id="SRC-1",
            evidence_type="core_phenotype",
            claim_en="The disease is associated with hypotonia.",
            claim_ja="この疾患は筋緊張低下と関連します。",
            excerpt="Example syndrome is associated with hypotonia",
            support="supports",
            confidence="high",
        )

    def test_candidate_ids_and_queries(self):
        self.assertEqual(candidate_id_for("Example syndrome", "123456"), "OMIM:123456")
        candidate = to_v2_candidate({
            "disease_name": "Example syndrome",
            "OMIM_id": "123456",
            "tool_rankings": [{"tool": "GestaltMatcher", "rank": 1, "rank_source": "api"}],
        })
        self.assertEqual(candidate.candidate_id, "OMIM:123456")
        self.assertEqual(candidate.tool_supports[0].rank_source, "api")
        self.assertTrue(any("Example syndrome" in query for query in build_literature_queries(self.candidate, ["Hypotonia"])))

    def test_zero_shot_selection_reason_is_not_in_downstream_output(self):
        generated = ZeroShotReasonedOutput(ans=[{
            "disease_name": "Example syndrome",
            "rank": 1,
            "OMIM_id": "123456",
            "selection_reason": "Hypotonia is explicitly present in the supplied HPO terms.",
        }])
        llm = Mock()
        llm.get_structured_llm.return_value = Mock()
        llm.invoke_with_content_filter_retry.return_value = generated
        reasons = []
        output, prompt = createZeroshot({
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "onset": "Unknown",
            "sex": "unknown",
            "llm": llm,
        }, reasoning_sink=reasons.append)
        self.assertNotIn("selection_reason", output.model_dump_json())
        self.assertEqual(reasons[0]["ans"][0]["selection_reason"], generated.ans[0].selection_reason)
        self.assertIn("selection_reason", prompt)

    def test_tentative_identity_validation_allows_partial_selection(self):
        valid = DiagnosisOutputV2(ans=[{
            "candidate_id": "OMIM:123456",
            "disease_name_en": "Example syndrome",
            "disease_name_ja": "Example syndrome",
            "OMIM_id": "OMIM:123456",
            "rank": 1,
            "description_en": "Supported by the supplied evidence.",
            "description_ja": "提供された根拠により支持されます。",
            "evidence_ids": [],
            "literature_status": "not_retrieved",
        }])
        validation = validate_v2_candidate_output(valid, [self.candidate])
        self.assertTrue(validation["matched"])
        self.assertEqual(validation["omitted_candidate_ids"], [])

        invalid = valid.model_copy(deep=True)
        invalid.ans[0].OMIM_id = "OMIM:654321"
        validation = validate_v2_candidate_output(invalid, [self.candidate])
        self.assertFalse(validation["matched"])
        self.assertTrue(any("OMIM mismatch" in error for error in validation["errors"]))

    def test_literature_identity_rejects_conflicting_subtype(self):
        cdls1 = self.candidate.model_copy(update={
            "disease_name": "Cornelia de Lange syndrome 1; CDLS1",
        })
        self.assertFalse(matches_candidate_identity({
            "title": "First Cornelia de Lange Syndrome Type 4 caused by RAD21 deletion",
            "abstract": "This is a report of CDLS4.",
        }, cdls1))
        self.assertTrue(matches_candidate_identity({
            "title": "Cornelia de Lange syndrome: clinical features",
            "abstract": "A clinical review of Cornelia de Lange syndrome.",
        }, cdls1))

    def test_literature_search_covers_all_merged_candidates(self):
        second_candidate = self.candidate.model_copy(update={
            "candidate_id": "OMIM:654321",
            "disease_name": "Another syndrome",
            "OMIM_id": "OMIM:654321",
        })
        searched = []

        def fake_search(candidate, **_kwargs):
            searched.append(candidate.candidate_id)
            return []

        with patch("agent.tools.literature_search.search_candidate_literature", side_effect=fake_search):
            sources = search_literature_for_candidates(
                [self.candidate, second_candidate],
                hpo_dict={"HP:1": "Hypotonia"},
            )

        self.assertEqual(searched, ["OMIM:123456", "OMIM:654321"])
        self.assertEqual(sources, [])

    def test_literature_queries_prioritize_identifier_and_disease_synonyms(self):
        candidate = self.candidate.model_copy(update={
            "candidate_id": "OMIM:618505",
            "disease_name": "STOLERMAN NEURODEVELOPMENTAL SYNDROME; NEDSST",
            "OMIM_id": "OMIM:618505",
            "gene_symbols": [],
            "synonyms": [
                "NEDCFSA",
                "NEURODEVELOPMENTAL DISORDER WITH COARSE FACIES AND MILD DISTAL SKELETAL ABNORMALITIES",
            ],
            "phenotype_labels": [],
        })
        queries = build_literature_queries(candidate)
        self.assertIn('"618505"[All Fields]', queries)
        self.assertIn('"NEDCFSA"', queries)
        self.assertIn(
            '"NEURODEVELOPMENTAL DISORDER WITH COARSE FACIES AND MILD DISTAL SKELETAL ABNORMALITIES"',
            queries,
        )
        self.assertLess(queries.index('"618505"[All Fields]'), queries.index('"STOLERMAN NEURODEVELOPMENTAL SYNDROME"'))

    def test_empty_literature_search_logs_queries_and_no_hit_outcome(self):
        output = io.StringIO()
        with patch("agent.tools.literature_search._request_json", return_value={"esearchresult": {"idlist": []}}):
            with redirect_stdout(output):
                sources = search_candidate_literature(self.candidate, session=Mock())
        self.assertEqual(sources, [])
        self.assertIn("LiteratureSearchDiagnostic", output.getvalue())
        self.assertIn("Example syndrome", output.getvalue())
        self.assertIn('"hits": 0', output.getvalue())

    def test_reflection_retrieval_status_is_run_scoped_and_not_a_negative_judgment(self):
        no_literature = create_reflection_v2({
            "mergedDiseaseCandidates": [self.candidate],
            "evidence": [],
            "sources": [],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "llm": None,
        })[0].ans[0]
        self.assertEqual(no_literature.literature_status, "not_retrieved")
        self.assertTrue(no_literature.correctness)

        source_without_extractable_evidence = create_reflection_v2({
            "mergedDiseaseCandidates": [self.candidate],
            "evidence": [],
            "sources": [self.source],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "llm": None,
        })[0].ans[0]
        self.assertEqual(source_without_extractable_evidence.literature_status, "insufficient")
        self.assertTrue(source_without_extractable_evidence.correctness)

    def test_reflection_fallback_does_not_default_every_candidate_to_true(self):
        generic_candidate = self.candidate.model_copy(update={
            "phenotype_labels": ["Seizures"],
        })
        result, _ = create_reflection_v2({
            "mergedDiseaseCandidates": [generic_candidate],
            "evidence": [],
            "sources": [],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "llm": None,
        })
        self.assertFalse(result.ans[0].correctness)

    def test_reflection_prompt_defines_retention_separately_from_retrieval(self):
        candidate = self.candidate.model_copy(update={
            "tool_supports": [CandidateSupport(tool="GestaltMatcher", rank=1, score=0.91)]
        })
        llm = Mock()
        llm.get_structured_llm.return_value.invoke.return_value = ReflectionOutputV2(ans=[])
        _reflection, prompt = create_reflection_v2({
            "mergedDiseaseCandidates": [candidate],
            "evidence": [],
            "sources": [],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {"HP:2": "Seizures"},
            "use_absentHPO": True,
            "llm": llm,
        })
        self.assertIn("Missing or weakly retrieved literature is uncertainty", prompt)
        self.assertIn("Do not mark every candidate true by default", prompt)
        self.assertIn("A candidate returned by a diagnostic tool is a hypothesis", prompt)
        self.assertIn("GestaltMatcher:rank=1", prompt)
        self.assertIn("PATIENT_ABSENT_FEATURES: Seizures", prompt)

    def test_retry_evidence_extraction_skips_already_resolved_candidates(self):
        unresolved = self.candidate.model_copy(update={"candidate_id": "OMIM:654321"})
        unresolved.OMIM_id = "OMIM:654321"
        llm = Mock()
        llm.ans = [
            ReflectionFormatV2(
                candidate_id=self.candidate.candidate_id,
                disease_name_en=self.candidate.disease_name,
                disease_name_ja=self.candidate.disease_name,
                correctness=True,
                patient_summary_en="Hypotonia.",
                patient_summary_ja="筋緊張低下。",
                diagnosis_analysis_en="Supported.",
                diagnosis_analysis_ja="支持あり。",
                literature_status="supported",
            ),
            ReflectionFormatV2(
                candidate_id=unresolved.candidate_id,
                disease_name_en=unresolved.disease_name,
                disease_name_ja=unresolved.disease_name,
                correctness=True,
                patient_summary_en="Hypotonia.",
                patient_summary_ja="筋緊張低下。",
                diagnosis_analysis_en="Insufficient.",
                diagnosis_analysis_ja="不十分。",
                literature_status="insufficient",
            ),
        ]
        state = {
            "flow_iteration": 1,
            "reflection": ReflectionOutputV2(ans=llm.ans),
            "mergedDiseaseCandidates": [self.candidate, unresolved],
            "sources": [],
            "evidence": [],
            "hpoDict": {},
            "absentHpoDict": {},
            "llm": Mock(),
        }
        with patch("agent.nodes.extract_evidence_for_candidates", return_value=[]) as extract:
            evidenceExtractionNode(state)
        selected = extract.call_args.kwargs["candidates"]
        self.assertEqual([item.candidate_id for item in selected], [unresolved.candidate_id])

    def test_chunk_selection_reaches_late_full_text(self):
        source = self.source.model_copy(update={
            "raw_content": (
                "Introduction\n\n" + ("General background without the target terms. " * 180)
                + "\n\nResults\n\nNIPBL variants were identified in patients with developmental delay."
            )
        })
        self.assertGreater(len(_source_chunks(source)), 1)
        selected = _select_relevant_chunks(self.candidate, [source], ["Global developmental delay"])
        self.assertLessEqual(len(selected), 10)
        self.assertTrue(any("NIPBL variants" in chunk.text for chunk in selected))

    def test_evidence_contains_only_claim_and_sourceable_excerpt(self):
        class Structured:
            def invoke(self, _messages):
                return EvidenceExtractionOutput(evidence=[EvidenceDraft(
                    source_id="SRC-1",
                    evidence_type="core_phenotype",
                    claim_en="The disease is associated with hypotonia.",
                    claim_ja="この疾患は筋緊張低下と関連します。",
                    excerpt="Example syndrome is associated with hypotonia",
                    support="supports",
                    confidence="high",
                )])

        class FakeLLM:
            def get_structured_llm(self, _model):
                return Structured()

        evidence = extract_evidence_for_candidates(
            [self.candidate], [self.source], {"HP:1": "Hypotonia"}, {}, FakeLLM()
        )
        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertEqual(item.excerpt, "Example syndrome is associated with hypotonia")
        self.assertFalse(hasattr(item, "impact"))
        self.assertFalse(hasattr(item, "relationship"))
        self.assertFalse(hasattr(item, "patient_matches"))
        self.assertFalse(hasattr(item, "limitations"))

    def test_validator_needs_only_literature_evidence(self):
        diagnosis = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="OMIM:123456",
            rank=1,
            description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
            description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
            evidence_ids=["EVD-1"],
            literature_status="supported",
        )])
        validated = validate_v2_output(diagnosis, [self.candidate], [self.source], [self.evidence])
        self.assertEqual(validated.validation_status, "passed")
        self.assertEqual(validated.references[0].identifier, "PMID:12345678")

    def test_public_response_has_simple_numbered_references(self):
        diagnosis = DiagnosisOutputV2(
            ans=[DiagnosisFormatV2(
                candidate_id="OMIM:123456",
                disease_name_en="Example syndrome",
                disease_name_ja="Example syndrome",
                OMIM_id="123456",
                rank=1,
                description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
                description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
                evidence_ids=["EVD-1"],
                literature_status="supported",
            )],
            evidence=[self.evidence],
        )
        result = serialize_public_state({
            "evidence_mode": "v2",
            "sources": [self.source],
            "evidence": [self.evidence],
            "tentativeDiagnosis": diagnosis,
            "reflection": None,
            "finalDiagnosis": diagnosis,
            "zeroShotRaw": {"ans": []},
            "zeroShotValidation": [{"status": "match"}],
            "tentativeRaw": diagnosis,
            "tentativeValidation": {"status": "accepted"},
        })
        item = result["finalDiagnosis"]["ans"][0]
        reference = result["finalDiagnosis"]["references"][0]
        self.assertEqual(item["citation_numbers"], [1])
        self.assertIn("[1]", item["description_en"])
        self.assertEqual(reference["number"], 1)
        self.assertEqual(reference["title"], self.source.title)
        self.assertEqual(reference["url"], self.source.url)
        self.assertEqual(reference["evidence"][0]["excerpt"], self.evidence.excerpt)
        self.assertEqual(
            set(reference),
            {
                "number",
                "source_name",
                "title",
                "identifier",
                "url",
                "authors",
                "journal",
                "publication_year",
                "evidence",
            },
        )
        self.assertEqual(
            set(reference["evidence"][0]),
            {"claim_en", "claim_ja", "excerpt"},
        )
        self.assertEqual(
            set(result["evidence"][0]),
            {"candidate_id", "claim_en", "claim_ja", "excerpt", "reference_number"},
        )
        self.assertNotIn("raw_content", result["sources"][0])
        self.assertNotIn("reasoning", item)
        self.assertNotIn("reference_catalog", result)
        self.assertNotIn("reference_records", result["finalDiagnosis"])
        self.assertNotIn("zeroShotRaw", result)
        self.assertNotIn("zeroShotValidation", result)
        self.assertNotIn("tentativeRaw", result)
        self.assertNotIn("tentativeValidation", result)

    def test_public_citation_numbers_are_global_and_private_fields_are_removed(self):
        second_source = self.source.model_copy(update={
            "source_id": "SRC-2",
            "title": "A second publication",
            "identifier": "PMID:87654321",
            "url": "https://pubmed.ncbi.nlm.nih.gov/87654321/",
        })
        second_evidence = self.evidence.model_copy(update={
            "evidence_id": "EVD-2",
            "source_id": "SRC-2",
            "excerpt": "A second independently traceable excerpt.",
        })
        tentative = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="123456",
            rank=1,
            description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-2]]",
            description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-2]]",
            evidence_ids=["EVD-2"],
            literature_status="supported",
        )])
        final = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="123456",
            rank=1,
            description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
            description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
            evidence_ids=["EVD-1"],
            literature_status="supported",
        )])
        result = serialize_public_state({
            "evidence_mode": "v2",
            "prompt": "private prompt",
            "imagePath": "/Users/private/patient.png",
            "image_path": "https://example.org/patient.png",
            "sources": [self.source, second_source],
            "evidence": [self.evidence, second_evidence],
            "tentativeDiagnosis": tentative,
            "reflection": {"ans": [{
                "candidate_id": "OMIM:123456",
                "diagnosis_analysis_en": "The literature supports the phenotype. [[EVIDENCE:EVD-2]]",
                "diagnosis_analysis_ja": "文献が表現型を支持します。[[EVIDENCE:EVD-2]]",
                "evidence_ids": ["EVD-2"],
            }]},
            "finalDiagnosis": final,
        })

        self.assertNotIn("prompt", result)
        self.assertNotIn("imagePath", result)
        self.assertEqual(result["image_path"], "https://example.org/patient.png")
        self.assertEqual([item["number"] for item in result["references"]], [1, 2])
        self.assertEqual(result["tentativeDiagnosis"]["ans"][0]["citation_numbers"], [2])
        self.assertEqual(result["tentativeDiagnosis"]["references"][0]["number"], 2)
        self.assertEqual(result["finalDiagnosis"]["ans"][0]["citation_numbers"], [1])
        self.assertEqual(result["finalDiagnosis"]["references"][0]["number"], 1)
        reflection_item = result["reflection"]["ans"][0]
        self.assertEqual(reflection_item["citation_numbers"], [2])
        self.assertIn("[2]", reflection_item["diagnosis_analysis_en"])

    def test_serializer_does_not_backfill_a_citation_to_unrelated_prose(self):
        result = _add_citations(
            "This candidate remains a broad clinical possibility.",
            [self.evidence.model_dump()],
            {self.source.source_id: 1},
            "en",
        )
        self.assertEqual(result, "This candidate remains a broad clinical possibility.")

    def test_external_compatibility_keeps_legacy_wrappers_and_structured_trace(self):
        diagnosis = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="123456",
            rank=1,
            description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
            description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
            evidence_ids=["EVD-1"],
            literature_status="supported",
        )])
        result = serialize_external_compatibility({
            "evidence_mode": "v2",
            "pubCaseFinder": [],
            "GestaltMatcher": [],
            "phenotypeSearchResult": [],
            "zeroShotResult": {"ans": []},
            "mergedDiseaseCandidates": [],
            "sources": [self.source],
            "evidence": [self.evidence],
            "tentativeDiagnosis": diagnosis,
            "reflection": None,
            "finalDiagnosis": diagnosis,
        })

        final = result["finalDiagnosis"]["finalDiagnosis"]
        self.assertEqual(final["ans"][0]["disease_name"], "Example syndrome")
        self.assertEqual(final["ans"][0]["candidate_id"], "OMIM:123456")
        self.assertEqual(final["ans"][0]["citation_numbers"], [1])
        self.assertEqual(final["references"][0]["number"], 1)
        self.assertEqual(
            set(final["references"][0]["evidence"][0]),
            {"claim_en", "claim_ja", "excerpt"},
        )
        self.assertEqual(result["external_response"]["finalDiagnosis"], result["finalDiagnosis"])

    def test_final_fallback_reconnects_evidence_and_inline_citation(self):
        diagnosis = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="OMIM:123456",
            rank=1,
            description_en="Short provisional description.",
            description_ja="短い暫定説明です。",
            evidence_ids=[],
            literature_status="not_retrieved",
        )])

        class FailingLLM:
            def get_structured_llm(self, _schema):
                raise AttributeError("DiagnosisFormatV2 has no attribute disease_name")

        final, _prompt = createFinalDiagnosisV2({
            "tentativeDiagnosis": diagnosis,
            "reflection": None,
            "evidence": [self.evidence],
            "sources": [self.source],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "llm": FailingLLM(),
        })
        item = final.ans[0]
        self.assertEqual(item.evidence_ids, ["EVD-1"])
        self.assertEqual(item.literature_status, "supported")
        self.assertIn("[[EVIDENCE:EVD-1]]", item.description_en)

        public = serialize_public_state({
            "evidence_mode": "v2",
            "sources": [self.source],
            "evidence": [self.evidence],
            "tentativeDiagnosis": diagnosis,
            "reflection": None,
            "finalDiagnosis": final,
        })
        public_item = public["finalDiagnosis"]["ans"][0]
        self.assertIn("[1]", public_item["description_en"])
        self.assertNotIn("[[EVIDENCE:", public_item["description_en"])
        self.assertEqual(public_item["citation_numbers"], [1])

    def test_final_evidence_contract_does_not_attach_unrelated_citations(self):
        item = DiagnosisFormatV2(
            candidate_id=self.candidate.candidate_id,
            disease_name_en=self.candidate.disease_name,
            disease_name_ja=self.candidate.disease_name,
            OMIM_id=self.candidate.OMIM_id,
            rank=1,
            description_en="The patient has a broad neurodevelopmental presentation.",
            description_ja="患者には幅広い神経発達上の所見があります。",
            evidence_ids=[self.evidence.evidence_id],
            literature_status="supported",
        )
        result = _apply_evidence_contract(
            item,
            [self.evidence],
            ["Hypotonia"],
            [self.source],
        )
        self.assertEqual(result.evidence_ids, [])
        self.assertEqual(result.literature_status, "insufficient")
        self.assertNotIn("[[EVIDENCE:", result.description_en)
        self.assertNotIn("[[EVIDENCE:", result.description_ja)

    def test_final_evidence_contract_pairs_exact_bilingual_claim_markers(self):
        item = DiagnosisFormatV2(
            candidate_id=self.candidate.candidate_id,
            disease_name_en=self.candidate.disease_name,
            disease_name_ja=self.candidate.disease_name,
            OMIM_id=self.candidate.OMIM_id,
            rank=1,
            description_en=self.evidence.claim_en,
            description_ja=self.evidence.claim_ja,
            evidence_ids=[self.evidence.evidence_id],
            literature_status="supported",
        )
        result = _apply_evidence_contract(
            item,
            [self.evidence],
            ["Hypotonia"],
            [self.source],
        )
        self.assertEqual(result.evidence_ids, [self.evidence.evidence_id])
        self.assertIn("[[EVIDENCE:EVD-1]]", result.description_en)
        self.assertIn("[[EVIDENCE:EVD-1]]", result.description_ja)

    def test_final_generation_validates_against_authoritative_candidate_model(self):
        diagnosis = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="OMIM:123456",
            rank=1,
            description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
            description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
            evidence_ids=[],
            literature_status="not_retrieved",
        )])

        generated = FinalDiagnosisGenerationOutput(ans=[DiagnosisFormatV2(
            candidate_id="OMIM:123456",
            disease_name_en="Example syndrome",
            disease_name_ja="Example syndrome",
            OMIM_id="OMIM:123456",
            rank=1,
            description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
            description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
            evidence_ids=["EVD-1"],
            literature_status="supported",
        )])

        class Structured:
            def invoke(self, _messages):
                return generated

        class FakeLLM:
            def get_structured_llm(self, _schema):
                return Structured()

        final, _prompt = createFinalDiagnosisV2({
            "tentativeDiagnosis": diagnosis,
            "mergedDiseaseCandidates": [self.candidate],
            "reflection": None,
            "evidence": [self.evidence],
            "sources": [self.source],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "llm": FakeLLM(),
        })

        self.assertEqual(final.ans[0].candidate_id, "OMIM:123456")
        self.assertEqual(final.ans[0].literature_status, "supported")
        self.assertEqual(final.validation_errors, [])
        self.assertIn("[[EVIDENCE:EVD-1]]", final.ans[0].description_en)

    def test_final_prompt_uses_phenotype_and_tool_support_not_literature_volume_for_rank(self):
        candidate = self.candidate.model_copy(update={
            "tool_supports": [CandidateSupport(tool="GestaltMatcher", rank=1, score=0.91)]
        })
        tentative = DiagnosisOutputV2(ans=[DiagnosisFormatV2(
            candidate_id=candidate.candidate_id,
            disease_name_en=candidate.disease_name,
            disease_name_ja=candidate.disease_name,
            OMIM_id=candidate.OMIM_id,
            rank=1,
            description_en="Candidate hypothesis.",
            description_ja="候補仮説です。",
            evidence_ids=[],
            literature_status="not_retrieved",
        )])
        reflection = ReflectionOutputV2(ans=[ReflectionFormatV2(
            candidate_id=candidate.candidate_id,
            disease_name_en=candidate.disease_name,
            disease_name_ja=candidate.disease_name,
            correctness=True,
            patient_summary_en="Hypotonia is present.",
            patient_summary_ja="筋緊張低下があります。",
            diagnosis_analysis_en="Plausible based on phenotype and tool support; no source was retrieved in this run.",
            diagnosis_analysis_ja="表現型とツール支持から候補に残ります。この実行では文献を取得できませんでした。",
            literature_status="not_retrieved",
        )])

        class Structured:
            def invoke(self, _messages):
                return FinalDiagnosisGenerationOutput(ans=[tentative.ans[0]])

        class FakeLLM:
            def get_structured_llm(self, _schema):
                return Structured()

        final, prompt = createFinalDiagnosisV2({
            "tentativeDiagnosis": tentative,
            "mergedDiseaseCandidates": [candidate],
            "reflection": reflection,
            "evidence": [],
            "sources": [],
            "hpoDict": {"HP:1": "Hypotonia"},
            "absentHpoDict": {},
            "use_absentHPO": False,
            "llm": FakeLLM(),
        })

        self.assertEqual(final.ans[0].candidate_id, candidate.candidate_id)
        self.assertIn("Literature retrieval success or the number of retrieved papers must not determine candidate rank", prompt)
        self.assertIn("correctness=true remains eligible", prompt)
        self.assertIn("GestaltMatcher:rank=1", prompt)
        self.assertIn("Hypotonia", prompt)

    def test_node_result_uses_same_simple_projection(self):
        node_result = serialize_node_result(
            "finalDiagnosisNode",
            {"finalDiagnosis": DiagnosisOutputV2(
                ans=[DiagnosisFormatV2(
                    candidate_id="OMIM:123456",
                    disease_name_en="Example syndrome",
                    disease_name_ja="Example syndrome",
                    OMIM_id="123456",
                    rank=1,
                    description_en="The disease is associated with hypotonia. [[EVIDENCE:EVD-1]]",
                    description_ja="この疾患は筋緊張低下と関連します。[[EVIDENCE:EVD-1]]",
                    evidence_ids=["EVD-1"],
                    literature_status="supported",
                )],
                evidence=[self.evidence],
            )},
            {"evidence_mode": "v2", "sources": [self.source], "evidence": [self.evidence]},
        )
        payload = node_result["result"]["finalDiagnosis"]
        self.assertEqual(payload["ans"][0]["citation_numbers"], [1])
        self.assertEqual(payload["references"][0]["number"], 1)

    def test_v2_default_and_original_node_order_are_preserved(self):
        self.assertEqual(inspect.signature(RareDiseaseDiagnosisPipeline.run).parameters["evidence_mode"].default, "v2")
        source = inspect.getsource(RareDiseaseDiagnosisPipeline._build_graph)
        self.assertLess(source.find('("candidateMetadataNode", "createDiagnosisNode")'), source.find('("diseaseNormalizeNode", "literatureSearchNode")'))
        self.assertLess(source.find('("literatureSearchNode", "evidenceExtractionNode")'), source.find('("evidenceExtractionNode", "reflectionNode")'))
        self.assertLess(source.find('("evidenceExtractionNode", "reflectionNode")'), source.find('("finalDiagnosisNode", "diseaseNormalizeForFinalNode")'))
