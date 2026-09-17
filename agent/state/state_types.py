from typing import Annotated
from typing_extensions import List, TypedDict, Optional, Literal
import operator
from pydantic import BaseModel, Field
from ..llm.llm_wrapper import AzureOpenAIWrapper

class PCFres(TypedDict):
    omim_disease_name_en: str
    description: str
    score: Optional[float]
    omim_id: str

class PhenoBrainResult(TypedDict, total=False):
    disease_name: str
    omim_id: Optional[str]
    orpha_id: Optional[str]
    source_codes: List[str]
    rd_id: str
    rank: int
    score: Optional[float]

class HistoryItem(TypedDict):
    role: str  # "user" or "agent" or "tool"
    content: str

class ToolRankingItem(TypedDict, total=False):
    tool: str
    rank: int
    score: Optional[float]
    matched_hpo_id: str
    note: str

class MergedDiseaseCandidate(TypedDict):
    disease_name: str
    OMIM_id: Optional[str]
    consensus_count: int
    best_rank: int
    tool_rankings: List[ToolRankingItem]


class ToolResponseRecord(TypedDict, total=False):
    """取得したツールレスポンスを、表示用Top5とは別に保持する記録。"""
    tool: str
    request: dict
    response: object
    top5: list
    retrieved_at: str
    rank_status: str
    all_results: list
    raw_response_ref: str
    raw_response_hash: str
    raw_response_ref: str
    source_ids: List[str]
    request_context: dict
    run_id: str
    response_status: str
    completeness: str
    elapsed_ms: float


class SourceRecord(TypedDict, total=False):
    """実際にデータを取得した情報源のメタデータ。"""
    source_id: str
    access_layer: str
    tool: str
    database: str
    source_type: str
    entry_id: str
    url: str
    query: str
    request: dict
    retrieved_at: str
    raw_response: object
    raw_response_hash: str
    uri: str
    endpoint: str
    content_hash: str
    status: str


class CandidateLink(TypedDict, total=False):
    candidate_id: str
    candidate_label: str
    relation: str
    polarity: str
    reason: str


class EvidenceRecord(TypedDict, total=False):
    """ある情報源が、どの候補疾患に何を意味するかを表す根拠。"""
    evidence_id: str
    source_id: str
    candidate_links: List[CandidateLink]
    claim: dict
    content: dict
    polarity: str
    assertion: str
    retrieved_at: str
    source_locator: dict
    candidate_id: str
    source_ids: List[str]
    relation: str
    excerpt: str
    structured_value: dict
    extraction_method: str
    created_at: str


class DiscoveredDiseaseCandidate(TypedDict, total=False):
    candidate_id: str
    disease_name: str
    discovered_by: dict
    discovery_evidence_ids: List[str]
    status: str


class CandidateAdmissionItem(BaseModel):
    candidate_id: str = Field(..., description="Stable disease identifier when available.")
    disease_name: str = Field(..., description="Disease name returned by the source.")
    include: bool = Field(..., description="Whether to add this disease to the verification candidate pool.")
    reason: str = Field(..., description="Evidence-grounded reason for the decision.")


class CandidateAdmissionOutput(BaseModel):
    candidates: List[CandidateAdmissionItem] = Field(default_factory=list)


class TogoMCPCallRecord(TypedDict, total=False):
    tool_name: str
    arguments: dict
    result: dict
    started_at: str
    elapsed_ms: float
    status: str
    error: str


class TogoMCPSearchPlan(TypedDict, total=False):
    candidate_id: str
    candidate_name: str
    objective: str
    missing_information: List[str]
    selected_actions: List[str]
    need_more_search: bool
    reason: str


# ---------------------------------------------------------------------------
# ZebraSeek fixed-flow data contracts
# ---------------------------------------------------------------------------

class PatientInput(TypedDict, total=False):
    """Canonical input.  ``clinical_text`` is intentionally absent."""
    patient_id: str
    present_hpo_ids: List[str]
    absent_hpo_ids: List[str]
    sex: str
    onset: str
    image_path: Optional[str]


class RankedResult(TypedDict, total=False):
    candidate_id: str
    disease_name: str
    omim_id: Optional[str]
    orpha_id: Optional[str]
    rank: int
    score: Optional[float]
    tool: str
    raw: dict


class DiseaseRankingIndexItem(TypedDict, total=False):
    disease_key: str
    normalized_ids: List[str]
    rankings: List[dict]
    candidate_id: str
    disease_name: str
    identifiers: dict
    by_tool: dict[str, RankedResult]
    top5_tools: List[str]
    initial_candidate: bool
    discovered_candidate: bool
    discovery_evidence_ids: List[str]


class CandidateRecord(TypedDict, total=False):
    """Same shape for initial and one-hop discovered candidates."""
    candidate_id: str
    disease_name: str
    identifiers: dict
    discovery: dict
    ranking: dict
    searches: List[str]
    reflection_id: Optional[str]
    gene_annotation_ids: List[str]


class CandidateSearchSession(TypedDict, total=False):
    candidate_id: str
    candidate_name: str
    hop: int
    fixed_actions: List[str]
    action_bindings: dict
    tool_call_ids: List[str]
    evidence_ids: List[str]
    status: str


class ActionBinding(TypedDict, total=False):
    action: str
    tool_name: Optional[str]
    status: str
    reason: str
    matched_by: str
    input_schema: dict


class ToolCallRecord(TypedDict, total=False):
    call_id: str
    tool_call_id: str
    action: str
    candidate_id: Optional[str]
    tool_name: Optional[str]
    arguments: dict
    result: dict
    source_id: str
    source_ids: List[str]
    result_ref: str
    database: str
    query: str
    started_at: str
    elapsed_ms: float
    status: str
    error: str


class ReflectionAssessment(TypedDict, total=False):
    reflection_id: str
    candidate_id: str
    disease_name: str
    judgment: Literal["correct", "incorrect", "uncertain"]
    rationale: str
    supporting_evidence_ids: List[str]
    contradicting_evidence_ids: List[str]
    unknown_evidence_ids: List[str]
    source_ids: List[str]
    created_at: str
    patient_summary: str
    analysis: str
    model: str
    prompt_ref: str


class GeneAnnotationRecord(TypedDict, total=False):
    annotation_id: str
    gene_annotation_id: str
    candidate_id: str
    disease_candidate_id: str
    disease_id: str
    gene_symbol: str
    gene_id: Optional[str]
    relation: str
    evidence_ids: List[str]
    source_ids: List[str]
    status: str
    retrieved_at: str


class FinalCandidateResult(TypedDict, total=False):
    candidate_id: str
    disease_name: str
    identifiers: dict
    rank: int
    ranking_score: float
    ranking_mode: str
    reflection: ReflectionAssessment
    supporting_evidence: List[dict]
    contradicting_evidence: List[dict]
    genes: List[GeneAnnotationRecord]
    normalized_ids: List[str]
    judgment: str
    unknown_evidence_ids: List[str]
    known_causal_gene_annotation_ids: List[str]
    supporting_evidence_ids: List[str]
    contradicting_evidence_ids: List[str]
    rationale: str


class ExecutionMetadata(TypedDict, total=False):
    run_id: str
    started_at: str
    finished_at: str
    ranking_mode: str
    config: dict
    errors: List[dict]


class ZebraSeekState(TypedDict, total=False):
    patient_input: PatientInput
    initial_tool_responses: List[ToolResponseRecord]
    normalization_records: List[dict]
    normalization_records: List[dict]
    disease_ranking_index: dict[str, DiseaseRankingIndexItem]
    candidate_pool: List[CandidateRecord]
    source_records: List[SourceRecord]
    evidence_records: List[EvidenceRecord]
    search_sessions: List[CandidateSearchSession]
    reflection_assessments: List[ReflectionAssessment]
    final_ranking: List[FinalCandidateResult]
    gene_annotations: List[GeneAnnotationRecord]
    ranking_mode: str
    execution_metadata: ExecutionMetadata
    prompt_records: List[dict]
    procedure_trace: dict


class ZebraSeekOutput(TypedDict, total=False):
    patient_id: str
    ranked_candidates: List[FinalCandidateResult]
    source_records: List[SourceRecord]
    evidence_records: List[EvidenceRecord]
    tool_response_records: List[ToolResponseRecord]
    normalization_records: List[dict]
    reflection_assessments: List[ReflectionAssessment]
    gene_annotations: List[GeneAnnotationRecord]
    execution_metadata: dict
    prompt_records: List[dict]
    procedure_trace: dict


class ReflectionAssessmentOutput(BaseModel):
    candidate_id: str
    judgment: Literal["correct", "incorrect", "uncertain"]
    rationale: str = ""
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    unknown_evidence_ids: List[str] = Field(default_factory=list)


class LLMRankingItem(BaseModel):
    candidate_id: str
    rank: int
    rationale: str = ""
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)


class LLMRankingOutput(BaseModel):
    candidates: List[LLMRankingItem] = Field(default_factory=list)

class State(TypedDict):
    depth: int
    # Legacy graph compatibility field.  The canonical ``patient_input``
    # deliberately omits clinical text; this field is not populated by the
    # new ZebraSeek route.
    clinicalText: Optional[str]
    imagePath: Optional[str]
    hpoList: List[str]
    hpoDict: dict[str, str]
    absentHpoList: List[str]
    absentHpoDict: dict[str, str] 
    use_absentHPO: bool
    use_phenobrain: bool
    filter_impotance: bool
    pubCaseFinder: List[PCFres]
    phenoBrain: List[PhenoBrainResult]
    GestaltMatcher: List['GestaltMatcherFormat']
    phenotypeSearchResult: Optional[List['PhenotypeSearchFormat']]
    mergedDiseaseCandidates: List[MergedDiseaseCandidate]
    webresources: List['webresource']
    # evidence are stored in memory
    memory: List['InformationItem']
    zeroShotResult: Optional['ZeroShotOutput']
    tentativeDiagnosis: Optional['DiagnosisOutput']
    reflection: Optional['ReflectionOutput']
    finalDiagnosis: Optional['DiagnosisOutput']
    use_togomcp: bool
    tool_response_records: Annotated[List[ToolResponseRecord], operator.add]
    source_records: List[SourceRecord]
    evidence_records: List[EvidenceRecord]
    discovered_disease_candidates: List[DiscoveredDiseaseCandidate]
    candidate_admission_decisions: List[CandidateAdmissionItem]
    togomcp_call_history: List[TogoMCPCallRecord]
    togomcp_search_plans: List[TogoMCPSearchPlan]
    research_call_count: int
    togomcp_search_status: Optional[str]
    onset: Optional[str]
    sex: Optional[str]
    patient_id: Optional[str]
    llm: Optional[AzureOpenAIWrapper]
    # Canonical ZebraSeek state retained alongside the historical fields.
    patient_input: PatientInput
    initial_tool_responses: List[ToolResponseRecord]
    disease_ranking_index: dict
    candidate_pool: List[CandidateRecord]
    candidate_records: List[CandidateRecord]
    search_sessions: List[CandidateSearchSession]
    candidate_search_sessions: List[CandidateSearchSession]
    action_bindings: dict
    tool_call_records: List[ToolCallRecord]
    prompt_records: List[dict]
    procedure_trace: dict
    reflection_assessments: List[ReflectionAssessment]
    ranked_candidates: List[FinalCandidateResult]
    final_ranking: List[FinalCandidateResult]
    final_candidates: List[FinalCandidateResult]
    gene_annotations: List[GeneAnnotationRecord]
    execution_metadata: ExecutionMetadata
    ranking_mode: str
    zebraseek_depth: int
    togo_client: object
    zebraseek_output: dict
    zebraseek_context: dict
    final_output: dict
    
# --- Pydantic Model for Zero-Shot Diagnosis Output ---
class ZeroShotFormat(BaseModel):
    disease_name: str = Field(..., description="The formal name of the most likely rare disease, based solely on the patient's HPO terms.")
    rank: int = Field(..., description="The rank of the disease in the differential diagnosis list, where 1 is the most likely.")
    OMIM_id: Optional[str] = Field(None, description="The OMIM identifier for the disease, if available.")

class ZeroShotOutput(BaseModel):
    ans: List[ZeroShotFormat]


# --- Pydantic Models for Tentative Diagnosis Output ---
class DiagnosisFormat(BaseModel):
    disease_name: str = Field(..., description="The formal name of the most likely rare disease, derived from synthesizing multiple data sources (HPO, PubCaseFinder, ZeroShot, GestaltMatcher).")
    OMIM_id: Optional[str] = Field(None, description="The OMIM identifier for the disease, if available.")
    description: str = Field(..., description="The diagnostic reasoning explaining why this diagnosis is clinically plausible. Must specify which of the patient's symptoms support this diagnosis and include in-text citations [1], [2] to the evidence sources.")
    rank: int = Field(..., description="The final rank of the disease in the differential diagnosis list, where 1 is the most likely.")

class DiagnosisOutput(BaseModel):
    ans: list['DiagnosisFormat']
    reference: Optional[str] = Field(None, description="A numbered list of all sources cited in the 'description' field. Each entry must include the source type, a summary of its content, and a URL if available.")

# --- Pydantic Models for Self-Reflection Output ---
class ReflectionFormat(BaseModel):
    disease_name: str = Field(..., description="The name of the diagnosis being evaluated.")
    Correctness: bool = Field(..., description="A professional judgment on whether this diagnosis is clinically correct (True) or incorrect (False) for the patient, based on the provided medical literature.")
    PatientSummary: str = Field(..., description="A  about three-sentence summary of the patient's most critical clinical features, which forms the basis for the diagnostic evaluation.")
    DiagnosisAnalysis: str = Field(..., description="A detailed analysis of why the diagnosis was judged as correct or incorrect. This must be supported by logically connecting the patient's symptoms with direct evidence from the provided medical literature, using in-text citations [1], [2].")
    references: List[str] = Field(
        ...,
        description="A numbered list of direct quotes extracted from the provided medical literature that support the analysis. Do not list URLs; extract the specific sentences. Example: [\"1. 'Cohen syndrome is characterized by truncal obesity.'\", \"2. 'Neutropenia is a frequent finding.'\"]"
    )

class ReflectionOutput(BaseModel):
    ans: List['ReflectionFormat'] = Field(..., description="A list of evaluation results, with each item in the list corresponding to a single tentative diagnosis that was reviewed.")


#---Pydantic Model for  GM---

class GestaltMatcherFormat(BaseModel):
    subject_id: str
    syndrome_name: str
    omim_id: str
    image_id: str
    score: float
    

#---TypedDict ---
class InformationItem(TypedDict):
    title: str
    url: str
    content: str
    disease_name: str

class webresource(TypedDict):
    title: str
    url: str
    snippet: str

# --- Pydantic Model for Phenotype-based Embedding Search Output (新規追加) ---
class OMIMEntry(BaseModel):
    OMIM_id: str
    disease_name: str
    synonym: Optional[str] = None
    definition: Optional[str] = None
    phenotype: Optional[List[str]] = None


class PhenotypeSearchFormat(BaseModel):
    disease_info: OMIMEntry = Field(..., description="Information about the disease from the OMIM database.")
    similarity_score: float = Field(..., description="Cosine similarity score with the patient's phenotypes.")
