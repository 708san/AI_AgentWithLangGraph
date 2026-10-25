from typing import Any, Literal, Union
from typing_extensions import List, TypedDict, Optional
from pydantic import BaseModel, Field
from ..llm.llm_wrapper import AzureOpenAIWrapper

class PCFres(TypedDict, total=False):
    omim_disease_name_en: str
    description: str
    score: Optional[float]
    omim_id: str
    rank: int

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


class CandidateSupport(BaseModel):
    """A diagnostic-tool signal, deliberately kept separate from citations."""

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


class MergedDiseaseCandidateV2(BaseModel):
    candidate_id: str
    disease_name: str
    OMIM_id: Optional[str] = None
    gene_symbols: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    phenotype_labels: List[str] = Field(default_factory=list)
    consensus_count: int = 0
    best_rank: int = 9999
    tool_supports: List[CandidateSupport] = Field(default_factory=list)


class SourceRecord(BaseModel):
    source_id: str
    source_type: Literal["literature", "database", "diagnostic_tool", "web_search"]
    source_name: str
    title: str
    url: Optional[str] = None
    identifier: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_year: Optional[str] = None
    candidate_ids: List[str] = Field(default_factory=list)
    disease_name: Optional[str] = None
    OMIM_id: Optional[str] = None
    gene_symbols: List[str] = Field(default_factory=list)
    content_type: Literal[
        "abstract", "full_text", "database_record", "tool_result", "search_snippet"
    ]
    raw_content: str = ""
    search_query: Optional[str] = None
    retrieved_at: Optional[str] = None
    citation_allowed: bool = False


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
    source_chunk_id: Optional[str] = None
    source_section: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    reference_number: Optional[int] = None


class EvidenceAssessment(BaseModel):
    evidence_id: str
    relevance: Literal["high", "medium", "low"]
    supports_patient: Literal["yes", "no", "unknown"]
    reason_en: str
    reason_ja: str


class DiagnosisFormatV2(BaseModel):
    candidate_id: str
    disease_name_en: str
    disease_name_ja: str
    OMIM_id: Optional[str] = None
    rank: int
    description_en: str
    description_ja: str
    evidence_ids: List[str] = Field(default_factory=list)
    literature_status: Literal[
        "supported", "insufficient", "contradictory", "not_retrieved"
    ]


class ReferenceRecord(SourceRecord):
    evidence_ids: List[str] = Field(default_factory=list)


class DiagnosisOutputV2(BaseModel):
    ans: List[DiagnosisFormatV2] = Field(default_factory=list)
    references: List[ReferenceRecord] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    validation_status: Literal["passed", "failed", "partial"] = "passed"
    validation_errors: List[str] = Field(default_factory=list)
    reference: Optional[str] = None


class FinalDiagnosisGenerationOutput(BaseModel):
    """LLM-only schema for final prose generation.

    References and Evidence are authoritative State data, not fields the
    final-diagnosis model should regenerate. Keeping this schema narrow also
    prevents the structured-output parser from mixing source records with
    diagnosis items.
    """

    ans: List[DiagnosisFormatV2] = Field(default_factory=list)


class ReflectionFormatV2(BaseModel):
    candidate_id: str
    disease_name_en: str
    disease_name_ja: str
    correctness: bool
    patient_summary_en: str
    patient_summary_ja: str
    diagnosis_analysis_en: str
    diagnosis_analysis_ja: str
    evidence_assessments: List[EvidenceAssessment] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    literature_status: Literal[
        "supported", "insufficient", "contradictory", "not_retrieved"
    ]


class ReflectionOutputV2(BaseModel):
    ans: List[ReflectionFormatV2] = Field(default_factory=list)


class EvidenceDraft(BaseModel):
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
    source_chunk_id: Optional[str] = None
    source_section: Optional[str] = None


class EvidenceExtractionOutput(BaseModel):
    evidence: List[EvidenceDraft] = Field(default_factory=list)

class State(TypedDict):
    depth: int
    evidence_mode: Literal["legacy", "v2"]
    imagePath: Optional[str]
    clinicalText: Optional[str]
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
    mergedDiseaseCandidates: List[Union[MergedDiseaseCandidate, MergedDiseaseCandidateV2]]
    webresources: List['webresource']
    # evidence are stored in memory
    memory: List['InformationItem']
    sources: List[SourceRecord]
    evidence: List[EvidenceItem]
    validation_status: Literal["not_run", "passed", "failed", "partial"]
    validation_errors: List[str]
    flow_iteration: int
    max_flow_iterations: int
    zeroShotResult: Optional['ZeroShotOutput']
    zeroShotRaw: Optional['ZeroShotOutput']
    zeroShotValidation: Optional[List[dict[str, Any]]]
    tentativeRaw: Optional[Union['DiagnosisOutput', DiagnosisOutputV2]]
    tentativeValidation: Optional[dict[str, Any]]
    tentativeDiagnosis: Optional[Union['DiagnosisOutput', DiagnosisOutputV2]]
    reflection: Optional[Union['ReflectionOutput', ReflectionOutputV2]]
    finalDiagnosis: Optional[Union['DiagnosisOutput', DiagnosisOutputV2]]
    onset: Optional[str]
    sex: Optional[str]
    patient_id: Optional[str]
    llm: Optional[AzureOpenAIWrapper]
    
# --- Pydantic Model for Zero-Shot Diagnosis Output ---
class ZeroShotFormat(BaseModel):
    disease_name: str = Field(..., description=(
        "The disease name in OMIM entry-title format, preserving capitalization, punctuation, "
        "and subtype numbers. Use plain text without explanations or Markdown."
    ))
    rank: int = Field(..., description="The rank of the disease in the differential diagnosis list, where 1 is the most likely.")
    OMIM_id: Optional[str] = Field(None, description=(
        "The disease's OMIM identifier as a six-digit string without the OMIM prefix. "
        "Use JSON null when it is unknown; never invent a disease or gene identifier."
    ))

class ZeroShotOutput(BaseModel):
    ans: List[ZeroShotFormat]


class ZeroShotReasonedCandidate(ZeroShotFormat):
    selection_reason: str = Field(..., min_length=1, description=(
        "A concise explanation, at most two sentences, of why this candidate was selected. "
        "Use supplied patient findings only; mention an explicit contradiction or uncertainty "
        "when relevant. This field is for logging and evaluation only."
    ))


class ZeroShotReasonedOutput(BaseModel):
    """LLM-only output; selection_reason must never enter downstream State."""

    ans: List[ZeroShotReasonedCandidate]


# --- Pydantic Models for Tentative Diagnosis Output ---
class DiagnosisFormat(BaseModel):
    disease_name: str = Field(..., description="The formal name of the most likely rare disease, derived from synthesizing multiple data sources (HPO, PubCaseFinder, ZeroShot, GestaltMatcher).")
    OMIM_id: Optional[str] = Field(None, description="The OMIM identifier for the disease, if available.")
    description: str = Field(..., description="The diagnostic reasoning explaining why this diagnosis is clinically plausible. Must specify which of the patient's symptoms support this diagnosis and include in-text citations [1], [2] to the evidence sources.")
    rank: int = Field(..., description="The final rank of the disease in the differential diagnosis list, where 1 is the most likely.")
    reference: List[str] = Field(
        default_factory=list,
        description="References cited by this diagnosis description. Each item should include the source and URL when available.",
    )

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
        description="A numbered list of direct quotes extracted from the provided medical literature that support the analysis. Include the source name and URL when available. Example: [\"[PubMed] (https://pubmed.ncbi.nlm.nih.gov/12345678/): 'Neutropenia is a frequent finding.'\"]"
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
    rank: Optional[int] = None
    

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
    rank: Optional[int] = Field(None, description="The rank returned by the phenotype similarity search.")
