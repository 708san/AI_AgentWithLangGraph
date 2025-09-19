from typing_extensions import List, TypedDict, Optional
from pydantic import BaseModel, Field

class PCFres(TypedDict):
    omim_disease_name_en: str
    description: str
    score: Optional[float]
    omim_id: str

class HistoryItem(TypedDict):
    role: str  # "user" or "agent" or "tool"
    content: str

class State(TypedDict):
    depth: int
    imagePath: Optional[str]
    clinicalText: Optional[str]
    hpoList: List[str]
    hpoDict: dict[str, str]
    absentHpoList: List[str]
    absentHpoDict: dict[str, str] 
    pubCaseFinder: List[PCFres]
    GestaltMatcher: List['GestaltMatcherFormat']
    webresources: List['webresource']
    # evidence are stored in memory
    memory: List['InformationItem']
    zeroShotResult: Optional['ZeroShotOutput']
    tentativeDiagnosis: Optional['DiagnosisOutput']
    reflection: Optional['ReflectionOutput']
    finalDiagnosis: Optional['DiagnosisOutput']
    
# --- Pydantic Model for Zero-Shot Diagnosis Output ---
class ZeroShotFormat(BaseModel):
    disease_name: str = Field(..., description="The formal name of the most likely rare disease, based solely on the patient's HPO terms.")
    rank: int = Field(..., description="The rank of the disease in the differential diagnosis list, where 1 is the most likely.")

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