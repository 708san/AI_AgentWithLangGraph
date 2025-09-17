from typing_extensions import List, TypedDict, Optional
from pydantic import BaseModel, Field

class PCFres(TypedDict):
    omim_disease_name_en: str
    description: str
    score: Optional[float]

class HistoryItem(TypedDict):
    role: str  # "user" or "agent" or "tool"
    content: str

class State(TypedDict):
    depth: int
    imagePath: Optional[str]
    clinicalText: Optional[str]
    hpoList: List[str]
    hpoDict: dict[str, str]
    pubCaseFinder: List[PCFres]
    GestaltMatcher: List['GestaltMatcherFormat']
    # evidence are stored in memory
    memory: List['InformationItem']
    zeroShotResult: Optional['ZeroShotOutput']
    tentativeDiagnosis: Optional['DiagnosisOutput']
    reflection: Optional['ReflectionOutput']
    finalDiagnosis: Optional['DiagnosisOutput']
    


class ReflectionOutput(BaseModel):
    ans: List['ReflectionFormat']

class ReflectionFormat(BaseModel):
    disease_name: str = Field(..., description="The name of the disease.")
    Correctness: bool = Field(..., description="Whether the diagnosis is correct or not.")
    PatientSummary: str = Field(..., description="A brief summary of the patient's key symptoms.")
    DiagnosisAnalysis: str = Field(..., description="Analysis of the diagnosis, including reasoning and evidence.")
    reference: str = Field(..., description="Reference information or URL you used to make diagnosis.")

class DiagnosisOutput(BaseModel):
    ans: list['DiagnosisFormat']
    reference: Optional[str] = Field(None, description="Reference information or URL you used to make diagnosis.")

class DiagnosisFormat(BaseModel):
    disease_name: str = Field(..., description="The name of the disease.")
    description: str = Field(..., description="A brief description and reason of the diagnosis.")
    rank: int = Field(..., description="The rank of this disease among the candidates.")

class ZeroShotFormat(BaseModel):
    disease_name: str = Field(..., description="The name of the disease.")
    rank: int = Field(..., description="The rank of this disease among the candidates.")
    
class ZeroShotOutput(BaseModel):
    ans: List[ZeroShotFormat]

class GestaltMatcherFormat(BaseModel):
    gene_name: str
    gene_entrez_id: str
    distance: float
    gestalt_score: float
    image_id: str
    subject_id: str

class InformationItem(TypedDict):
    title: str
    url: str
    content: str
    disease_name: str