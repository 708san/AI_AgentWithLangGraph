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
    clinicalTest: Optional[str]
    hpoList: List[str]
    hpoDict: dict[str, str]
    pubCaseFinder: List[PCFres]
    history: List[HistoryItem]
    zeroShotResult: Optional['ZeroShotOutput']
    finalDiagnosis: Optional['DiagnosisOutput']

class DiagnosisFormat(BaseModel):
    disease_name: str = Field(..., description="The name of the disease.")
    description: str = Field(..., description="A brief description of the disease.")
    rank: int = Field(..., description="The rank of this disease among the candidates.")

class DiagnosisOutput(BaseModel):
    ans: list['DiagnosisFormat']
    reference: Optional[str] = Field(None, description="Reference information or URL you used to make diagnosis.")

class ZeroShotFormat(BaseModel):
    disease_name: str = Field(..., description="The name of the disease.")
    rank: int = Field(..., description="The rank of this disease among the candidates.")
    
class ZeroShotOutput(BaseModel):
    ans: List[ZeroShotFormat]