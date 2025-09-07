from typing import Literal
from enum import Enum
from pydantic import BaseModel, Field


class CialdiniPrinciple(Enum):
    RECIPROCITY = "reciprocity"
    COMMITMENT = "commitment"
    SOCIAL_PROOF = "social_proof"
    AUTHORITY = "authority"
    LIKING = "liking"
    SCARCITY = "scarcity"


class BiasDetection(BaseModel):
    principle: CialdiniPrinciple
    severity: Literal["low", "mid", "high"]
    confidence: float = Field(default=0.0)
    evidence: str = Field(default="")


class BiasAnalysis(BaseModel):
    detected_principles: list[BiasDetection]
    overall_bias_score: float = Field(..., gt=0, le=1)
    logical_fallacies: list[str] = Field(default_factory=list)
    emotional_manipulation_score: float = Field(..., gt=0, le=1)
    credibility_issues: list[str] = Field(default_factory=list)
    target_audience_analysis: str = Field(description="Description of targeting strategy")
