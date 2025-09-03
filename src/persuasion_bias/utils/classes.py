from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class CialdiniPrinciple(Enum):
    RECIPROCITY = "reciprocity"
    COMMITMENT = "commitment"
    SOCIAL_PROOF = "social_proof"
    AUTHORITY = "authority"
    LIKING = "liking"
    SCARCITY = "scarcity"


@dataclass
class BiasDetection:
    principle: CialdiniPrinciple
    severity: Literal["low", "mid", "high"]
    confidence: float = field(default=0.0)
    evidence: str = field(default="")


@dataclass
class BiasAnalysis:
    detected_principles: list[BiasDetection]
    overall_bias_score: float
    logical_fallacies: list[str]
    emotional_manipulation_score: float
    credibility_issues: list[str]
    target_audience_analysis: str
