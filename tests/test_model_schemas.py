import pytest

from pydantic import ValidationError

from persuasion_bias.schemas.models import BiasAnalysis, BiasDetection, CialdiniPrinciple, ArgumentClassification


def test_cialdini_principle_values():
    assert CialdiniPrinciple.RECIPROCITY.value == "reciprocity"
    assert CialdiniPrinciple.COMMITMENT.value == "commitment"
    assert CialdiniPrinciple.SOCIAL_PROOF.value == "social_proof"
    assert CialdiniPrinciple.AUTHORITY.value == "authority"
    assert CialdiniPrinciple.LIKING.value == "liking"
    assert CialdiniPrinciple.SCARCITY.value == "scarcity"


def test_bias_detection_defaults():
    bd = BiasDetection(principle=CialdiniPrinciple.AUTHORITY, severity="low")
    assert bd.confidence == 0.0
    assert bd.evidence == ""


def test_bias_detection_valid():
    bd = BiasDetection(
        principle=CialdiniPrinciple.SCARCITY, severity="high", confidence=0.9, evidence="Limited time offer"
    )
    assert bd.principle == CialdiniPrinciple.SCARCITY
    assert bd.severity == "high"


def test_bias_detection_invalid_severity():
    with pytest.raises(ValidationError):
        BiasDetection(principle=CialdiniPrinciple.LIKING, severity="extreme")


def test_bias_detection_invalid_principle():
    with pytest.raises(ValidationError):
        BiasDetection(principle="not_a_principle", severity="low")


def test_argument_classification_true():
    assert ArgumentClassification(is_argument=True).is_argument is True


def test_argument_classification_false():
    assert ArgumentClassification(is_argument=False).is_argument is False


def test_argument_classification_missing_field():
    with pytest.raises(ValidationError):
        ArgumentClassification()


@pytest.fixture
def valid_bias_analysis():
    return {
        "detected_principles": [
            BiasDetection(
                principle=CialdiniPrinciple.AUTHORITY,
                severity="high",
            )
        ],
        "overall_bias_score": 0.8,
        "emotional_manipulation_score": 0.5,
        "target_audience_analysis": "Targets young professionals",
    }


def test_bias_analysis_valid(valid_bias_analysis):
    ba = BiasAnalysis(**valid_bias_analysis)
    assert ba.overall_bias_score == 0.8  # noqa: PLR2004
    assert ba.logical_fallacies == []
    assert ba.credibility_issues == []


def test_bias_analysis_score_out_of_range(valid_bias_analysis):
    valid_bias_analysis["overall_bias_score"] = 0.0  # gt=0 so 0 is invalid
    with pytest.raises(ValidationError):
        BiasAnalysis(**valid_bias_analysis)


def test_bias_analysis_score_above_one(valid_bias_analysis):
    valid_bias_analysis["overall_bias_score"] = 1.5
    with pytest.raises(ValidationError):
        BiasAnalysis(**valid_bias_analysis)


def test_bias_analysis_missing_required_fields():
    with pytest.raises(ValidationError):
        BiasAnalysis()
