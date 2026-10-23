"""Small compatibility helpers for evidence-grounded outputs.

The clinical explanation is written by the diagnosis/reflection models. The
State only keeps the Evidence IDs needed to verify the explanation; it does
not maintain a second graph of patient-feature relationships.
"""

from __future__ import annotations

from typing import Iterable

from ..state.state_types import (
    DiagnosisOutputV2,
    EvidenceItem,
    MergedDiseaseCandidateV2,
    ReflectionOutputV2,
    SourceRecord,
)


def attach_diagnosis_reasoning(
    output: DiagnosisOutputV2 | None,
    candidates: Iterable[MergedDiseaseCandidateV2],
    sources: Iterable[SourceRecord],
    evidence: Iterable[EvidenceItem],
) -> DiagnosisOutputV2 | None:
    """Keep model-generated explanations without adding duplicate metadata."""
    return output


def attach_reflection_reasoning(
    output: ReflectionOutputV2 | None,
    candidates: Iterable[MergedDiseaseCandidateV2],
    sources: Iterable[SourceRecord],
    evidence: Iterable[EvidenceItem],
) -> ReflectionOutputV2 | None:
    """Keep model-generated reflection without a parallel relation structure."""
    return output
