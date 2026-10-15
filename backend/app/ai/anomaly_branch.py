"""Internal callable boundary; evidence generation remains separate from QC."""

from typing import Protocol, runtime_checkable

from backend.app.core.inspection_contracts import DenseAnomalyEvidence, InspectionContext, InspectionInput


@runtime_checkable
class AnomalyDetector(Protocol):
    def inspect(self, inspection: InspectionInput, context: InspectionContext) -> DenseAnomalyEvidence:
        ...
