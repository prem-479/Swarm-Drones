from .controller import DecisionStats, SwarmDecisionEngine
from .allocation import (
    AllocationCandidate,
    AllocationRecord,
    AllocationWeights,
    CommunicationAwareAllocator,
    PreemptionDecision,
    PriorityTaskInserter,
)
from .energy import (
    EnergyEstimate,
    EnergyManager,
    HandoverState,
    RelayHandoverManager,
)
from .faults import (
    FaultAssessment,
    FaultClassification,
    FaultEvidence,
    FailureDetector,
    RecoveryAction,
    RecoveryCandidate,
    RecoveryPlanner,
)
from .motion import (
    AsymmetricCollisionAvoider,
    CommunicationPreservingPlanner,
    GeofenceGuard,
    GlobalPlanner,
    MotionCandidate,
    MotionLimits,
)
from .scenario import ScenarioDefinition, ScenarioEvent, ScenarioGenerator

__all__ = [
    "AllocationCandidate",
    "AllocationRecord",
    "AllocationWeights",
    "CommunicationAwareAllocator",
    "PreemptionDecision",
    "PriorityTaskInserter",
    "EnergyEstimate",
    "EnergyManager",
    "HandoverState",
    "RelayHandoverManager",
    "FaultAssessment",
    "FaultClassification",
    "FaultEvidence",
    "FailureDetector",
    "RecoveryAction",
    "RecoveryCandidate",
    "RecoveryPlanner",
    "AsymmetricCollisionAvoider",
    "CommunicationPreservingPlanner",
    "GeofenceGuard",
    "GlobalPlanner",
    "MotionCandidate",
    "MotionLimits",
    "ScenarioDefinition",
    "ScenarioEvent",
    "ScenarioGenerator",
    "DecisionStats",
    "SwarmDecisionEngine",
]

from .backbone import PredictiveRelayBackbone, RelayWaypoint

from .reporting import ReportingEngine, ReportRecord

from .recovery import FailureClassification, FailureDetector, RecoveryAction, RecoveryCoordinator

from .handover import EnergyPredictor, HandoverPhase, PriorityInsertion, PriorityTaskManager, RelayHandover

from .faults import FaultClassification, FaultEvidence, FailureAssessment

try:
    from .runtime_patch import install as _install_runtime_patch
    _install_runtime_patch()
except Exception:
    # Keep package importable during low-level isolated imports.
    pass
