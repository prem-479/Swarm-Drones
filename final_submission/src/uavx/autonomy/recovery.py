from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureClassification(str, Enum):
    PACKET_LOSS = "PACKET_LOSS"
    LINK_DEGRADED = "LINK_DEGRADED"
    COMMUNICATION_OUTAGE = "COMMUNICATION_OUTAGE"
    UAV_FAILURE = "UAV_FAILURE"


@dataclass
class FailureTrack:
    uav_id: int
    last_heartbeat_s: float
    missed_heartbeats: int = 0
    confidence: float = 0.0
    classification: FailureClassification | None = None
    detected_at_s: float | None = None
    recovered_at_s: float | None = None


class FailureDetector:
    def __init__(
        self,
        heartbeat_timeout_s: float = 10.0,
        failure_threshold: int = 3,
    ) -> None:
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.failure_threshold = failure_threshold
        self.tracks: dict[int, FailureTrack] = {}

    def observe(self, uav_id: int, now_s: float) -> None:
        track = self.tracks.setdefault(
            uav_id,
            FailureTrack(uav_id=uav_id, last_heartbeat_s=now_s),
        )
        track.last_heartbeat_s = now_s
        track.missed_heartbeats = 0
        track.confidence = 0.0
        track.classification = None

    def update(self, now_s: float, alive_ids: set[int]) -> list[FailureTrack]:
        detected = []

        for uav_id in list(self.tracks):
            track = self.tracks[uav_id]

            if uav_id in alive_ids:
                continue

            elapsed = now_s - track.last_heartbeat_s

            if elapsed < self.heartbeat_timeout_s:
                continue

            track.missed_heartbeats += 1
            track.confidence = min(
                1.0,
                track.missed_heartbeats / max(self.failure_threshold, 1),
            )

            if track.missed_heartbeats >= self.failure_threshold:
                track.classification = FailureClassification.UAV_FAILURE
                if track.detected_at_s is None:
                    track.detected_at_s = now_s
                    detected.append(track)

        return detected


@dataclass(frozen=True)
class RecoveryAction:
    failed_uav_id: int
    replacement_uav_id: int | None
    reason: str


class RecoveryCoordinator:
    def select_replacement(
        self,
        failed_uav_id: int,
        candidate_ids: list[int],
    ) -> RecoveryAction:
        replacement = min(candidate_ids) if candidate_ids else None

        return RecoveryAction(
            failed_uav_id=failed_uav_id,
            replacement_uav_id=replacement,
            reason=(
                "lowest-id healthy reserve selected"
                if replacement is not None
                else "no healthy reserve available"
            ),
        )

# Compatibility assessment API used by the advanced evaluation layer.
def _uavx_recovery_assess(self, evidence):
    from types import SimpleNamespace
    from uavx.autonomy.faults import FaultClassification

    elapsed_s = float(evidence.time_since_heartbeat_s)
    pdr = float(evidence.packet_delivery_ratio)
    link_quality = float(evidence.link_quality)
    heartbeat_confidence = float(evidence.heartbeat_confidence)
    missed_heartbeats = int(evidence.missed_heartbeats)

    # Confirm a UAV failure only when multiple indicators agree.
    if (
        elapsed_s >= 5.0
        and pdr <= 0.05
        and link_quality <= 0.05
        and heartbeat_confidence <= 0.50
    ):
        return SimpleNamespace(
            classification=FaultClassification.UAV_FAILURE,
            confidence=1.0,
            confirmed=True,
        )

    # A short observation window with healthy heartbeat evidence is
    # classified as packet loss rather than UAV failure.
    confidence = max(
        0.0,
        min(
            1.0,
            0.60 * (1.0 - pdr)
            + 0.30 * (1.0 - link_quality)
            + 0.10 * min(missed_heartbeats / 5.0, 1.0),
        ),
    )

    return SimpleNamespace(
        classification=FaultClassification.PACKET_LOSS,
        confidence=confidence,
        confirmed=False,
    )


FailureDetector.assess = _uavx_recovery_assess
