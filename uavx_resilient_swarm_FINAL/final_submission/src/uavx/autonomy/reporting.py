from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReportRecord:
    task_id: str
    uav_id: int
    discovered_at_s: float
    transmitted_at_s: Optional[float] = None
    delivered_at_s: Optional[float] = None
    packets_sent: int = 0
    packets_delivered: int = 0
    latency_ms: float = 0.0
    status: str = "PENDING"

    @property
    def pdr(self) -> float:
        if self.packets_sent <= 0:
            return 0.0
        return self.packets_delivered / self.packets_sent

    @property
    def report_success(self) -> bool:
        return self.delivered_at_s is not None


class ReportingEngine:
    def __init__(self, deadline_s: float) -> None:
        if deadline_s <= 0:
            raise ValueError("deadline_s must be > 0")
        self.deadline_s = float(deadline_s)
        self.records: dict[str, ReportRecord] = {}

    def begin(self, task_id: str, uav_id: int, discovered_at_s: float) -> ReportRecord:
        record = ReportRecord(
            task_id=task_id,
            uav_id=uav_id,
            discovered_at_s=discovered_at_s,
        )
        self.records[task_id] = record
        return record

    def transmit(
        self,
        task_id: str,
        now_s: float,
        success: bool,
        latency_ms: float,
    ) -> ReportRecord:
        record = self.records[task_id]
        record.transmitted_at_s = now_s
        record.packets_sent += 1
        record.latency_ms = latency_ms

        if success:
            record.packets_delivered += 1
            record.delivered_at_s = now_s
            record.status = (
                "ON_TIME"
                if now_s - record.discovered_at_s <= self.deadline_s
                else "LATE"
            )
        return record
