from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import exp, log10
from typing import Mapping

import numpy as np

from uavx.core.config import CommunicationConfig
from uavx.core.models import (
    CommunicationLink,
    LinkStatus,
    NetworkGraph,
)


# Reserved node ID for the Ground Control Station.
# UAV IDs are non-negative in the current implementation.
GCS_NODE_ID = -1


@dataclass(frozen=True)
class LinkModelParameters:
    """
    Engineering parameters for the non-binary communication models.

    These are intentionally kept outside the official YAML configuration
    during the initial implementation. They will be promoted to configuration
    fields after the communication interface is validated.
    """

    distance_decay_exponent: float = 2.0

    # Log-distance model parameters.
    reference_distance_m: float = 1.0
    reference_path_loss_db: float = 40.0
    path_loss_exponent: float = 2.2
    tx_power_dbm: float = 20.0
    receiver_threshold_dbm: float = -70.0
    shadowing_sigma_db: float = 3.0
    transition_width_db: float = 4.0


class CommunicationModel(ABC):
    """Base interface for all communication models."""

    def __init__(
        self,
        config: CommunicationConfig,
        rng: np.random.Generator,
        parameters: LinkModelParameters | None = None,
    ) -> None:
        self.config = config
        self.rng = rng
        self.parameters = parameters or LinkModelParameters()

    @abstractmethod
    def compute(
        self,
        source_id: int,
        target_id: int,
        distance_m: float,
    ) -> CommunicationLink:
        """Compute a single link from source to target."""

    def _down_link(
        self,
        source_id: int,
        target_id: int,
        distance_m: float,
    ) -> CommunicationLink:
        return CommunicationLink(
            source_id=source_id,
            target_id=target_id,
            distance_m=distance_m,
            link_quality=0.0,
            packet_delivery_probability=0.0,
            latency_ms=self.config.latency_ms,
            availability=False,
            status=LinkStatus.DOWN,
        )

    @staticmethod
    def _status(
        availability: bool,
        packet_delivery_probability: float,
    ) -> LinkStatus:
        if not availability or packet_delivery_probability <= 0.0:
            return LinkStatus.DOWN
        if packet_delivery_probability < 0.95:
            return LinkStatus.DEGRADED
        return LinkStatus.UP


class BinaryRangeModel(CommunicationModel):
    """
    Model A: ideal binary communication range.

    A link exists when distance <= configured maximum range.
    """

    def compute(
        self,
        source_id: int,
        target_id: int,
        distance_m: float,
    ) -> CommunicationLink:
        if distance_m > self.config.max_range_m:
            return self._down_link(source_id, target_id, distance_m)

        pdr = self.config.packet_delivery_probability

        return CommunicationLink(
            source_id=source_id,
            target_id=target_id,
            distance_m=distance_m,
            link_quality=pdr,
            packet_delivery_probability=pdr,
            latency_ms=self.config.latency_ms,
            availability=pdr > 0.0,
            status=self._status(pdr > 0.0, pdr),
        )


class DistanceProbabilisticModel(CommunicationModel):
    """
    Model B: distance-dependent probabilistic communication.

    PDR decreases continuously with normalized distance:

        PDR = base_PDR * (1 - (d / R)^k)

    for 0 <= d <= R.

    The formula is an engineering baseline rather than a claim about
    a particular radio technology.
    """

    def compute(
        self,
        source_id: int,
        target_id: int,
        distance_m: float,
    ) -> CommunicationLink:
        if distance_m > self.config.max_range_m:
            return self._down_link(source_id, target_id, distance_m)

        if distance_m <= 0.0:
            pdr = self.config.packet_delivery_probability
        else:
            normalized_distance = distance_m / self.config.max_range_m
            exponent = self.parameters.distance_decay_exponent

            pdr = self.config.packet_delivery_probability * max(
                0.0,
                1.0 - normalized_distance**exponent,
            )

        pdr = float(np.clip(pdr, 0.0, 1.0))
        availability = pdr > 0.0

        return CommunicationLink(
            source_id=source_id,
            target_id=target_id,
            distance_m=distance_m,
            link_quality=pdr,
            packet_delivery_probability=pdr,
            latency_ms=self.config.latency_ms,
            availability=availability,
            status=self._status(availability, pdr),
        )


class LogDistancePathLossModel(CommunicationModel):
    """
    Model C: log-distance path-loss model with stochastic shadowing.

    Path loss:

        PL(d) = PL(d0) + 10*n*log10(d/d0) + X_sigma

    Received power is compared with a receiver threshold and converted
    to a smooth PDR using a logistic transition.

    The configured maximum communication range remains a hard boundary.
    """

    def compute(
        self,
        source_id: int,
        target_id: int,
        distance_m: float,
    ) -> CommunicationLink:
        if distance_m > self.config.max_range_m:
            return self._down_link(source_id, target_id, distance_m)

        if distance_m <= 0.0:
            distance_m = self.parameters.reference_distance_m

        p = self.parameters

        if distance_m <= p.reference_distance_m:
            path_loss_db = p.reference_path_loss_db
        else:
            shadowing_db = float(
                self.rng.normal(
                    loc=0.0,
                    scale=p.shadowing_sigma_db,
                )
            )

            path_loss_db = (
                p.reference_path_loss_db
                + 10.0
                * p.path_loss_exponent
                * log10(distance_m / p.reference_distance_m)
                + shadowing_db
            )

        received_power_dbm = p.tx_power_dbm - path_loss_db
        margin_db = received_power_dbm - p.receiver_threshold_dbm

        width = max(p.transition_width_db, 1e-6)

        try:
            pdr = 1.0 / (1.0 + exp(-margin_db / width))
        except OverflowError:
            pdr = 0.0 if margin_db < 0.0 else 1.0

        # Preserve the official hard boundary.
        pdr = float(np.clip(pdr, 0.0, 1.0))

        availability = pdr > 0.0

        return CommunicationLink(
            source_id=source_id,
            target_id=target_id,
            distance_m=distance_m,
            link_quality=pdr,
            packet_delivery_probability=pdr,
            latency_ms=self.config.latency_ms,
            availability=availability,
            status=self._status(availability, pdr),
        )


_MODEL_TYPES: dict[str, type[CommunicationModel]] = {
    "binary_range": BinaryRangeModel,
    "distance_probabilistic": DistanceProbabilisticModel,
    "log_distance": LogDistancePathLossModel,
    "path_loss": LogDistancePathLossModel,
}


class CommunicationEngine:
    """
    Constructs the dynamic communication graph G(t) = (V(t), E(t)).
    """

    def __init__(
        self,
        config: CommunicationConfig,
        seed: int,
        parameters: LinkModelParameters | None = None,
    ) -> None:
        if config.model not in _MODEL_TYPES:
            supported = ", ".join(sorted(_MODEL_TYPES))
            raise ValueError(
                f"Unknown communication model '{config.model}'. "
                f"Supported models: {supported}"
            )

        self.config = config
        self.rng = np.random.default_rng(seed)
        self.parameters = parameters or LinkModelParameters()

        model_class = _MODEL_TYPES[config.model]

        self.model = model_class(
            config=config,
            rng=self.rng,
            parameters=self.parameters,
        )

    def compute_link(
        self,
        source_id: int,
        target_id: int,
        source_position: np.ndarray,
        target_position: np.ndarray,
    ) -> CommunicationLink:
        source = np.asarray(source_position, dtype=float)
        target = np.asarray(target_position, dtype=float)

        if source.shape != (3,) or target.shape != (3,):
            raise ValueError("Positions must have shape (3,)")

        distance_m = float(np.linalg.norm(source - target))

        return self.model.compute(
            source_id=source_id,
            target_id=target_id,
            distance_m=distance_m,
        )

    def build_graph(
        self,
        positions: Mapping[int, np.ndarray],
        timestamp_s: float,
        gcs_position: np.ndarray,
    ) -> NetworkGraph:
        """
        Build an undirected graph containing all UAVs plus the GCS.

        GCS_NODE_ID (-1) is reserved for the GCS.
        """

        if GCS_NODE_ID in positions:
            raise ValueError(
                f"{GCS_NODE_ID} is reserved for the GCS"
            )

        gcs = np.asarray(gcs_position, dtype=float)

        if gcs.shape != (3,):
            raise ValueError("gcs_position must have shape (3,)")

        all_positions: dict[int, np.ndarray] = dict(positions)
        all_positions[GCS_NODE_ID] = gcs

        nodes = sorted(all_positions)
        edges: set[tuple[int, int]] = set()
        links: dict[tuple[int, int], CommunicationLink] = {}

        for index, source_id in enumerate(nodes):
            for target_id in nodes[index + 1 :]:
                link = self.compute_link(
                    source_id=source_id,
                    target_id=target_id,
                    source_position=all_positions[source_id],
                    target_position=all_positions[target_id],
                )

                key = (source_id, target_id)
                links[key] = link

                if link.availability and link.status != LinkStatus.DOWN:
                    edges.add(key)

        graph = NetworkGraph(
            timestamp_s=timestamp_s,
            nodes=nodes,
            edges=edges,
            links=links,
        )
        graph.update_laplacian()

        return graph
