"""Opt-in request telemetry sessions for external benchmark orchestrators."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .utils.system_sampler import SystemSampler


class TelemetrySessionRequest(BaseModel):
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    sample_interval_seconds: float = Field(default=0.1, gt=0)


@dataclass
class _TelemetrySession:
    session_id: str
    metadata: dict[str, Any]
    sampler: SystemSampler
    started_at: float
    started_monotonic: float
    requests: list[dict[str, Any]] = field(default_factory=list)
    finished_at: float | None = None
    finished_monotonic: float | None = None


class RequestTelemetryCapture:
    """Mutable capture for one request within a telemetry session."""

    def __init__(
        self,
        manager: TelemetryManager,
        session: _TelemetrySession,
        request_id: str,
        model_id: str,
        started_at: float,
    ) -> None:
        self._manager = manager
        self._session = session
        self._started_at = started_at
        self._request_id = request_id
        self._model_id = model_id
        self._token_samples: list[dict[str, Any]] = []
        self._finished = False

    def observe_output(self, output: Any) -> None:
        """Record cumulative token counters at an engine output boundary."""
        observed_at = time.monotonic()
        generated_at = getattr(output, "generated_at", None)
        self._token_samples.append(
            {
                "elapsed_seconds": observed_at - self._started_at,
                "generated_elapsed_seconds": (
                    float(generated_at) - self._started_at
                    if generated_at is not None
                    else None
                ),
                "completion_tokens": int(getattr(output, "completion_tokens", 0)),
            }
        )

    def finish(
        self,
        usage: dict[str, Any],
        *,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Commit this request capture to its session exactly once."""
        if self._finished:
            return
        self._finished = True
        self._manager._finish_request(
            self,
            usage,
            time.monotonic(),
            error=error,
        )


class TelemetryManager:
    """Manage opt-in telemetry sessions without affecting untagged requests."""

    def __init__(
        self,
        sampler_factory: Callable[[float], SystemSampler] | None = None,
    ) -> None:
        self._sampler_factory = sampler_factory or (
            lambda interval_seconds: SystemSampler(interval_s=interval_seconds)
        )
        self._sessions: dict[str, _TelemetrySession] = {}
        self._lock = threading.Lock()

    def start_session(
        self,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sample_interval_seconds: float = 0.1,
    ) -> dict[str, Any]:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than zero")
        resolved_session_id = session_id or uuid.uuid4().hex
        sampler = self._sampler_factory(sample_interval_seconds)
        started_at = time.time()
        started_monotonic = time.monotonic()
        session = _TelemetrySession(
            session_id=resolved_session_id,
            metadata=deepcopy(metadata or {}),
            sampler=sampler,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        with self._lock:
            if resolved_session_id in self._sessions:
                raise ValueError(
                    f"Telemetry session already exists: {resolved_session_id}"
                )
            self._sessions[resolved_session_id] = session
        sampler.start()
        return self._session_payload(session)

    def begin_request(
        self,
        session_id: str | None,
        *,
        request_id: str,
        model_id: str,
    ) -> RequestTelemetryCapture | None:
        if session_id is None:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.finished_at is not None:
                return None
        return RequestTelemetryCapture(
            self,
            session,
            request_id,
            model_id,
            time.monotonic(),
        )

    def stop_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.finished_at is None:
                session.finished_at = time.time()
                session.finished_monotonic = time.monotonic()
        session.sampler.stop()
        return self._session_payload(session)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return self._session_payload(session)

    def is_session_active(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            return session is not None and session.finished_at is None

    def _finish_request(
        self,
        capture: RequestTelemetryCapture,
        usage: dict[str, Any],
        finished_at: float,
        *,
        error: dict[str, Any] | None,
    ) -> None:
        session = capture._session
        record = {
            "request_id": capture._request_id,
            "model_id": capture._model_id,
            "started_elapsed_seconds": (
                capture._started_at - session.started_monotonic
            ),
            "duration_seconds": finished_at - capture._started_at,
            "status": "failed" if error is not None else "completed",
            "usage": deepcopy(usage),
            "error": deepcopy(error),
            "token_samples": deepcopy(capture._token_samples),
            "host_samples": session.sampler.raw_window(
                capture._started_at,
                finished_at,
                origin=capture._started_at,
            ),
        }
        with self._lock:
            session.requests.append(record)

    def _session_payload(self, session: _TelemetrySession) -> dict[str, Any]:
        finished_monotonic = session.finished_monotonic
        return {
            "session_id": session.session_id,
            "metadata": deepcopy(session.metadata),
            "started_at": session.started_at,
            "finished_at": session.finished_at,
            "duration_seconds": (
                finished_monotonic - session.started_monotonic
                if finished_monotonic is not None
                else None
            ),
            "requests": deepcopy(session.requests),
        }


telemetry_manager = TelemetryManager()
