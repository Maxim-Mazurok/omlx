import time
from types import SimpleNamespace

import pytest

from omlx import server
from omlx.api.openai_models import ChatCompletionRequest
from omlx.telemetry import TelemetryManager, TelemetrySessionRequest


class FakeSampler:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def raw_window(self, started_at, finished_at, *, origin=None):
        return [{"elapsed_seconds": started_at - (origin or started_at)}]


@pytest.fixture
def telemetry_manager_and_samplers():
    samplers = []

    def sampler_factory(interval_seconds):
        sampler = FakeSampler(interval_seconds)
        samplers.append(sampler)
        return sampler

    return TelemetryManager(sampler_factory=sampler_factory), samplers


def test_session_captures_correlated_request(telemetry_manager_and_samplers):
    telemetry_manager, samplers = telemetry_manager_and_samplers
    telemetry_manager.start_session(
        session_id="benchmark-run",
        metadata={"benchmark": "human-eval"},
        sample_interval_seconds=0.2,
    )

    capture = telemetry_manager.begin_request(
        "benchmark-run",
        request_id="task-1",
        model_id="test-model",
    )
    assert capture is not None
    capture.observe_output(SimpleNamespace(completion_tokens=3, generated_at=None))
    capture.finish({"prompt_tokens": 10, "completion_tokens": 3})
    result = telemetry_manager.stop_session("benchmark-run")

    assert result is not None
    assert samplers[0].interval_seconds == 0.2
    assert samplers[0].started is True
    assert samplers[0].stopped is True
    assert result["metadata"] == {"benchmark": "human-eval"}
    assert result["requests"][0]["request_id"] == "task-1"
    assert result["requests"][0]["status"] == "completed"
    assert result["requests"][0]["error"] is None
    assert result["requests"][0]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
    }
    assert result["requests"][0]["token_samples"][0]["completion_tokens"] == 3


def test_untagged_request_is_not_captured(telemetry_manager_and_samplers):
    telemetry_manager, _ = telemetry_manager_and_samplers
    telemetry_manager.start_session(session_id="benchmark-run")

    capture = telemetry_manager.begin_request(
        None,
        request_id="ordinary-request",
        model_id="test-model",
    )

    assert capture is None
    assert telemetry_manager.get_session("benchmark-run")["requests"] == []


def test_unknown_or_stopped_session_is_not_captured(
    telemetry_manager_and_samplers,
):
    telemetry_manager, _ = telemetry_manager_and_samplers
    telemetry_manager.start_session(session_id="benchmark-run")
    telemetry_manager.stop_session("benchmark-run")

    assert (
        telemetry_manager.begin_request(
            "missing-session",
            request_id="task-1",
            model_id="test-model",
        )
        is None
    )
    assert (
        telemetry_manager.begin_request(
            "benchmark-run",
            request_id="task-2",
            model_id="test-model",
        )
        is None
    )


def test_duplicate_session_id_is_rejected(telemetry_manager_and_samplers):
    telemetry_manager, _ = telemetry_manager_and_samplers
    telemetry_manager.start_session(session_id="benchmark-run")

    with pytest.raises(ValueError, match="already exists"):
        telemetry_manager.start_session(session_id="benchmark-run")


def test_failed_request_is_captured_once(telemetry_manager_and_samplers):
    telemetry_manager, _ = telemetry_manager_and_samplers
    telemetry_manager.start_session(session_id="benchmark-run")
    capture = telemetry_manager.begin_request(
        "benchmark-run",
        request_id="task-1",
        model_id="test-model",
    )
    assert capture is not None

    capture.finish({}, error={"message": "generation failed"})
    capture.finish({"completion_tokens": 1})
    result = telemetry_manager.stop_session("benchmark-run")

    assert result is not None
    assert len(result["requests"]) == 1
    assert result["requests"][0]["status"] == "failed"
    assert result["requests"][0]["error"] == {"message": "generation failed"}


@pytest.mark.asyncio
async def test_telemetry_route_functions_manage_session(
    telemetry_manager_and_samplers,
    monkeypatch,
):
    telemetry_manager, samplers = telemetry_manager_and_samplers
    monkeypatch.setattr(server, "telemetry_manager", telemetry_manager)
    request = TelemetrySessionRequest(
        session_id="benchmark-run",
        metadata={"benchmark": "human-eval"},
        sample_interval_seconds=0.2,
    )

    started = await server.start_telemetry_session(request, True)
    fetched = await server.get_telemetry_session("benchmark-run", True)
    stopped = await server.stop_telemetry_session("benchmark-run", True)

    assert started["session_id"] == "benchmark-run"
    assert fetched["metadata"] == {"benchmark": "human-eval"}
    assert stopped["finished_at"] is not None
    assert samplers[0].stopped is True


@pytest.mark.asyncio
async def test_stream_chat_completion_captures_tagged_request(
    telemetry_manager_and_samplers,
    monkeypatch,
):
    telemetry_manager, _ = telemetry_manager_and_samplers
    telemetry_manager.start_session(session_id="benchmark-run")
    monkeypatch.setattr(server, "telemetry_manager", telemetry_manager)

    class FakeEngine:
        tokenizer = None
        is_diffusion_model = False

        async def stream_chat(self, messages, **arguments):
            observed_at = time.perf_counter()
            yield SimpleNamespace(
                cached_tokens=0,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                first_token_at=observed_at,
                generated_at=observed_at,
                new_text="answer",
                prompt_tokens=2,
                tool_calls=None,
            )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "question"}],
        stream=True,
    )
    chunks = [
        chunk
        async for chunk in server.stream_chat_completion(
            FakeEngine(),
            request.messages,
            request,
            telemetry_session_id="benchmark-run",
            telemetry_request_id="task-1",
        )
    ]
    result = telemetry_manager.stop_session("benchmark-run")

    assert result is not None
    assert chunks[-1] == "data: [DONE]\n\n"
    assert result["requests"][0]["request_id"] == "task-1"
    assert result["requests"][0]["usage"]["prompt_tokens"] == 2
    assert result["requests"][0]["token_samples"][0]["completion_tokens"] == 1
