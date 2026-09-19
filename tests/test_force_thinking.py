"""Tests for the force_thinking request option."""

from types import SimpleNamespace

from omlx.adapter.output_parser import (
    GEMMA4_FORCED_THINKING_PREFIX,
    append_forced_thinking_prefix,
    forced_thinking_prefix_for_model,
)
from omlx.api.openai_models import ChatCompletionRequest, Message, ModelInfo
from omlx.engine.batched import BatchedEngine


def test_forced_thinking_prefix_is_model_specific():
    assert (
        forced_thinking_prefix_for_model("gemma-4-12B-it-8bit")
        == GEMMA4_FORCED_THINKING_PREFIX
    )
    assert (
        forced_thinking_prefix_for_model("some-model", {"model_type": "gemma4_unified"})
        == GEMMA4_FORCED_THINKING_PREFIX
    )
    assert forced_thinking_prefix_for_model("qwen3.5-0.8b") is None
    assert forced_thinking_prefix_for_model("llama-3.2-1b-instruct") is None


def test_append_forced_thinking_prefix_is_idempotent_and_opt_in():
    prompt = "<|turn>model\n"
    forced = append_forced_thinking_prefix(prompt, "gemma-4-12B-it-8bit")

    assert forced == prompt + GEMMA4_FORCED_THINKING_PREFIX
    assert append_forced_thinking_prefix(forced, "gemma-4-12B-it-8bit") == forced
    assert append_forced_thinking_prefix(prompt, "qwen3.5-0.8b") == prompt


def test_append_forced_thinking_prefix_prefers_parser_metadata():
    assert (
        append_forced_thinking_prefix(
            "prompt", "unknown-model", forced_thinking_prefix="<|think|>"
        )
        == "prompt<|think|>"
    )


def test_request_defaults_to_no_forced_thinking():
    request = ChatCompletionRequest(
        model="gemma-4-12B-it-8bit",
        messages=[Message(role="user", content="hi")],
    )
    assert request.force_thinking is False

    forced = ChatCompletionRequest(
        model="gemma-4-12B-it-8bit",
        messages=[Message(role="user", content="hi")],
        force_thinking=True,
    )
    assert forced.force_thinking is True


def test_model_info_advertises_support():
    assert ModelInfo(id="m").force_thinking is False
    assert ModelInfo(id="m", force_thinking=True).force_thinking is True


def test_batched_engine_appends_prefix_only_when_forced():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model_name = "gemma-4-12B-it-8bit"
    engine._engine = None
    rendered = []

    def _render(messages, tools=None, chat_template_kwargs=None, is_partial=None):
        rendered.append(messages)
        return "<|turn>model\n"

    engine._render_chat_template = _render

    assert engine._apply_chat_template([{"role": "user", "content": "hi"}]) == (
        "<|turn>model\n"
    )
    assert engine._apply_chat_template(
        [{"role": "user", "content": "hi"}], force_thinking=True
    ) == ("<|turn>model\n" + GEMMA4_FORCED_THINKING_PREFIX)
    assert len(rendered) == 2


def test_forced_thinking_prefix_reads_loaded_parser_factory():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._engine = SimpleNamespace(
        engine=SimpleNamespace(
            scheduler=SimpleNamespace(
                _output_parser_factory=SimpleNamespace(
                    forced_thinking_prefix="<|channel>thought\n"
                )
            )
        )
    )

    assert engine._forced_thinking_prefix() == "<|channel>thought\n"


def test_minimum_thinking_tokens_block_an_immediate_close():
    import mlx.core as mx

    from omlx.api.thinking import ThinkingBudgetProcessor

    close_id = 7
    processor = ThinkingBudgetProcessor(
        think_end_token_ids=[close_id],
        budget=1000,
        minimum_tokens=3,
    )
    prompt_tokens = [1, 2]
    logits = mx.zeros((1, 16))

    # While below the floor the close marker is unreachable.
    for step in range(3):
        masked = processor(prompt_tokens + [9] * step, logits)
        assert masked[..., close_id].item() == float("-inf")

    # Past the floor the model may close the channel again.
    allowed = processor(prompt_tokens + [9, 9, 9], logits)
    assert allowed[..., close_id].item() == 0.0


def test_minimum_thinking_tokens_default_to_disabled():
    import mlx.core as mx

    from omlx.api.thinking import ThinkingBudgetProcessor

    processor = ThinkingBudgetProcessor(think_end_token_ids=[7], budget=1000)
    logits = mx.zeros((1, 16))

    assert processor([1, 2], logits)[..., 7].item() == 0.0


def test_request_min_thinking_tokens_is_optional():
    request = ChatCompletionRequest(
        model="gemma-4-12B-it-8bit",
        messages=[Message(role="user", content="hi")],
        force_thinking=True,
        min_thinking_tokens=8,
    )
    assert request.min_thinking_tokens == 8
