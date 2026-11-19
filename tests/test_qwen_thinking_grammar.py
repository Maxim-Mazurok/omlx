# SPDX-License-Identifier: Apache-2.0
"""Qwen thinking and schema checks without model loading or inference."""

from types import SimpleNamespace

import pytest
import xgrammar

from omlx.api.openai_models import ResponseFormat, StructuredOutputOptions
from omlx.server import _compile_grammar_for_request, _grammar_reasoning_parser

MODEL_TYPES = [
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
]
VOCABULARY = [bytes([i]) for i in range(256)] + [
    b"</think>",
    b"<think>",
    b"</s>",
]
THOUGHT_END = 256
THOUGHT_START = 257
TURN_END = 258
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"const": "yes"}},
    "required": ["answer"],
    "additionalProperties": False,
}
ANSWER = b'{"answer":"yes"}'


@pytest.fixture(scope="module")
def compiler():
    return xgrammar.GrammarCompiler(
        xgrammar.TokenizerInfo(
            VOCABULARY, xgrammar.VocabType.RAW, stop_token_ids=[TURN_END]
        )
    )


@pytest.fixture(params=["response_format", "structured_outputs"])
def request_format(request):
    if request.param == "response_format":
        return {
            "response_format": ResponseFormat(
                type="json_schema",
                json_schema={"name": "answer", "schema": ANSWER_SCHEMA, "strict": True},
            )
        }
    return {"structured_outputs": StructuredOutputOptions(json=ANSWER_SCHEMA)}


def compile_request(compiler, request_format, *, enable_thinking):
    engine = SimpleNamespace(model_type="qwen3_5", grammar_compiler=compiler)
    chat_template_kwargs = {"enable_thinking": enable_thinking}
    reasoning_parser = _grammar_reasoning_parser(engine, None, chat_template_kwargs)
    compiled = _compile_grammar_for_request(
        engine,
        **request_format,
        chat_template_kwargs=chat_template_kwargs,
        reasoning_parser=reasoning_parser,
    )
    assert compiled is not None
    return compiled


def accept_tokens(matcher, tokens):
    for token in tokens:
        assert matcher.accept_token(token), VOCABULARY[token]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_auto_selection_preserves_explicit_parser(model_type):
    engine = SimpleNamespace(model_type=model_type)
    chat_template_kwargs = {"enable_thinking": True}
    assert _grammar_reasoning_parser(engine, None, chat_template_kwargs) == "qwen_3_5"
    assert _grammar_reasoning_parser(engine, "custom", chat_template_kwargs) == "custom"


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("chat_template_kwargs", [None, {}, {"enable_thinking": False}])
def test_auto_selection_requires_explicit_thinking(model_type, chat_template_kwargs):
    engine = SimpleNamespace(model_type=model_type)
    assert _grammar_reasoning_parser(engine, None, chat_template_kwargs) is None


@pytest.mark.parametrize("generated_opener", [False, True])
def test_accepts_reasoning_then_schema_answer(
    compiler, request_format, generated_opener
):
    compiled = compile_request(compiler, request_format, enable_thinking=True)
    assert compiled._omlx_has_thinking_phase is True
    matcher = xgrammar.GrammarMatcher(compiled)
    if generated_opener:
        accept_tokens(matcher, [THOUGHT_START])
    accept_tokens(
        matcher,
        [*b"Read the photo first.", THOUGHT_END, *b"\n\n", *ANSWER, TURN_END],
    )
    assert matcher.is_terminated()


@pytest.mark.parametrize("invalid_answer", ["plain text", '{"answer":"no"}', "{}"])
def test_rejects_invalid_answer_after_reasoning(
    compiler, request_format, invalid_answer
):
    matcher = xgrammar.GrammarMatcher(
        compile_request(compiler, request_format, enable_thinking=True)
    )
    accept_tokens(matcher, [*b"reason", THOUGHT_END, *b"\n\n"])
    assert not matcher.accept_string(invalid_answer)


def test_cannot_stop_before_reasoning_closes(compiler, request_format):
    matcher = xgrammar.GrammarMatcher(
        compile_request(compiler, request_format, enable_thinking=True)
    )
    accept_tokens(matcher, [*b"reason"])
    assert not matcher.accept_token(TURN_END)


def test_disabled_thinking_constrains_answer_immediately(compiler, request_format):
    compiled = compile_request(compiler, request_format, enable_thinking=False)
    assert not getattr(compiled, "_omlx_has_thinking_phase", False)
    assert not xgrammar.GrammarMatcher(compiled).accept_token(THOUGHT_START)
    assert not xgrammar.GrammarMatcher(compiled).accept_string("plain text")
    matcher = xgrammar.GrammarMatcher(compiled)
    accept_tokens(matcher, [*ANSWER, TURN_END])
    assert matcher.is_terminated()
