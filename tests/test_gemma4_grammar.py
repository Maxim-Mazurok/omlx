# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 grammar checks without model loading or inference."""

from types import SimpleNamespace

import mlx.core as mlx
import numpy
import pytest
import xgrammar

from omlx.api.grammar import GrammarConstraintProcessor
from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.server import (
    _compile_grammar_for_request,
    _compile_with_structural_tag,
    _grammar_reasoning_parser,
)

VOCABULARY = [bytes([i]) for i in range(256)] + [
    b"<|channel>",
    b"<channel|>",
    b"<turn|>",
]
THOUGHT_START = 256
THOUGHT_END = 257
TURN_END = 258
ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "type": "object",
        "properties": {"answer": {"const": "yes"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}
ANSWER = b'{"answer":"yes"}'


@pytest.fixture(scope="module")
def compiler():
    return xgrammar.GrammarCompiler(
        xgrammar.TokenizerInfo(
            VOCABULARY, xgrammar.VocabType.RAW, stop_token_ids=[TURN_END]
        )
    )


def compile_gemma4(compiler, *, enable_thinking=True):
    return _compile_with_structural_tag(
        compiler,
        ANSWER_FORMAT,
        "gemma4",
        {"enable_thinking": enable_thinking},
    )


def accept_tokens(matcher, tokens):
    for token in tokens:
        assert matcher.accept_token(token), VOCABULARY[token]


@pytest.mark.parametrize(
    "model_type",
    ["gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"],
)
def test_auto_selects_gemma4_without_overriding_explicit_parser(model_type):
    engine = SimpleNamespace(model_type=model_type)
    assert _grammar_reasoning_parser(engine, None) == "gemma4"
    assert _grammar_reasoning_parser(engine, "custom") == "custom"


def test_other_model_selection_is_unchanged():
    assert (
        _grammar_reasoning_parser(SimpleNamespace(model_type="gpt_oss"), None)
        == "harmony"
    )
    assert (
        _grammar_reasoning_parser(SimpleNamespace(model_type="llama"), None) is None
    )


def test_model_settings_accept_local_gemma_parser():
    from omlx.admin.routes import ModelSettingsRequest

    assert ModelSettingsRequest(reasoning_parser="gemma4").reasoning_parser == "gemma4"


@pytest.mark.parametrize("generated_opener", [False, True])
def test_accepts_reasoning_then_schema_answer(compiler, generated_opener):
    compiled = compile_gemma4(compiler)
    matcher = xgrammar.GrammarMatcher(compiled)
    if generated_opener:
        accept_tokens(matcher, [THOUGHT_START, *b"thought\n"])
    accept_tokens(matcher, [*b"Read the photo first.", THOUGHT_END, *ANSWER, TURN_END])
    assert matcher.is_terminated()
    assert compiled._omlx_has_thinking_phase is True


def test_schema_is_enforced_after_reasoning(compiler):
    compiled = compile_gemma4(compiler)
    for invalid_answer in [b"plain text", b'{"answer":"no"}', b"{}"]:
        matcher = xgrammar.GrammarMatcher(compiled)
        accept_tokens(matcher, [THOUGHT_START, *b"thought\nreason", THOUGHT_END])
        assert not matcher.accept_string(invalid_answer)


def test_cannot_stop_before_reasoning_closes(compiler):
    matcher = xgrammar.GrammarMatcher(compile_gemma4(compiler))
    accept_tokens(matcher, [THOUGHT_START, *b"thought\nreason"])
    assert not matcher.accept_token(TURN_END)


def test_disabled_thinking_constrains_answer_immediately(compiler):
    compiled = compile_gemma4(compiler, enable_thinking=False)
    assert compiled._omlx_has_thinking_phase is False
    assert not xgrammar.GrammarMatcher(compiled).accept_token(THOUGHT_START)
    assert not xgrammar.GrammarMatcher(compiled).accept_string("plain text")
    matcher = xgrammar.GrammarMatcher(compiled)
    accept_tokens(matcher, [*ANSWER, TURN_END])
    assert matcher.is_terminated()


@pytest.mark.parametrize("budget", [0, 1, 2, 8])
def test_budget_close_remains_valid_even_during_opener(compiler, budget):
    compiled = compile_gemma4(compiler)
    grammar = GrammarConstraintProcessor(compiled, len(VOCABULARY))
    thinking = ThinkingBudgetProcessor(
        think_end_token_ids=[THOUGHT_END],
        budget=budget,
        think_start_token_id=THOUGHT_START,
        token_to_piece=lambda token: VOCABULARY[token],
    )
    history = []
    preferred_reasoning = [THOUGHT_START, *b"thought\nstill thinking"]
    for i in range(len(preferred_reasoning) + 2):
        logits = mlx.full((len(VOCABULARY),), -10.0)
        logits[preferred_reasoning[min(i, len(preferred_reasoning) - 1)]] = 10.0
        logits = grammar(mlx.array(history, dtype=mlx.int32), thinking(history, logits))
        assert bool(mlx.any(mlx.isfinite(logits)).item())
        sampled = int(mlx.argmax(logits).item())
        grammar.accept_token(sampled)
        history.append(sampled)
        if sampled == THOUGHT_END:
            break
    assert history[-1] == THOUGHT_END
    accept_tokens(grammar.matcher, [*ANSWER, TURN_END])
    assert grammar.matcher.is_terminated()


@pytest.mark.parametrize("request_field", ["response_format", "structured_outputs"])
def test_request_schema_compiles_with_gemma_parser(compiler, request_field):
    schema = ANSWER_FORMAT["json_schema"]
    request_format = (
        {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        if request_field == "response_format"
        else {"json": schema}
    )
    compiled = _compile_grammar_for_request(
        SimpleNamespace(grammar_compiler=compiler),
        **{request_field: request_format},
        reasoning_parser="gemma4",
        chat_template_kwargs={"enable_thinking": True},
    )
    matcher = xgrammar.GrammarMatcher(compiled)
    accept_tokens(
        matcher, [THOUGHT_START, *b"thought\nreason", THOUGHT_END, *ANSWER, TURN_END]
    )
    assert matcher.is_terminated()


def test_final_answer_mask_rejects_reasoning_tokens(compiler):
    matcher = xgrammar.GrammarMatcher(compile_gemma4(compiler))
    accept_tokens(matcher, [THOUGHT_START, *b"thought\nreason", THOUGHT_END])
    bitmask = numpy.full((1, (len(VOCABULARY) + 31) // 32), -1, dtype=numpy.int32)
    matcher.fill_next_token_bitmask(bitmask)
    assert not (int(bitmask[0, THOUGHT_START // 32]) >> (THOUGHT_START % 32)) & 1
    assert (int(bitmask[0, ord("{") // 32]) >> (ord("{") % 32)) & 1


@pytest.mark.parametrize(
    "output_format,answer",
    [
        ({"type": "regex", "pattern": r"\d+"}, b"123"),
        ({"type": "grammar", "grammar": 'root ::= "yes"'}, b"yes"),
    ],
)
def test_non_json_constraints_are_preserved(compiler, output_format, answer):
    compiled = _compile_with_structural_tag(
        compiler, output_format, "gemma4", {"enable_thinking": True}
    )
    matcher = xgrammar.GrammarMatcher(compiled)
    accept_tokens(matcher, [THOUGHT_START, *b"thought\nreason", THOUGHT_END])
    assert not matcher.accept_string("invalid answer")
    accept_tokens(matcher, [*answer, TURN_END])
    assert matcher.is_terminated()
