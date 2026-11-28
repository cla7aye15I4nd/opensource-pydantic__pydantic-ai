"""The OpenAI Chat Completions and Responses streams collect the text they keep across deltas in linear time.

Appending each delta to a string with `+=` copies the whole string each time, so the work grows with the square of
the text's length. The tests stream deltas as a `str` subclass that counts the characters each concatenation onto it
copies, which turns that cost into an operation count rather than a timing: a linear stream copies each character at
most once. They use synthetic events, since the cost only shows on streams far longer than any recording.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types import chat, responses
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI, MockOpenAIResponses, response_stream

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

DELTA = 'abcdefgh'


@dataclass
class _Copies:
    chars: int = 0

    def delta(self, text: str = DELTA) -> str:
        """`text` as a delta that adds the length of the string it's appended to to `chars`."""
        copies = self

        class CountingStr(str):
            def __radd__(self, other: str) -> str:
                # Python calls this before `str.__add__` for `other + self` and `other += self`.
                copies.chars += len(other)
                return other + str(self)

        return CountingStr(text)


def _responses_model(events: Sequence[responses.ResponseStreamEvent]) -> Model:
    return OpenAIResponsesModel(
        'gpt-5', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock_stream(response_stream(events)))
    )


def _chat_model(deltas: list[ChoiceDelta], profile: OpenAIModelProfile | None = None) -> Model:
    chunks = [
        chat.ChatCompletionChunk(
            id='chunk',
            created=0,
            model='gpt-4o',
            object='chat.completion.chunk',
            choices=[ChunkChoice(index=0, delta=delta, finish_reason='stop' if final else None)],
        )
        for delta, final in [*((delta, False) for delta in deltas), (ChoiceDelta(), True)]
    ]
    return OpenAIChatModel(
        'gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(chunks)), profile=profile
    )


def _refusal(response: ModelResponse) -> object:
    return (response.provider_details or {}).get('refusal')


def _texts(response: ModelResponse) -> object:
    return [part.content for part in response.parts if isinstance(part, TextPart)]


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[_Copies, int], Model]
    output: Callable[[ModelResponse], object]
    expected: Callable[[int], object]


CASES = [
    Case(
        'responses-function-call-args',
        lambda copies, n: _responses_model(
            [
                responses.ResponseOutputItemAddedEvent(
                    item=responses.ResponseFunctionToolCall(
                        id='fc_1', call_id='call_1', name='lookup', arguments='', type='function_call'
                    ),
                    output_index=0,
                    type='response.output_item.added',
                    sequence_number=0,
                ),
                *(
                    responses.ResponseFunctionCallArgumentsDeltaEvent(
                        item_id='fc_1',
                        output_index=0,
                        delta='',
                        type='response.function_call_arguments.delta',
                        sequence_number=0,
                    ).model_copy(update={'delta': copies.delta()})
                    for _ in range(n)
                ),
                responses.ResponseFunctionCallArgumentsDoneEvent(
                    arguments=DELTA * n,
                    item_id='fc_1',
                    output_index=0,
                    type='response.function_call_arguments.done',
                    sequence_number=0,
                ),
            ]
        ),
        lambda response: [part.args for part in response.parts if isinstance(part, ToolCallPart)],
        lambda n: [DELTA * n],
    ),
    Case(
        'responses-refusal',
        lambda copies, n: _responses_model(
            [
                responses.ResponseRefusalDeltaEvent(
                    content_index=0,
                    delta='',
                    item_id='msg_1',
                    output_index=0,
                    type='response.refusal.delta',
                    sequence_number=0,
                ).model_copy(update={'delta': copies.delta()})
                for _ in range(n)
            ]
        ),
        _refusal,
        lambda n: DELTA * n,
    ),
    Case(
        'chat-text',
        lambda copies, n: _chat_model(
            [ChoiceDelta(content='').model_copy(update={'content': copies.delta()}) for _ in range(n)]
        ),
        _texts,
        lambda n: [DELTA * n],
    ),
    Case(
        'chat-refusal',
        lambda copies, n: _chat_model(
            [ChoiceDelta(refusal='').model_copy(update={'refusal': copies.delta()}) for _ in range(n)]
        ),
        _refusal,
        lambda n: DELTA * n,
    ),
    # Whitespace-only text after reasoning that interrupted a text part is held for the next text part.
    Case(
        'chat-whitespace-after-reasoning',
        lambda copies, n: _chat_model(
            [
                ChoiceDelta(content='a'),
                ChoiceDelta.model_validate({'reasoning_content': 'thinking'}),
                *(ChoiceDelta(content='').model_copy(update={'content': copies.delta(' ' * 8)}) for _ in range(n)),
                ChoiceDelta(content='b'),
            ],
            OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
        ),
        _texts,
        lambda n: ['a', ' ' * 8 * n + 'b'],
    ),
]


def test_counting_delta_counts_the_copied_characters():
    """The control for the test below: the counter sees `+=` onto a plain string copy that string."""
    copies = _Copies()
    text = 'abc'
    text += copies.delta('d')
    assert (text, copies.chars) == ('abcd', 3)


@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_stream_copies_each_character_at_most_once(allow_model_requests: None, case: Case, n: int):
    copies = _Copies()
    model = case.model(copies, n)
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
    ) as streamed:
        async for _ in streamed:
            pass
    assert case.output(streamed.get()) == case.expected(n)
    assert copies.chars <= 8 * n
