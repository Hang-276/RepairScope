"""No-network transport tests: unusable completions must never become programs."""
import io
import json
import urllib.error

import pytest

from repairscope.budget import Budget, BudgetExceeded
from repairscope.llm import (ChatClient, ContextLengthError, LLMError,
                             TruncatedResponseError, message_token_count)


MESSAGES = [{"role": "user", "content": "Return a fixed repair program."}]


class Response:
    def __init__(self, value):
        self.raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.read_limit = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        self.read_limit = limit
        return self.raw[:limit]


def completion(content='{"program":"print(1)"}', finish="stop", usage=True):
    value = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    if usage:
        value["usage"] = {"prompt_tokens": 12, "completion_tokens": 3}
    return value


def client(**kwargs):
    return ChatClient("https://transport.invalid/v1", "fixed-model", max_retries=0, **kwargs)


@pytest.fixture(autouse=True)
def no_accidental_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A test attempted real network access")
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", forbidden)
    monkeypatch.setattr("repairscope.llm.time.sleep", lambda seconds: None)


def test_positive_completion_reserves_before_http_and_uses_shared_budget(monkeypatch):
    ledger = Budget(max_calls=2, max_output_tokens=10)
    requests = []
    def transport(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        # The generation allowance is already unavailable while HTTP is pending.
        assert ledger.remaining_output_tokens == 3
        assert ledger.snapshot()["calls"] == 0
        return Response(completion())
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = client(chat_template_kwargs={"enable_thinking": False})
    result = chat.complete(MESSAGES, max_tokens=7, budget=ledger, purpose="offline_learning")
    assert result.content == '{"program":"print(1)"}'
    assert (result.input_tokens, result.output_tokens, result.usage_estimated) == (12, 3, False)
    assert requests[0]["messages"] == MESSAGES
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert ledger.snapshot()["calls"] == 1 and ledger.snapshot()["output_tokens"] == 3
    assert ledger.snapshot()["events"][0]["purpose"] == "offline_learning"
    assert chat.usage["calls"] == 0  # Event ledger is shared explicitly, not duplicated.


@pytest.mark.parametrize("finish", ["length", "max_tokens"])
def test_complete_looking_truncated_json_is_rejected_and_charged(monkeypatch, finish):
    calls = []
    def transport(request, timeout):
        calls.append(request)
        return Response(completion(finish=finish))
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = ChatClient("https://transport.invalid/v1", "m", max_retries=2)
    with pytest.raises(TruncatedResponseError):
        chat.complete(MESSAGES, max_tokens=20)
    assert len(calls) == chat.usage["calls"] == 1
    assert chat.usage["output_tokens"] == 3


def test_missing_usage_consumes_reserved_output_and_blocks_next_request(monkeypatch):
    calls = []
    def transport(request, timeout):
        calls.append(json.loads(request.data))
        return Response(completion(usage=False))
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = client(budget=Budget(max_output_tokens=5))
    result = chat.complete(MESSAGES, max_tokens=100)
    assert calls[0]["max_tokens"] == result.output_tokens == 5
    assert result.usage_estimated and chat.usage["estimated_usage_calls"] == 1
    with pytest.raises(BudgetExceeded):
        chat.complete(MESSAGES, max_tokens=1)
    assert len(calls) == 1


def test_retry_is_charged_and_next_attempt_uses_remaining_cap(monkeypatch):
    requests = []
    def transport(request, timeout):
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", {}, io.BytesIO(b"secret-body"))
        response = completion()
        response["usage"]["completion_tokens"] = 1
        return Response(response)
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = ChatClient("https://transport.invalid/v1", "m", max_retries=2,
                      budget=Budget(max_output_tokens=10, max_calls=3))
    result = chat.complete(MESSAGES, max_tokens=8)
    assert result.output_tokens == 1
    assert [r["max_tokens"] for r in requests] == [8, 2]
    assert chat.usage["calls"] == 2 and chat.usage["output_tokens"] == 9
    assert chat.usage["events"][0]["status"] == "http_error"


def test_call_budget_blocks_retry_before_second_http(monkeypatch):
    calls = []
    def transport(request, timeout):
        calls.append(request)
        raise urllib.error.URLError("temporary transport failure")
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = ChatClient("https://transport.invalid/v1", "m", max_retries=5,
                      budget=Budget(max_calls=1, max_output_tokens=100))
    with pytest.raises(BudgetExceeded):
        chat.complete(MESSAGES, max_tokens=5)
    assert len(calls) == chat.usage["calls"] == 1
    assert chat.usage["output_tokens"] == 5


def test_nonretryable_http_error_has_no_secret_echo(monkeypatch):
    calls = []
    def transport(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "SECRET-REASON", {}, io.BytesIO(b"SECRET-BODY"))
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = ChatClient("https://transport.invalid/v1", "m", max_retries=5)
    with pytest.raises(LLMError) as error:
        chat.complete(MESSAGES, max_tokens=5)
    assert "400" in str(error.value) and "SECRET" not in str(error.value)
    assert len(calls) == chat.usage["calls"] == 1


def test_retry_limit_is_finite_even_with_large_budget(monkeypatch):
    calls = []
    def transport(request, timeout):
        calls.append(request)
        raise TimeoutError("not returned")
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = ChatClient("https://transport.invalid/v1", "m", max_retries=2)
    with pytest.raises(LLMError):
        chat.complete(MESSAGES, max_tokens=4)
    assert len(calls) == chat.usage["calls"] == 3
    assert chat.usage["output_tokens"] == 12


@pytest.mark.parametrize("value", [
    {"choices": [None]}, {"choices": [1]}, {"choices": []},
    {"choices": [{"finish_reason": "stop", "message": None}]},
    {"choices": [{"finish_reason": "tool_calls", "message": {"content": "x"}}]},
    b'{"choices":[],"choices":[]}', b'{"score":NaN}', b'not JSON',
])
def test_malformed_response_fails_closed_without_free_retry(monkeypatch, value):
    calls = []
    def transport(request, timeout):
        calls.append(request)
        return Response(value)
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    chat = ChatClient("https://transport.invalid/v1", "m", max_retries=2)
    with pytest.raises(LLMError):
        chat.complete(MESSAGES, max_tokens=6)
    assert len(calls) == chat.usage["calls"] == 1
    assert chat.usage["output_tokens"] == 6


def test_response_byte_limit_is_bounded_and_charged(monkeypatch):
    response = Response(b"x" * 10000)
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", lambda request, timeout: response)
    chat = client(max_response_bytes=40)
    with pytest.raises(LLMError):
        chat.complete(MESSAGES, max_tokens=6)
    assert response.read_limit == 41
    assert chat.usage["calls"] == 1 and chat.usage["output_tokens"] == 6


def test_server_token_overrun_is_recorded_and_never_returned(monkeypatch):
    value = completion()
    value["usage"]["completion_tokens"] = 7
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", lambda request, timeout: Response(value))
    chat = client(budget=Budget(max_output_tokens=100))
    with pytest.raises(BudgetExceeded):
        chat.complete(MESSAGES, max_tokens=6)
    assert chat.usage["calls"] == 1 and chat.usage["output_tokens"] == 7


def test_context_cap_preserves_full_prompt_and_rejects_unfittable_input(monkeypatch):
    requests = []
    def transport(request, timeout):
        requests.append(json.loads(request.data))
        return Response(completion())
    monkeypatch.setattr("repairscope.llm.urllib.request.urlopen", transport)
    estimated_input = message_token_count(MESSAGES)
    chat = client(context_length=estimated_input + 5)
    chat.complete(MESSAGES, max_tokens=30)
    assert requests[0]["max_tokens"] == 5 and requests[0]["messages"] == MESSAGES
    too_small = client(context_length=estimated_input)
    with pytest.raises(ContextLengthError):
        too_small.complete(MESSAGES)
    assert too_small.usage["calls"] == 0 and len(requests) == 1
