"""Synthetic unit fixtures: no AppWorld execution or model quality claims."""
import io
import json
import unittest
from unittest.mock import patch
import urllib.error

from b2_appworld.budget import Budget, BudgetExceeded
from b2_appworld.llm import (ChatClient, ChatResult, ContextLengthError, LLMError,
                            TruncatedResponseError, message_token_count, strict_json_loads)
from b2_appworld.policy import (DirectSelector, PolicyError, PolicyStats, ZeroShotEffectPredictor,
                               estimate_api_calls, extract_goals, generate_candidates, validate_code)
from b2_appworld.schemas import Candidate, Goal
from b2_appworld.selection import effect_scores, select_by_effects, select_by_scalar


class FakeClient:
    tokenizer = None

    def __init__(self, value):
        self.value = value
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        if isinstance(self.value, Exception):
            raise self.value
        return ChatResult(json.dumps(self.value), 10, 10, 0.01)


class Response:
    def __init__(self, data):
        self.data = data if isinstance(data, bytes) else json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return self.data[:size]


def completion(content="pass", finish="stop", usage=True):
    data = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    if usage:
        data["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
    return data


class CorePolicyTests(unittest.TestCase):
    def setUp(self):
        self.instruction = "Create a task. Preserve the old note."
        self.goals = (Goal("g1", "Create task", "Create a task"),
                      Goal("g2", "Preserve note", "Preserve the old note"))
        self.original = "# unchanged actor block\nprint('hello')\n"

    def test_evidence_validation_fails_closed_without_hallucinated_goal(self):
        stats = PolicyStats()
        client = FakeClient({"goals": [{"id": "g1", "text": "Delete note", "evidence_span": "Delete note", "kind": "state_write"}]})
        self.assertEqual(extract_goals(client, self.instruction, stats=stats), ())
        self.assertEqual(stats.fallbacks, {"goal_extraction": 1})
        self.assertEqual(client.requests[0][1]["max_tokens"], 512)

    def test_goal_kind_preserved(self):
        client = FakeClient({"goals": [{"id": "g1", "text": "Answer", "evidence_span": "Create a task", "kind": "final_answer"}]})
        self.assertEqual(extract_goals(client, self.instruction)[0].kind, "final_answer")

    def test_original_exact_and_costs_are_static_estimates(self):
        rows = [{"id": name, "code": f"apis.notes.update_note(note_id={i})", "target_goal_ids": ["g2"]}
                for i, name in enumerate(("broad", "local", "composite"))]
        candidates = generate_candidates(FakeClient({"candidates": rows}), self.instruction, [], self.goals, self.original)
        self.assertEqual([c.id for c in candidates], ["original", "local", "composite", "broad"])
        self.assertEqual(candidates[0].code, self.original)
        self.assertEqual(candidates[1].estimated_api_calls, 1)
        self.assertEqual(estimate_api_calls("for i in range(100):\n apis.notes.update_note(note_id=i)"), 1)

    def test_invalid_candidate_falls_back_without_slicing_or_fabricating_modes(self):
        rows = [{"id": name, "code": "x='" + ("a" * 1000) + "'", "target_goal_ids": []}
                for name in ("local", "composite", "broad")]
        stats = PolicyStats()
        candidates = generate_candidates(FakeClient({"candidates": rows}), self.instruction, [], self.goals, self.original, stats=stats)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].code, self.original)
        self.assertEqual(stats.fallbacks["candidate_generation"], 1)

    def test_bad_actor_block_never_becomes_candidate(self):
        with self.assertRaises(PolicyError):
            generate_candidates(FakeClient({}), self.instruction, [], self.goals, "apis.notes.update_note(")
        with self.assertRaises(PolicyError):
            validate_code("break")

    def test_hidden_runtime_fields_rejected_before_model_call(self):
        client = FakeClient({})
        with self.assertRaises(PolicyError):
            generate_candidates(client, self.instruction, [{"ground_truth": "secret"}], self.goals, self.original)
        self.assertEqual(client.requests, [])

    def test_direct_selection_invalid_id_is_observable_fallback(self):
        stats = PolicyStats()
        pool = (Candidate("original", "pass", ()), Candidate("local", "pass", ()))
        selected = DirectSelector(FakeClient({"candidate_id": "unknown"}), stats).select(self.instruction, [], self.goals, pool)
        self.assertEqual(selected.id, "original")
        self.assertEqual(stats.fallbacks["direct_selection"], 1)

    def test_zero_shot_effect_reorders_ids_without_changing_probabilities(self):
        pool = (Candidate("original", "pass", ()),)
        data = {"effects": [{"candidate_id": "original", "goals": [
            {"goal_id": "g2", "probabilities": [0, 0, 0, 1]},
            {"goal_id": "g1", "probabilities": [0, 1, 0, 0]}]}]}
        self.assertEqual(ZeroShotEffectPredictor(FakeClient(data)).predict(self.instruction, [], self.goals, pool),
                         [[[0, 1, 0, 0], [0, 0, 0, 1]]])
        data["effects"][0]["goals"][0]["probabilities"] = [0.1, 0.1, 0.1, 0.1]
        with self.assertRaises(PolicyError):
            ZeroShotEffectPredictor(FakeClient(data)).predict(self.instruction, [], self.goals, pool)

    def test_utility_normalizes_goal_count_and_pool_cost(self):
        pool = (Candidate("a", "pass", (), 2), Candidate("b", "pass", (), 4))
        scores = effect_scores(pool, [[[0, 1, 0, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 0, 0, 1]]])
        self.assertAlmostEqual(scores[0], 0.475)
        self.assertAlmostEqual(scores[1], 0.45)
        self.assertEqual(select_by_effects(pool, [[[0, 0, 1, 0]], [[0, 0, 0, 1]]]).id, "b")
        self.assertEqual(select_by_scalar(pool, [0.5, 0.5]).id, "a")

    def test_nonfinite_or_unnormalized_probability_rejected(self):
        for distribution in ([0, 0, 0, 0], [float("nan"), 0, 0, 1], [True, 0, 0, 0]):
            with self.assertRaises(ValueError):
                effect_scores([Candidate("a", "pass", ())], [[distribution]])
        with self.assertRaises(ValueError):
            strict_json_loads('{"x":1,"x":2}')

    def test_no_budget_errors_turn_into_fallback_mutations(self):
        with self.assertRaises(BudgetExceeded):
            extract_goals(FakeClient(BudgetExceeded("cap")), self.instruction)


class BudgetAndHTTPTests(unittest.TestCase):
    messages = [{"role": "user", "content": "hello"}]

    def test_request_is_clamped_to_shared_remaining_output(self):
        budget = Budget(max_output_tokens=10)
        client = ChatClient("http://localhost:8000/v1", "fixture", budget=budget)
        with patch("urllib.request.urlopen", return_value=Response(completion())) as opener:
            client.complete(self.messages, max_tokens=30)
            client.complete(self.messages, max_tokens=30)
        payloads = [json.loads(call.args[0].data) for call in opener.call_args_list]
        self.assertEqual([x["max_tokens"] for x in payloads], [10, 8])
        self.assertEqual(budget.snapshot()["output_tokens"], 4)
        self.assertEqual(budget.calls, 2)

    def test_reservation_and_execution_hard_caps(self):
        budget = Budget(max_output_tokens=5, max_total_tokens=20)
        reservation = budget.reserve_chat(18, 5)
        self.assertEqual(reservation.max_output_tokens, 2)
        with self.assertRaises(BudgetExceeded):
            budget.reserve_chat(0, 1)
        budget.record_chat(reservation, 18, 2)
        with self.assertRaises(BudgetExceeded):
            budget.ensure_can_execute()

    def test_missing_usage_is_conservatively_paid_and_blocks_execution(self):
        budget = Budget(max_output_tokens=7)
        client = ChatClient("http://localhost:8000/v1", "fixture", budget=budget)
        with patch("urllib.request.urlopen", return_value=Response(completion(usage=False))):
            result = client.complete(self.messages, max_tokens=7)
        self.assertTrue(result.usage_estimated)
        self.assertEqual(budget.output_tokens, 7)
        with self.assertRaises(BudgetExceeded):
            budget.ensure_can_execute()

    def test_malformed_response_and_http_retries_are_paid(self):
        budget = Budget(max_output_tokens=20)
        client = ChatClient("http://localhost:8000/v1", "fixture", budget=budget, max_retries=1)
        with patch("urllib.request.urlopen", return_value=Response(b"bad json")):
            with self.assertRaises(LLMError):
                client.complete(self.messages, max_tokens=3)
        failure = urllib.error.HTTPError("http://local", 503, "unavailable", {}, io.BytesIO())
        with patch("urllib.request.urlopen", side_effect=[failure, Response(completion())]), patch("time.sleep"):
            client.complete(self.messages, max_tokens=3)
        self.assertEqual(budget.calls, 3)
        self.assertEqual(budget.output_tokens, 3 + 3 + 2)

    def test_truncated_completion_is_paid_but_never_returned(self):
        budget = Budget(max_output_tokens=10)
        client = ChatClient("http://localhost:8000/v1", "fixture", budget=budget)
        with patch("urllib.request.urlopen", return_value=Response(completion("x=(", "length"))):
            with self.assertRaises(TruncatedResponseError):
                client.complete(self.messages, max_tokens=5)
        self.assertEqual(budget.calls, 1)
        self.assertEqual(budget.output_tokens, 2)

    def test_classifier_is_separately_accounted_and_capped(self):
        budget = Budget(max_classifier_tokens=100)
        budget.record_classifier(80, seconds=1)
        self.assertEqual(budget.snapshot()["classifier_processed_tokens"], 80)
        with self.assertRaises(BudgetExceeded):
            budget.check_classifier(21)
        with self.assertRaises(BudgetExceeded):
            budget.record_classifier(21)
        self.assertEqual(budget.classifier_processed_tokens, 101)

    def test_context_caps_generation_and_oversized_input_is_never_sent(self):
        prompt_size = message_token_count(self.messages)
        client = ChatClient("http://localhost:8000/v1", "fixture", context_length=prompt_size + 3)
        with patch("urllib.request.urlopen", return_value=Response(completion())) as opener:
            client.complete(self.messages, max_tokens=50)
        self.assertEqual(json.loads(opener.call_args.args[0].data)["max_tokens"], 3)
        client = ChatClient("http://localhost:8000/v1", "fixture", context_length=prompt_size)
        with patch("urllib.request.urlopen") as opener:
            with self.assertRaises(ContextLengthError):
                client.complete(self.messages, max_tokens=50)
        opener.assert_not_called()
        self.assertEqual(client.budget.calls, 0)

    def test_elapsed_http_time_survives_frozen_python_clocks(self):
        client = ChatClient("http://localhost:8000/v1", "fixture")
        with patch("time.perf_counter", return_value=0), patch("time.monotonic", return_value=0), \
                patch("time.clock_gettime", side_effect=[100.0, 101.5]), \
                patch("urllib.request.urlopen", return_value=Response(completion())):
            result = client.complete(self.messages, max_tokens=5)
        self.assertEqual(result.seconds, 1.5)
        self.assertEqual(client.budget.seconds, 1.5)


if __name__ == "__main__":
    unittest.main()
