import copy
import json
from types import SimpleNamespace

import pytest

from repairscope.actor import FrozenActor, normalize_action, public_state
from repairscope.control import BudgetExceeded, BudgetLedger, RepairController, evaluate_goals, score_prediction
from repairscope.fixtures import FixtureEnv, fixture_goals, fixture_records, fixture_state
from repairscope.metrics import calibration, compute_metrics, official_metrics
from repairscope.evaluation import evaluate_run
from repairscope.probing import ExperienceReplay, _branch_records, _load_plans, _prepare_plans, candidate_statistics, plan_cost, run_probing, selection_probabilities
from repairscope.candidates import aggregate_candidates, predict_and_rank


def altered(state, key, value):
    result = copy.deepcopy(state)
    next(f for f in result["fields"] if f["key"] == key)["value"] = value
    return result


def test_public_actor_inputs_cannot_expose_audit_or_hidden_labels():
    state = fixture_state()
    state.update(audit={"secret": "hidden-db"}, checker="hidden-checker")
    state["fields"][0]["secret"] = "hidden-column"
    actor = FrozenActor({"kind": "scripted", "actions": [{"name": "shop.read_order", "arguments": {}}]})
    messages = actor._messages(state, fixture_goals(), [{"state": state, "action": {"name": "noop"}, "audit": "hidden-history"}], 4, "execute", "public")
    text = json.dumps(messages)
    assert all(secret not in text for secret in ("hidden-db", "hidden-checker", "hidden-column", "hidden-history"))
    assert "shop.Order[1].paid" in text


def test_unknown_fields_do_not_reveal_stale_values():
    state = fixture_state()
    state["fields"][0].update(known=False, value="secret")
    assert public_state(state)["fields"][0]["value"] is None
    with pytest.raises(ValueError):
        normalize_action({"name": "noop", "cost": 0})


def test_failed_calls_consume_budget_and_further_call_is_blocked():
    ledger = BudgetLedger(1)
    with pytest.raises(RuntimeError):
        ledger.call("failed_api", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert ledger.spent == 1 and ledger.events[-1]["valid"] is False
    with pytest.raises(BudgetExceeded):
        ledger.call("not_executed", lambda: pytest.fail("must not call"))


def test_incurred_atomic_overrun_is_not_lost_from_cost():
    ledger = BudgetLedger(2)
    ledger.charge("action", 1)
    with pytest.raises(BudgetExceeded):
        ledger.settle("actual", 4, 1)
    assert ledger.spent == 4


def test_goal_scoring_prefers_protected_payment_over_shipping_gain():
    state = fixture_state("fragile")
    safe = altered(state, "shop.Order[1].shipping", "express")
    unsafe = altered(safe, "shop.Order[1].paid", False)
    action = {"name": "shop.set_shipping", "cost": 1}
    a = score_prediction(state, {"next_state": safe}, fixture_goals(), action)
    b = score_prediction(state, {"next_state": unsafe}, fixture_goals(), action)
    assert a["score"] > b["score"] and b["damage"] == 2


def test_controller_prevents_damage_and_never_calls_hidden_checker():
    env = FixtureEnv({"environment": {"condition": "fragile", "offline_audit": True}}, "fixture", "train", "test")
    actor = FrozenActor({"kind": "fixture"})
    records = []
    def predict(record):
        records.append(record)
        state = altered(record["state"], "shop.Order[1].shipping", "express")
        if record["action"]["name"] == "shop.set_shipping":
            state = altered(state, "shop.Order[1].paid", False)
        return {"next_state": state, "scope": ["shop.Order[1].shipping"], "escape": 0}
    env.evaluate = lambda: pytest.fail("hidden checker must not be queried by controller")
    result = RepairController(actor, predict, {"max_steps": 1, "max_calls": 20}).run(env)
    assert evaluate_goals(result["state"], fixture_goals()) == [True, True]
    assert result["logs"][0]["action"]["name"] == "shop.set_shipping_preserve_payment"
    assert all("audit" not in r and "goal_after" not in r for r in records)
    assert result["cost"] == 5  # reset, actor, two predictions, execute


def test_repair_candidates_reserve_noop_without_expanding_candidate_count():
    controller = RepairController(FrozenActor({"kind": "fixture"}), None, {"candidate_count": 2})
    candidates = controller._candidates(fixture_state(), fixture_goals(), 0, "repair", "test")
    assert len(candidates) <= 2 and any(a["name"] == "noop" for a in candidates)


def test_untrusted_read_check_is_rejected():
    env = SimpleNamespace(metadata={"read_only_actions": [{"name": "shop.delete_order", "arguments": {}}]})
    controller = RepairController(None, None)
    with pytest.raises(ValueError, match="read_only"):
        controller._legal_check(env, fixture_state(), fixture_goals(), {"escape": .9}, 0)


def test_partial_audit_negative_is_unknown_not_escape_zero():
    record = {"scope": [], "audit": {"complete": False, "changes": []}, "prediction": {"escape": .2}}
    metrics = compute_metrics([record])
    assert metrics["escape_evaluable_steps"] == 0
    assert metrics["escape_calibration"]["n"] == 0
    positive = {**record, "audit": {"complete": False, "changes": ["x"]}}
    assert compute_metrics([positive])["escape_rate"] == 1


def test_native_tgc_and_sgc_use_scenario_strict_success():
    results = [{"task_id": task, "official": {"task_id": task, "evidence_origin": "appworld", "tracker": {"success": success}}}
               for task, success in [("a_1", True), ("a_2", False), ("b_1", True)]]
    assert official_metrics(results)["TGC"] == 66.7
    assert official_metrics(results)["SGC"] == 50
    assert official_metrics(results, ["a_1", "a_2", "b_1", "b_2"])["TGC"] is None
    assert official_metrics([{ "task_id": "a_1", "evidence_origin": "synthetic", "tracker": {"success": True}}])["TGC"] is None


def test_calibration_perfect_predictions():
    result = calibration([0, 1, 0], [0, 1, 0])
    assert result["brier"] == 0 and result["ece"] == 0


def test_probing_epsilon_preserves_every_candidate_probability():
    stats = [{"U": 0, "D": 0, "R": 0, "C": 1}, {"U": 1, "D": 1, "R": 1, "C": 1}]
    probabilities = selection_probabilities(stats, "risk", .1)
    assert probabilities == pytest.approx([.05, .95])
    assert selection_probabilities(stats, "random", .1) == pytest.approx([.5, .5])


def test_replay_forbids_query_labels_and_keeps_counterfactual_siblings():
    replay = ExperienceReplay(100, 0)
    records = fixture_records(1)
    replay.add(records, old=True)
    pair_records = [r for r in records if r.get("cf_pair_id")]
    replay2 = ExperienceReplay(100, 0)
    replay2.add(pair_records)
    assert len(replay2.sample(1)) == 4
    with pytest.raises(ValueError):
        replay.add([{**records[0], "frozen_query": True}])


def test_two_branch_continuation_cost_counts_all_replays():
    state = fixture_state()
    plan = {"task_id": "fixture", "state": state, "goals": fixture_goals(),
            "action": {"name": "shop.set_shipping", "arguments": {"shipping": "express"}},
            "reference": {"name": "shop.read_order", "arguments": {}},
            "continuation": [{"name": "shop.pay_order", "arguments": {}}]}
    env = FixtureEnv({"environment": {"offline_audit": True}}, "fixture", "train", "probe")
    predict = lambda record: {"next_state": record["state"], "scope": ["shop.Order[1].shipping"], "escape": .1}
    ledger = BudgetLedger(30)
    records, receipts = _branch_records(env, plan, predict, ledger, "test")
    assert len(records) == 4 and ledger.spent == 6 and plan_cost(plan) == 6
    assert [r["actual_cost"] for r in receipts] == [1, 2, 1, 2]


def test_mc_dropout_hypotheses_drive_probe_disagreement():
    plan = {"state": fixture_state(), "goals": fixture_goals(), "action": {"name": "a"}, "reference": {"name": "b"}}
    prediction = {"next_state": fixture_state(), "escape": 1, "uncertainty": .5,
                  "scope_probabilities": {}, "mc_scope_probabilities": [{"x": 0}, {"x": 1}]}
    stats = candidate_statistics(plan, lambda record: prediction)
    assert stats["D"] == 1 and stats["R"] > 0


def test_candidates_preserve_frequency_topk_and_first_seen_ties():
    selected = aggregate_candidates(["B", "A", "A", "B", "C"], count=2)
    assert [r["action"] for r in selected] == ["B", "A"]
    assert [r["prob"] for r in selected] == [.5, .5]
    ranked = predict_and_rank(["B", "A"], predict=lambda a: "next-" + a, score=lambda a, p: 1)
    assert [r["action"] for r in ranked] == ["B", "A"]
    assert ranked[0]["prediction"] == "next-B"


def test_legal_four_grid_with_continuation_counts_preparation_and_replay():
    config = {"environment": {"backend": "fixture", "offline_audit": True}}
    plan = {"task_id": "fixture", "action": {"name": "shop.set_shipping", "arguments": {"shipping": "express"}},
            "reference": {"name": "shop.set_shipping_preserve_payment", "arguments": {"shipping": "express"}},
            "conditions": [{"actions": [{"name": "shop.set_condition", "arguments": {"condition": c}}],
                            "query": {"name": "shop.read_order", "arguments": {}, "read_only": True}, "expected_output": c}
                           for c in ("standard", "fragile")],
            "continuation": [{"name": "shop.pay_order", "arguments": {}, "cost": 1}]}
    ledger = BudgetLedger(34)
    prepared = _prepare_plans(config, [plan], ledger)[0]
    assert ledger.spent == 6
    predictor = lambda r: {"next_state": r["state"], "scope": [], "escape": .5}
    env = FixtureEnv(config, "fixture", "train", "test")
    records, receipts = _branch_records(env, prepared, predictor, ledger, "test")
    assert ledger.spent == 34 and len(records) == 8
    assert all(r["prequential_prediction_valid"] for r in records)
    assert all(evaluate_goals(r["next_state"], fixture_goals())[1] is True for r in records if r["step"] == 1)


def test_typed_mc_disagreement_remains_nonzero_with_deterministic_scope():
    state = fixture_state()
    plan = {"state": state, "goals": fixture_goals(), "action": {"name": "a"}, "reference": {"name": "b"}}
    prediction = {"next_state": state, "escape": .8, "uncertainty": 0,
                  "mc_scope_probabilities": [{"x": .5}, {"x": .5}],
                  "mc_predictions": [{"next_state": state, "change_probabilities": {"x": .1}},
                                     {"next_state": altered(state, "shop.Order[1].paid", False), "change_probabilities": {"x": .9}}]}
    assert candidate_statistics(plan, lambda r: prediction)["D"] > 0


def test_single_condition_dictionary_produces_legal_two_branch_plan():
    plan = {"task_id": "fixture", "conditions": {"c0": {"actions": [], "query": {"name": "shop.read_order"}, "expected_output": "standard"}}}
    result = _load_plans({"probing": {"plans": {"plans": [plan]}}})
    assert len(result[0]["conditions"]) == 1


def test_disabled_probing_keeps_initial_frozen_query_baseline(monkeypatch, tmp_path):
    import repairscope.training as training
    import repairscope.model as model_module
    import repairscope.probing as probing
    model = SimpleNamespace(use_escape=True)
    monkeypatch.setattr(training, "load_predictor", lambda *a: (model, None, {"model": {"d_model": 32}, "checkpoint_manifest": {"groups": [], "evidence_origin": ["synthetic"]}}))
    monkeypatch.setattr(model_module, "predict_record", lambda *a, **kw: {"next_state": fixture_state()})
    def query(config, tasks, predict, output):
        assert tasks == ["fixture-query"]
        predict({"state": fixture_state()})
        return {"cost": 2, "metrics": {"goal_regression_rate": 0}}
    monkeypatch.setattr(probing, "_frozen_queries", query)
    result = run_probing({"environment": {"backend": "fixture"}, "probing": {"budget_calls": 0, "query_task_ids": ["fixture-query"]}}, "unused.pt", tmp_path)
    assert result["cost"] == 0 and result["stop_reason"] == "disabled"
    assert result["learning_curve"][0]["initial"] is True
    assert result["frozen_query_total_calls"] == 2 and result["model_calls"]["predictions"] == 1


def test_online_checkpoint_save_uses_loaded_architecture(monkeypatch, tmp_path):
    import repairscope.training as training
    import repairscope.model as model_module
    saved = []
    model = SimpleNamespace(use_escape=True)
    monkeypatch.setattr(training, "load_predictor", lambda *a: (model, None, {"model": {"d_model": 32}, "checkpoint_manifest": {"groups": [], "evidence_origin": ["synthetic"]}}))
    monkeypatch.setattr(model_module, "predict_record", lambda m, t, record, **kw: {"next_state": record["state"], "scope": [], "escape": .2, "uncertainty": .1})
    monkeypatch.setattr(training, "update_predictor", lambda *a, **kw: {"optimizer": None, "updates": 1})
    def save(model, tensorizer, config, path, **kwargs):
        saved.append(config["model"])
        return str(path)
    monkeypatch.setattr(training, "save_predictor", save)
    plan = {"task_id": "fixture", "state": fixture_state(), "action": {"name": "shop.set_shipping", "arguments": {}},
            "reference": {"name": "shop.read_order", "arguments": {}}}
    result = run_probing({"model": {"d_model": 384}, "environment": {"backend": "fixture"},
                          "probing": {"plans": [plan], "budget_calls": 2, "update_every": 1, "checkpoints": []}}, "unused.pt", tmp_path)
    assert saved == [{"d_model": 32}]
    assert result["cost"] == 2 and result["records"] == 2


def test_native_experiment_names_are_unique_and_evidence_cannot_be_overwritten(monkeypatch, tmp_path):
    import repairscope.environment as environment
    from repairscope.probing import _frozen_queries
    names = []
    def create(config, task_id, split, experiment):
        names.append(experiment)
        return FixtureEnv(config, task_id, split, experiment)
    monkeypatch.setattr(environment, "create_environment", create)
    config = {"variant": "full", "train": {"seed": 100}, "environment": {"backend": "fixture"},
              "actor": {"kind": "fixture"}, "controller": {"actor_only": True, "max_steps": 1, "max_repairs": 0},
              "evaluation": {"split": "test_normal", "bootstrap_samples": 2}}
    a = evaluate_run(config, None, ["fixture"], tmp_path / "a")
    b = evaluate_run(config, None, ["fixture"], tmp_path / "b")
    assert a["experiment"] != b["experiment"] and names == [a["experiment"], b["experiment"]]
    predictor = lambda record: {"next_state": record["state"], "scope": [], "escape": 0}
    q = _frozen_queries(config, ["fixture-query"], predictor, tmp_path / "query-budget-1")
    assert q["experiment"] not in names[:2]
    with pytest.raises(FileExistsError):
        evaluate_run(config, None, ["fixture"], tmp_path / "a")
    with pytest.raises(FileExistsError):
        _frozen_queries(config, ["fixture-query"], predictor, tmp_path / "query-budget-1")
    assert len(names) == 3
