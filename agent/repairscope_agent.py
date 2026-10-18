"""RepairScope integration with WMA's candidate stage and five-element contract.

This module can be imported without Playwright, LangChain or browser_env. Pass an
already initialized WMAgent, or call from_wma_config on a configured server.
"""
from __future__ import annotations

import copy
import json

from repairscope.actor import normalize_action, public_state
from repairscope.control import score_prediction
from agent.wma_core import generate_candidates, stable_rank


class WMACandidateAdapter:
    """Directly reuse WMA's prompt constructor, policy_llm and action parser."""

    def __init__(self, wma_agent):
        self.wma = wma_agent
        self.calls = 0

    def generate_native(self, trajectory, intent, meta_data, count=4):
        wma = self.wma
        state_info = trajectory[-1]
        page = state_info.get("info", {}).get("page")
        raw_url = getattr(page, "url", state_info.get("url", ""))
        prompt = wma.generate_prompt(wma.action_prediction_template)
        previous = "\n".join(meta_data.get("action_history", [])) or "None"
        values = {"objective": intent, "url": wma.map_url_to_real(raw_url),
                  "observation": state_info["observation"]["text"], "previous_action": previous}
        response, selected = generate_candidates(wma.policy_llm, prompt, values,
            parse=lambda generation: wma.extract_action(generation.text), count=count)
        self.calls += 1
        return [{"name": "wma", "arguments": {}, "wma_action": item["action"],
                 "prob": item["prob"], "cost": 1.0, "raw_prediction": item["raw"].text} for item in selected]

    def materialize(self, action):
        """Return exactly the browser action dictionary WMA next_action returns."""
        from browser_env.actions import create_id_based_action, create_none_action, create_playwright_action
        try:
            constructor = create_playwright_action if self.wma.action_set_tag == "playwright" else create_id_based_action
            native = constructor(action["wma_action"])
        except Exception:
            native = create_none_action()
        native["raw_prediction"] = action.get("raw_prediction", "")
        native["prob"] = action.get("prob", 1.0)
        return native


class RepairScopeAgent:
    """Composition wrapper replacing WMA state/value LLMs with RepairScope.

    ``state_adapter`` is mandatory: it maps the legal browser observation into the
    same public structured field schema used by Predictor training. No database or
    task checker is read by this adapter. AppWorld uses RepairController instead.
    """

    def __init__(self, wma_agent, predictor, state_adapter, goal_adapter=None, config=None):
        self.candidates = WMACandidateAdapter(wma_agent)
        self.predictor = predictor
        self.state_adapter = state_adapter
        self.goal_adapter = goal_adapter or (lambda intent, meta: meta.get("goals", []))
        self.config = config or {}
        self.history = []

    @classmethod
    def from_wma_config(cls, wma_config, predictor, state_adapter, goal_adapter=None, config=None):
        from agent.world_model_agent import WMAgent
        return cls(WMAgent(**wma_config), predictor, state_adapter, goal_adapter, config)

    def next_action(self, trajectory, intent, meta_data, images=None, branching_factor=4):
        generated = self.candidates.generate_native(trajectory, intent, meta_data, branching_factor)
        if not generated:
            raise ValueError("WMA policy generated no candidate actions")
        state = public_state(self.state_adapter(trajectory[-1]))
        goals = self.goal_adapter(intent, meta_data)
        predictions, utilities = [], []
        for candidate in generated:
            record = {"state": state, "action": normalize_action(candidate), "history": copy.deepcopy(self.history),
                      "goals": goals, "step": len(self.history)}
            prediction = self.predictor(record)
            predictions.append(prediction)
            utilities.append(score_prediction(state, prediction, goals, candidate, len(self.history), self.config))
        selected = stable_rank(generated, predictions, utilities)[0]["action"]
        self.history.append({"state": state, "action": normalize_action(selected)})
        self.history = self.history[-8:]
        return (self.candidates.materialize(selected), [a["wma_action"] for a in generated],
                [json.dumps(p["next_state"], ensure_ascii=False) for p in predictions],
                [u["score"] for u in utilities], [[json.dumps(u)] for u in utilities])

    def reset(self, test_config_file=None):
        self.history = []
        return self.candidates.wma.reset(test_config_file)

    def set_action_set_tag(self, tag):
        self.candidates.wma.set_action_set_tag(tag)
