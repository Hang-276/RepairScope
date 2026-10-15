"""Mock integration of the REAL pinned ReAct source, not live AppWorld.

When upstream has not been bootstrapped these source-level integration checks
skip. HTTP/accounting, runtime and selection unit tests remain self contained.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from b2_appworld.budget import Budget
from b2_appworld.config import load_config
from b2_appworld.llm import ChatResult
from b2_appworld.policy import PolicyError
from b2_appworld.runtime import (checkpoint_reasons, count_native_api_calls,
                                validate_live_predictor_metadata)
from b2_appworld.schemas import Candidate, Goal


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = [ROOT / "vendor/appworld/experiments/code/simplified/react_code_agent.py",
                     ROOT.parent / "work/upstream-appworld/experiments/code/simplified/react_code_agent.py"]
UPSTREAM_SOURCE = next((path for path in SOURCE_CANDIDATES if path.is_file()), None)


def live_metadata(kind="effect"):
    return {"kind": kind, "backend": "hf", "fixture_backend": False,
            "label_map": ["00", "01", "10", "11"], "harm_weight": 2,
            "cost_weight": .05, "cost_normalization": "max(1,max_candidate_pool_estimated_api_calls)"}


class RuntimeInvariants(unittest.TestCase):
    def test_timeout_failed_prefix_and_fifth_executed_step_checkpoint(self):
        self.assertEqual(checkpoint_reasons(2, "Execution failed. Traceback:\nExecution timed out", "pass"),
                         ["observed_error"])
        self.assertEqual(checkpoint_reasons(5, "okay", "pass"), [])
        self.assertEqual(checkpoint_reasons(6, "okay", "pass"), ["periodic"])

    def test_native_count_requires_valid_request_tracker_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "api_calls.jsonl"
            path.write_text(json.dumps({"method": "get", "url": "https://notes.local/a", "data": {}}) + "\n\n")
            self.assertEqual(count_native_api_calls(path), 1)
            path.write_text("malformed\n")
            with self.assertRaises(ValueError):
                count_native_api_calls(path)

    def test_live_guard_rejects_fixture_and_mismatched_objective(self):
        validate_live_predictor_metadata(live_metadata(), "effect")
        for change in ({"fixture_backend": True}, {"backend": "tiny"}, {"cost_weight": .5},
                       {"label_map": ["00", "10", "01", "11"]}, {"kind": "scalar"}):
            with self.assertRaises(ValueError):
                validate_live_predictor_metadata({**live_metadata(), **change}, "effect")


@dataclass
class Tokens:
    input_cache_miss: int = 0
    output: int = 0


@dataclass
class Usage:
    tokens: Tokens | None = None


@dataclass
class ExecutionIO:
    content: str


@dataclass
class Status:
    failed: bool = False
    message: str = ""


class Logger:
    def show_message(self, **kwargs):
        pass

    def start_task(self, world):
        pass

    def complete_task(self):
        pass


class UsageTracker:
    def reset(self, task_id):
        pass

    def add(self, task_id, usage):
        pass


class MinimalAgentDependency:
    """Only the base Agent's services are mocked; ReAct itself is real source."""
    @classmethod
    def register(cls, name):
        return lambda agent: agent

    def __init__(self, max_steps=50, **kwargs):
        self.max_steps = max_steps
        self.logger = Logger()
        self.usage_tracker = UsageTracker()
        self.step_number = 0
        self.messages = []

    def initialize(self, world):
        self.world = world
        self.messages = []
        self.logger.start_task(world)

    def log_usage(self):
        pass


class Template:
    def __init__(self, text):
        self.text = text

    def render(self, values):
        return self.text


@contextmanager
def import_real_react_with_mock_services():
    modules = {name: types.ModuleType(name) for name in (
        "appworld", "appworld.common", "appworld.common.io", "jinja2",
        "appworld_agents", "appworld_agents.code", "appworld_agents.code.common",
        "appworld_agents.code.common.usage_tracker", "appworld_agents.code.simplified",
        "appworld_agents.code.simplified.agent")}
    modules["appworld"].AppWorld = object
    modules["appworld.common.io"].read_file = lambda path: (
        "USER:\nDemo task.\nASSISTANT:\n```python\nfake_demonstration()\n```\n"
        "USER:\nActual task for the current supervisor.\n")
    modules["jinja2"].Template = Template
    modules["appworld_agents.code.common.usage_tracker"].Usage = Usage
    modules["appworld_agents.code.common.usage_tracker"].Tokens = Tokens
    agent_services = modules["appworld_agents.code.simplified.agent"]
    agent_services.Agent, agent_services.ExecutionIO, agent_services.Status = MinimalAgentDependency, ExecutionIO, Status
    official_name = "appworld_agents.code.simplified.react_code_agent"
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(official_name, UPSTREAM_SOURCE)
        official = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {official_name: official}):
            spec.loader.exec_module(official)
            ours_spec = importlib.util.spec_from_file_location("b2_appworld._agent_under_test", ROOT / "src/b2_appworld/agent.py")
            ours = importlib.util.module_from_spec(ours_spec)
            ours_spec.loader.exec_module(ours)
            yield official, ours


class FakeWorld:
    def __init__(self, folder):
        self.task_id = "mock_1"
        self.output_logs_directory = str(Path(folder) / "native_logs")
        self.task = types.SimpleNamespace(instruction="Create a task and preserve the note.",
                                          app_descriptions={}, supervisor={})
        self.executed = []

    def execute(self, code):
        self.executed.append(code)
        return "Execution failed. Traceback:\nExecution timed out" if len(self.executed) == 1 else "repaired"

    def task_completed(self):
        return len(self.executed) >= 2

    def save(self):
        path = Path(self.output_logs_directory) / "api_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"method": "post", "url": "https://app.local/mutate", "data": {}}) + "\n"
                                for _ in self.executed))


class FakeFactory:
    def __init__(self, world):
        self.world = world

    @contextmanager
    def create(self, *args):
        yield self.world


class ActorFixture:
    def __init__(self, *args, budget, **kwargs):
        self.budget = budget
        self.calls = []
        self.codes = ["partial_write_then_fail()", "unexecuted_actor_repair()"]

    def complete(self, messages, **kwargs):
        self.calls.append([dict(message) for message in messages])
        code = self.codes[len(self.calls) - 1]
        reservation = self.budget.reserve_chat(2, 10, "actor")
        self.budget.record_chat(reservation, 2, 4)
        return ChatResult(f"```python\n{code}\n```", 2, 4, 0.0)


class InvalidActorFixture(ActorFixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codes[1] = "if :"


@unittest.skipUnless(UPSTREAM_SOURCE, "Bootstrap the pinned upstream source for real ReAct mock integration")
class UpstreamReActIntegration(unittest.TestCase):
    goals = (Goal("g1", "Create", "Create a task"), Goal("g2", "Preserve", "preserve the note"))

    def test_true_subclass_and_executed_replacement_preserve_failed_prefix(self):
        with tempfile.TemporaryDirectory() as folder, import_real_react_with_mock_services() as (official, ours):
            self.assertTrue(issubclass(ours.RepairScopeAgent, official.SimplifiedReActCodeAgent))
            config = load_config()
            config["output_dir"] = folder
            config["budget"]["max_steps"] = 3
            agent = ours.RepairScopeAgent(config, "direct", "train", "mock", None)
            world = FakeWorld(folder)
            agent.factory = FakeFactory(world)
            pool = (Candidate("original", "unexecuted_actor_repair()", ("g1",)),
                    Candidate("local", "selected_repair()", ("g1",)))
            observed = []

            def proposals(client, instruction, history, goals, code, **kwargs):
                observed.extend(history)
                return pool

            selector = types.SimpleNamespace(select=lambda *args: pool[1])
            with patch.object(ours, "ChatClient", ActorFixture), patch.object(ours, "extract_goals", return_value=self.goals), \
                    patch.object(ours, "generate_candidates", side_effect=proposals), \
                    patch.object(ours, "DirectSelector", return_value=selector):
                agent.solve_task("mock_1")
            record = json.loads((Path(folder) / "mock/tasks/mock_1/trajectory.json").read_text())
            self.assertEqual(world.executed, ["partial_write_then_fail()", "selected_repair()"])
            self.assertEqual(record["checkpoints"][0]["prefix_length"], 1)
            self.assertEqual(record["checkpoints"][0]["state_id"], "mock_1:s001")
            self.assertTrue(record["checkpoints"][0]["executed"])
            self.assertEqual(record["actual_api_calls"], 2)
            history_text = json.dumps(observed)
            self.assertIn("partial_write_then_fail", history_text)
            self.assertNotIn("unexecuted_actor_repair", history_text)
            self.assertNotIn("fake_demonstration", history_text)
            self.assertIn("selected_repair", agent.messages[-1]["content"])
            self.assertNotIn("unexecuted_actor_repair", agent.messages[-1]["content"])

    def test_predictor_budget_prevents_second_environment_mutation(self):
        with tempfile.TemporaryDirectory() as folder, import_real_react_with_mock_services() as (_, ours):
            config = load_config()
            config["output_dir"] = folder
            config["budget"]["max_classifier_tokens"] = 10
            predictor = types.SimpleNamespace(metadata=live_metadata(), estimate_tokens=lambda *args: 20)
            agent = ours.RepairScopeAgent(config, "effect", "train", "mock", None, predictor)
            world = FakeWorld(folder)
            agent.factory = FakeFactory(world)
            pool = (Candidate("original", "unexecuted_actor_repair()", ("g1",)),
                    Candidate("local", "selected_repair()", ("g1",)))
            with patch.object(ours, "ChatClient", ActorFixture), patch.object(ours, "extract_goals", return_value=self.goals), \
                    patch.object(ours, "generate_candidates", return_value=pool):
                agent.solve_task("mock_1")
            record = json.loads((Path(folder) / "mock/tasks/mock_1/trajectory.json").read_text())
            self.assertEqual(world.executed, ["partial_write_then_fail()"])
            self.assertEqual(record["termination"], "budget_exhausted")
            self.assertEqual(record["budget"]["calls"], 2)

    def test_invalid_original_candidate_falls_back_to_official_execution(self):
        with tempfile.TemporaryDirectory() as folder, import_real_react_with_mock_services() as (_, ours):
            config = load_config()
            config["output_dir"] = folder
            agent = ours.RepairScopeAgent(config, "capture", "train", "mock", None)
            world = FakeWorld(folder)
            agent.factory = FakeFactory(world)
            with patch.object(ours, "ChatClient", InvalidActorFixture), patch.object(ours, "extract_goals", return_value=self.goals):
                agent.solve_task("mock_1")
            record = json.loads((Path(folder) / "mock/tasks/mock_1/trajectory.json").read_text())
            self.assertEqual(world.executed[-1], "if :")
            self.assertEqual(record["policy"]["fallbacks"]["candidate_invalid_original"], 1)


if __name__ == "__main__":
    unittest.main()
