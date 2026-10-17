"""Fixed 2 x 2 condition/program experiments with bounded branch reservations."""
from __future__ import annotations

import ast
import builtins
import copy
from typing import Any, Callable

from .replay import run_branch, receipt_success, stable_digest, validate_intervention


def _template_ast(program: str) -> str:
    """Erase entity values and local spellings, retain API/operation structure.

    This deliberately undercounts ambiguous cases. A signature is a mechanical
    duplicate guard, not evidence of semantic independence across mechanisms.
    """
    names = {}
    reserved = set(dir(builtins)) | {"apis"}

    def local(name):
        if name in reserved:
            return name
        if name not in names:
            names[name] = f"local_{len(names)}"
        return names[name]

    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value=f"<{type(node.value).__name__}>"), node)

        def visit_Name(self, node):
            node.id = local(node.id)
            return node

        def visit_arg(self, node):
            node.arg = local(node.arg)
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            node.name = local(node.name)
            return self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_alias(self, node):
            if node.asname:
                node.asname = local(node.asname)
            return node

        def visit_ExceptHandler(self, node):
            if node.name:
                node.name = local(node.name)
            return self.generic_visit(node)

        def visit_Global(self, node):
            node.names = [local(name) for name in node.names]
            return node

        visit_Nonlocal = visit_Global

    return ast.dump(Normalize().visit(ast.parse(program)), include_attributes=False)


def mechanism_signature(plan: dict[str, Any]) -> str:
    """Conservative AST template identity, independent of model-chosen names."""
    programs = sorted(_template_ast(plan[key]) for key in ("r0", "r1"))
    queries = sorted({_template_ast(c["query"]) for c in plan["conditions"] if c is not None})
    return stable_digest({"version": 1, "programs": programs, "condition_queries": queries})


def run_joint_probe(base_request: dict[str, Any], plan: dict[str, Any],
                    run: Callable[..., dict[str, Any]] = run_branch,
                    reserve: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Execute the four prespecified cells; never invent missing checker truth.

    reserve() is called immediately before each branch, including failed ones.
    Returning False or raising stops further execution without refunding prior
    cells. A zero interaction is valid negative evidence, not a discovery.
    """
    result = {"schema_version": 1, "valid": False, "conditional_effect": False,
              "compensation_supported": False,
              "evidence_id": stable_digest({"request": base_request, "plan": plan}),
              "mechanism_signature": None,
              "mechanism_id": plan.get("mechanism_id"), "cells": {}, "reasons": [],
              "source": base_request.get("source"),
              "goal_interaction": {}, "interaction_by_goal": {}}
    try:
        if base_request.get("split") != "train":
            raise ValueError("Joint probes are train-only")
        if not isinstance(plan.get("mechanism_id"), str) or not plan["mechanism_id"]:
            raise ValueError("mechanism_id is required")
        explanations = plan.get("explanations", [])
        if not isinstance(explanations, list) or len(explanations) > 3:
            raise ValueError("At most three explanations are permitted")
        programs = [plan["r0"], plan["r1"]]
        if any(not isinstance(p, str) or not p.strip() for p in programs):
            raise ValueError("Both programs must be nonempty code")
        trees = []
        for index, program in enumerate(programs):
            compile(program, f"<r{index}>", "exec")
            trees.append(ast.dump(ast.parse(program), include_attributes=False))
        if trees[0] == trees[1]:
            raise ValueError("Programs must differ beyond whitespace/comments")
        conditions = plan["conditions"]
        if not isinstance(conditions, list) or len(conditions) != 2 or conditions == [None, None] or stable_digest(conditions[0]) == stable_digest(conditions[1]):
            raise ValueError("Two distinct condition specifications are required")
        for intervention in conditions:
            if intervention is not None:
                validate_intervention(intervention)
        result["mechanism_signature"] = mechanism_signature(plan)
        nonempty = [c for c in conditions if c is not None]
        if len(nonempty) == 2 and (nonempty[0]["query"] != nonempty[1]["query"] or nonempty[0]["expected_output"] == nonempty[1]["expected_output"]):
            raise ValueError("Both conditions must prove different values with the same query")
        # A null cell means the unmodified prefix condition. Give it the other
        # condition's query, so both observable conditions are measured.
        baseline_query = nonempty[0]["query"] if len(nonempty) == 1 else None
        for ci, intervention in enumerate(conditions):
            for ri, program in enumerate(programs):
                if reserve is not None and reserve() is False:
                    raise RuntimeError("Branch budget exhausted before next cell")
                request = copy.deepcopy(base_request)
                request.update(program=program, intervention=intervention)
                if intervention is None and baseline_query:
                    request["condition_query"] = baseline_query
                receipt = run(request)
                if not isinstance(receipt, dict):
                    raise ValueError("Branch runner did not return a receipt")
                result["cells"][f"c{ci}r{ri}"] = receipt
        cells = result["cells"]
        if any(not c.get("ok") or c.get("errors") or c.get("program_error") for c in cells.values()):
            raise ValueError("At least one cell failed execution; errors cannot establish necessity")
        if any(not c.get("checker_valid") for c in cells.values()):
            raise ValueError("Unknown or unaudited goal labels; conditional effect remains unknown")
        anchor = cells["c0r0"]
        for name, expected in (("task_id", base_request["task_id"]),
                               ("scenario_id", base_request["scenario_id"]),
                               ("identity", base_request["identity"]),
                               ("g_fix", base_request["g_fix"]), ("g_keep", base_request["g_keep"])):
            if anchor.get(name) != expected:
                raise ValueError("Receipt identity differs from the requested experiment")
        for cell in cells.values():
            if any(cell.get(k) != anchor.get(k) for k in ("task_id", "scenario_id", "split", "seed", "identity", "g_fix", "g_keep", "goal_mapping", "prefix_fingerprint", "goals_initial", "source", "evidence_origin")):
                raise ValueError("Cells do not share identical task, frozen identity, prefix and initial goal vector")
            if not cell.get("prefix_verified"):
                raise ValueError("Cell prefix replay is not verified")
            if any(v is None for v in cell.get("goals_initial", {}).values()):
                raise ValueError("Initial goal vector is incomplete")
            if cell.get("goals_before") != anchor.get("goals_before"):
                raise ValueError("Condition intervention changed the initial goal vector")
            if not cell.get("intervention_verified"):
                raise ValueError("Condition intervention lacks observable proof")
        for ci in (0, 1):
            left, right = cells[f"c{ci}r0"], cells[f"c{ci}r1"]
            if left.get("program_sha256") != stable_digest(programs[0]) or right.get("program_sha256") != stable_digest(programs[1]):
                raise ValueError("Receipt does not bind the fixed program")
            if left.get("before_fingerprint") != right.get("before_fingerprint") or left.get("goals_before") != right.get("goals_before") or left.get("query_output") != right.get("query_output"):
                raise ValueError("Sibling programs do not share the same condition state")
        if conditions[0] is None or conditions[1] is None:
            null_index = 0 if conditions[0] is None else 1
            changed_index = 1 - null_index
            before_observable = cells[f"c{null_index}r0"].get("query_output")
            changed_observable = cells[f"c{changed_index}r0"].get("query_output")
            if before_observable is None or before_observable == changed_observable:
                raise ValueError("Condition change was not observably distinguished from the prefix")
        goals = anchor["goal_mapping"]
        interaction = {}
        for goal in goals:
            values = [cells[key]["goals_after"].get(goal) for key in ("c0r0", "c0r1", "c1r0", "c1r1")]
            if any(type(v) is not bool for v in values):
                raise ValueError("Incomplete goal outcomes")
            interaction[goal] = (int(values[3]) - int(values[2])) - (int(values[1]) - int(values[0]))
        result.update(valid=True, conditional_effect=any(interaction.values()),
                      goal_interaction=interaction, interaction_by_goal=interaction,
                      compensation_supported=(any(receipt_success(cells[f"c{ci}r1"]) for ci in (0, 1))
                          and any(interaction.get(g, 0) for g in anchor["g_fix"] + anchor["g_keep"])),
                      evidence_origin=anchor.get("evidence_origin"), source=anchor.get("source"))
    except Exception as error:
        result["reasons"].append(f"{type(error).__name__}: {error}")
    return result
