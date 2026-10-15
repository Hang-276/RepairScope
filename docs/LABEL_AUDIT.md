# Goal-label audit before main training

RepairScope predicts high-level goals grounded in the original task instruction.
AppWorld evaluator assertions are **labeling instruments**, not automatically
independent goals. A valid JSON mapping and a successful replay are necessary but
do not prove that a checker group measures its intended goal.

## Data boundary

- Build and inspect goal/checker mappings only for `train` and `dev`.
- Never use test instructions, requirement strings, checker traces, private data,
  solutions, fine-grained labels or per-goal diagnostics to tune the method.
- At native test evaluation, report only aggregate AppWorld TGC/SGC and execution
  cost. The actor sees instruction, permitted API observations and executed
  history. It never receives the mapping file or checker results.
- Keep trajectory files, label mappings, raw evaluator outputs and model training
  data in ignored private experiment directories. Public reports can include
  aggregate coverage and audited anonymized train/dev examples.

## Mapping format and meaning

The private JSON file has `schema_version: 1` and a `tasks` object keyed by task
ID. Each task entry contains `split`, `instruction_sha256`, the fixed public
`goals` list, and `goal_requirements`:

```json
{
  "g1": {"requirements": ["exact evaluator requirement string"], "complete": true},
  "g2": {"requirements": [], "complete": false}
}
```

For `complete: true`, **all** mapped requirements must pass for the goal to be
satisfied. This conjunction is valid only if it is sufficient and necessary for
that goal at the checkpoint being labeled. The mapper must preserve the original
instruction evidence and must not invent new goals from hidden tests. Final
answer goals, trajectory goals, nonfactorable checks, incomplete mappings and
missing evaluator statuses yield unknown labels (`-100`). One requirement shared
across multiple goals makes all those groups incomplete automatically: a joint
check cannot reveal which individual component was achieved.

An initial LLM mapping is a proposal. `complete: true` is not a semantic
certificate. In particular, a requirement that simultaneously checks a target
update and preservation of unrelated records may be unusable as an independent
target-goal checker. Leave it unknown unless the task-level interpretation is
justified without changing the research goal definition.

## Audit procedure

1. Inspect every train/dev scenario template before the main run, plus task
   variants where checker wording or goal extraction differs. Check that the
   fixed goals are final conditions with exact instruction evidence, rather than
   API substeps or a renamed list of assertions.
2. For each mapped group, record why its requirements jointly measure that goal.
   Inspect train/dev evaluator implementation when wording is ambiguous; do not
   send implementation code, private values or traces to the actor/predictor.
3. Inspect naturally occurring successful and failed before/after states. Where
   needed, make small **train-only labeled diagnostic perturbations**: modify
   just the goal state, retain unrelated goals, then modify only an unrelated
   state. The assigned checker should react to the intended condition and avoid
   coupling independent goals. These perturbations are audit evidence, not
   naturally occurring recovery results.
4. Record reviewed task IDs, goal IDs, mapper/model version, requirement-set hash,
   inclusion/exclusion decisions and reason. Keep an initial automated mapping
   snapshot so changes remain reviewable. Freeze the audited mapping before
   test evaluation. If a mapping changes, regenerate affected labels.
5. Run the collection-quality report. Inspect class counts `00/01/10/11`, unknown
   rate, per-task goal coverage, complete four-candidate pools, states with `01`,
   states with `10`, and their overlap. Inspect rejection receipts. A class can
   be present but too rare, template-specific or incorrectly mapped.
6. Before training, set and record a data sufficiency decision using independent
   scenarios, observed class balance and audited label reliability. The report
   intentionally does not invent a universal minimum sample count.

## Replay validation

The workflow gate reads `runs/labels/mapping_audit.json`. Its minimum fields are
`goals_reviewed` (an integer >=30) and `issues_resolved` (boolean true). These
fields must summarize actual reviewed task/goal records, mapper/model revision,
mapping hashes, findings and corrections saved alongside them; they are not a
permission token or a substitute for semantic checking. Never set them merely
to bypass `scripts/check_collection.py`. Preserve the pre-audit mapping and
regenerate labels after corrections.

Every reference and candidate replay starts a fresh spawned process and world,
uses a distinct experiment directory, and reexecutes the complete code prefix.
The collector preserves blocks whose execution failed: they may already have
written database records or defined Python variables. It never restores a DB
snapshot into a different Python shell. A candidate is accepted only if prefix
observations, semantic ordered DB changes and before-checker statuses match its
reference replay. Mismatches are rejected with a state/candidate receipt.

The DB digest retains SQL values, IDs and task timestamps. It drops only the
instrumentation envelope (process IDs, experiment paths) surrounding SQL change
records. Exact-prefix matching does not prove that every environment API is
deterministic; report rejection rates and inspect systematic failures.

## Evidence labels and reporting

`source=natural` means a recorded actor trajectory; `synthetic_fault` means an
explicitly introduced train/dev fault; `synthetic_fixture` means a fake-world
software test. Collection receipts also state whether an injected fake factory
was used. Never relabel fixtures as natural data or aggregate fixtures into real
AppWorld performance. Keep synthetic-fault results separate from natural-error
results unless a predeclared protocol explicitly calls for a combined analysis.

The included fake-world tests establish software properties such as isolation,
unknown masking, partial-write preservation and mismatch rejection. They do not
establish AppWorld compatibility, GPU memory use, label quality or research
effectiveness. Obtain and retain a real-data smoke receipt before claiming the
pipeline runs on AppWorld, and a complete frozen evaluation receipt before
claiming benchmark results.
