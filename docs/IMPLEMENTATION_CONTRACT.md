# Internal implementation contract

Package: `b2_appworld`, Python >=3.11. Upstream AppWorld commit is `42b5bcf3cd334fee33f0c37c02070a9f5807add5`. All modules must import without AppWorld, Torch or network dependencies until the relevant live command is invoked. No synthetic fixture may be described as real AppWorld validation. Keep own files separate; use standard library + optional ML extras.

Shared schemas will be authored by root in `schemas.py`:

- `Goal(id: str, text: str, evidence_span: str, kind: str='state_write')`; `Goal.from_dict(d)`, `.to_dict()`.
- `Candidate(id: str, code: str, target_goal_ids: tuple[str,...], estimated_api_calls: float=1.0)`; same methods.
- `Example(task_id: str, scenario_id: str, state_id: str, split: str, instruction: str, history: list[dict[str,str]], goals: tuple[Goal,...], candidate: Candidate, labels: tuple[int,...], source: str='natural', before_fingerprint: str='', metadata: dict=...)`. Labels are `0=00,1=01,2=10,3=11,-100=unknown`. Same methods. Unknown labels masked.
- Leakage guard `ensure_research_split(split)` allows train/dev only. Runtime observations must not contain evaluator/ground_truth fields.
- JSONL serialized Example keys match fields. `read_jsonl(path)`/`write_jsonl(path, rows)` in `io.py` accept mapping rows; write atomically.
- `model_inputs.py::build_effect_text(instruction,history,goals,candidate)` constructs prefix with full G/r plus observation history, then target goal query slots (ML agent owns precise tensor representation). Shared label order fixed above.

Predictor interface (ML agent): `EffectPredictor.predict(instruction, history, goals, candidates) -> list[list[list[float]]]` shape K,M,4. `ScalarPredictor.predict(...) -> list[float]`. Torch/HF predictors load local checkpoint + record processed tokens/time. Define `load_predictor(checkpoint, device='cuda', quantized=True)`. Metadata must validate model kind and loss weights. Checkpoint CLI training data train only; dev validation only; scenario holdout list applies to all state/candidate rows.

Core service interface (core agent): `ChatClient(base_url,model,api_key_env='B2_ACTOR_API_KEY',timeout=120, tokenizer=None)`, `.complete(messages,max_tokens=...,temperature=0,seed=100) -> ChatResult(content,input_tokens,output_tokens,seconds)`; expose usage. `extract_goals(client,instruction,...)`; `generate_candidates(client,instruction,history,goals,original_code,...)`; direct choice and checklist predictors. Root will adapt exact signatures after inspect.

Environment integration (collection agent): `AppWorldFactory(root,seed=100)` creates one world per context, imports lazily; `create(task_id,experiment_name,split,full_ground_truth=False)`. Never full GT on test. `dataset_manifest(root)` validates versions and ID/scenario isolation with source expectations. `collect_from_trajectories(trajectory_dir, mapping_file, output, root,...)` train/dev only; mapping includes goals + evaluator requirement IDs, never forwarded to runtime; separate child process for each replay; fingerprints must match before labels. Build mapping command may call goal extractor but preserve instruction evidence, use train/dev GT for labels only. Expose mapping structure clearly to root.

Online integration root owns `agent.py`, `runner.py`, `cli.py`. Methods baseline/direct/scalar/effect/rule, no runtime oracle. Hook original SimplifiedReActCodeAgent, fixed checkpoints (error/every5/precompletion); append only actually executed code, auxiliary generation usage included. Native test gets aggregate TGC/SGC only. Protected raw data/logs gitignored.

Ownership:
- Core agent: `llm.py`, `policy.py`, `selection.py`, `budget.py`, core tests.
- ML agent: `learning.py`, `model_inputs.py`, `training.py`, `metrics.py`, ML/metrics tests.
- Collection agent: `environment.py`, `collection.py`, `datasets.py`, mapping/replay tests.
- Root: schemas/io, config/CLI/runner/agent, scripts, docs, packaging, integration tests, GitHub push.
