# Collection schemas and server adaptation

`example_cf_plans.json` is fully executable with `environment.backend=fixture`.
It produces four independently executed cells and a genuine conditional side
effect inside that synthetic environment. It establishes engineering behavior,
not AppWorld efficacy. `example_goal_map.json` uses the same fixture.

For the server's AppWorld data, build mappings and plans once as data preparation:

1. Point `environment.root` to the official data root and retain its official
   `data/datasets/train.txt` membership. Use the installed AppWorld revision from
   the experiment manifest. Natural collection exposes public API docs in the
   frozen actor context; it records exact code, outputs, fingerprints and costs.
2. Read each selected task's public API schemas in `env.metadata['api_docs']`.
   Fill `environment.observation_map` with the JSON shape demonstrated by
   `example_observation_map.json`: only returned fields are mapped. Set actual
   response keys, entity/table key names, ID fields, and any public relations.
   `path` supports dotted keys and `*` list expansion. Unseen values remain
   unknown; this mapping never queries a database to fill a public input.
3. Define public goal predicates as task-ID -> goal-list in `data.goal_map`.
   Specify protection start/end steps explicitly for temporary requirements.
   Unknown observable predicates stay unknown. Offline audited fields can label
   the same predicate without entering the prediction input.
4. Select a real prefix from natural transitions with
   `collection.extract_prefix(records, stop_step)`. A real prefix requires exact
   action/output/fingerprint receipts; no manual state editing or checkpoint
   reuse across live worlds is supported.
5. Replace fixture actions with actual permitted API operations from those docs.
   Each condition is a list of legal API actions and a public read query plus its
   exact expected output (including newline). Read queries are checked against
   global write fingerprints. Choose a valid reference action; no-op is accepted
   only when the task interface actually permits it and time/randomness remain
   comparable. With only one condition use `{'c0': ...}` for a two-cell plan;
   boundary masks require all four valid aligned cells.
6. Run the standard counterfactual acquisition once. Every branch starts a fresh
   process and replays the full prefix. A mismatched prefix or failed condition
   stops acquisition with a failure receipt. Executed invalid target actions stay
   in receipts with `action_valid=false` and charged cost. Missing/invalid cells
   never become zero-valued labels. Do not use official dev/test tasks here.

The `execution_verified:false` tag on the observation-map example records its
status; the collector does not reject plans on that tag or require human approval.
It validates actual execution, split membership, prefix identity, conditions and
field alignment. The example API name is present in upstream tests; response and
table mappings must be read from the installed task rather than guessed.

Audit snapshots use `app.table[primary-key].column`. Each execution's declared
cost is `one code block + number of actual public API calls`; replay, conditions,
verification queries and failed calls are included. Wall time is also retained.
SQLite snapshots are offline train/dev outputs. Official evaluation runs in a
fresh process so AppWorld's process-global `CachedDBHandler` cannot alias a live
world. Single-task SGC is left null; evaluation must aggregate all planned tasks
in each scenario according to the official rule.
