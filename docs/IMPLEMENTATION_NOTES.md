# Fixed implementation decisions

These notes resolve differences between the earlier design proposal and the
implemented first version. Configuration and executable checks remain the source
of truth; no result is claimed here.

- Checkpoint selection minimizes **unweighted validation loss**, with patience
  counted in epochs. Use official dev or a whole-scenario holdout from train.
  Native dev TGC/SGC does not select predictor checkpoints.
- Both primary training methods default to `require_complete_labels=true`, so
  effect and scalar heads use the same candidate records. The effect head can
  mask unknown goals only in an explicitly separate diagnostic with this flag
  false; scalar targets always require complete labels. Filtering counts are
  saved. Candidate cost normalization is computed before this filtering.
- HF training and held-out evaluation accept only `source=natural`. The CPU tiny
  backend accepts only `source=synthetic_fixture` and is excluded from live
  benchmarks. Synthetic faults are outside this first main-experiment recipe.
- Utility is mean recovery probability minus twice mean harm probability minus
  `0.05 * estimated_calls / max(1,max_pool_estimated_calls)`. Effect/scalar inputs
  share the complete task, goals, candidate, normalized cost, utility weights,
  retained visible history and all goal queries. Scalar prediction pools the
  same query hidden states. All goal predictions use one forward per candidate.
- Main configs may set encoder/head LR separately (`lr=1e-4`, `head_lr=1e-3`),
  cosine schedule and `warmup_ratio=0.03`. Library defaults preserve `lr=2e-4`
  and `head_lr=lr`; tiny CPU fixtures default to zero warmup. Optimizer settings
  and package/model revisions are saved in checkpoint metadata.
- History truncation drops only complete old messages. The full instruction,
  full goals, full candidate, queries and latest message must fit. Overflow is
  an explicit error. No evaluator fields or labels enter model input.
- Nonempty training output directories are refused. A best checkpoint stages
  adapter/encoder, head, tokenizer and metadata before replacing an earlier best
  within that same run. There is no optimizer-state resume in this version.
- Labeled mechanism metrics (four-state F1, harm recall, calibration, candidate
  regret and paired scenario bootstrap) remain separate from official TGC/SGC.
