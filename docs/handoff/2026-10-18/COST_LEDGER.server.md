# Cost ledger and budget status

This ledger separates **actual charged environment units** from **estimates**.
It does not turn the old 300-unit supplement cap into an approved formal budget.
All formal acquisition/training/test figures are proposals in
`FORMAL_DATA_PLAN.md` unless explicitly marked actual.

## A. Actual charged environment units (existing receipts)

| stage | units | receipt/evidence | notes |
|---|---:|---|---|
| phone natural smoke | 10.0 | `v2_smoke/runs/natural/summary.json` | 5 steps × 2 |
| failed CF (condition newline mismatch) | 10.0 | error stdout, no retained summary | charged cost from the failed run |
| superseded 2-cell CF | 27.0 | command stdout; output overwritten before 4-pair rerun | receipt file no longer present |
| phone 4-pair CF | 96.0 | `v2_smoke/runs/cf/summary.json` | 8 cells |
| **smoke subtotal** | **143.0** | | |
| follow supplement CF | 132.0 | `v2_supplement/runs/cf/summary.json` | 3 pairs / 12 cells |
| old c1 (no intervention) | 28.0 | `v2_supplement/runs/phone_c1/summary.json` | retained as control, not final |
| c1 intervention | 28.0 | `v2_supplement/runs/phone_c1_intervention/summary.json` | final Boundary half |
| failed c1 (`show_profile` argument error) | 9.0 | error stdout, no retained summary | charged cost from failure |
| **bounded supplement subtotal** | **197.0** | | |
| **cumulative actual** | **340.0** | | |

- The 2-cell CF and failed-c1 rows are **not** backed by retained summary files;
  they must remain explicitly caveated in MacBook accounting.
- The supplement subtotal 197.0 is based on those charges; it is below the old
  300-unit supplement cap, but the cap itself is not a formal budget.
- This export round: **0** environment calls, **0** Actor calls, **0**
  optimizer steps. Only file copying/hashing and Markdown/JSON summary work.

## B. Actor / GPU

- Actor calls in all recorded collection runs: **0** (fixed public plans/static
  tokens; Actor service was not started).
- GPU work: smoke/continuation 5-update training and timing; measured
  15.7 s/update, 78.68 s for 5 updates on 15 records (RTX 4090, BF16). This is
  an engineering timing measurement, not a formal training budget.
- No formal Ours, no tmux, no formal checkpoint/evaluation.

## C. Proposed formal acquisition/evaluation (estimates, not approved)

See `FORMAL_DATA_PLAN.md` §3–§3.4 and `FORMAL_DATA_MANIFEST.json`.

- Candidate stage environment cap: **18,114 units** (data 1,095 + 219 retry
  + test 16,800), Actor cap **33,750 calls**.
- Candidate training: ~270 optimizer updates, ~1.2 h (15.7 s/update basis).
- Candidate Ours once data frozen: ~4–8 h; full-coverage estimate 12–18 h.
- No formal budget file exists; no acquisition or training was started. The
  estimates are review inputs, not authorization.

## D. Missing cost facts

- Exact natural/CF costs for the remaining 78 train / 54 dev unbound tasks:
  **not measured**; no legal plans yet.
- Exact test_normal controller calls/Actor generations: **not measured**; only
  protocol upper bounds (168×100 env; 168×50×4 Actor) are shown.
- Dev CF evaluation-only costs: **not measured**; candidate estimate only.
