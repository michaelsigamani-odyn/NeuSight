# NeuSight calibration on Qwen 7B — investigation context

Status as of 2026-09-18. NeuSight forward-latency predictions for Qwen-7B on A100 were
~9× too high. Root causes are identified below, with evidence, open questions, and the
agreed patch order. Read the whole file before touching the predictor or the proxy generator.

## Headline finding

The LINEAR (and BMM) predictors are trained exclusively on fp32 sgemm kernels and use a
non-dtype-specific `SingleFLOPs` feature set to the A100 fp32 peak (19.5 TFLOP/s).
The measured runs are bf16 tensor-core. The predictor has never seen a tensor-core kernel,
so it cannot predict bf16 matmul utilisation — this is the dominant error, not a
hardware-calibration constant.

## Evidence

### 1. Linear predictor anchored to fp32 (validated in code and config)
- `MLP_WAVE/LINEAR/config.json` uses feature `SingleFLOPs` (not dtype-specific).
- A100 device config: `SingleFLOPs: 19492` (≈ fp32 dense peak).
- `data/dataset/train/linear.csv` rows are legacy sgemm/float kernels only.

What-if (no retrain), 7B, `SingleFLOPs` → 312000 (bf16 dense peak):
- pred/measured before: 9.60, 9.19, 8.48
- after: 2.47, 1.94, 1.86
- CSV: `~/Desktop/singleflops_bf16peak_whatif.csv`

**Interpretation — important:** a 16× feature change produced only a ~4–5× prediction
change. That is the MLP extrapolating outside its training range, not a physical
residual. The remaining ~1.9–2.5× must NOT be reported as "error after fixing FLOPs";
it may move materially on retraining. Tensor-core utilisation also has different shape
dependence from SIMT fp32 (tile quantisation to multiples of 8/16/64, wave quantisation
across 108 SMs, steep fall-off at small M) that sgemm rows cannot teach.

### 2. `getitem` is overcounted
- Per-decoder-layer predicted `getitem` = 1.97 ms (≈5.6× the entire BMM cost of 0.35 ms).
- Summed over 28 layers ≈ 55 ms of the full prediction.
- What-if removing it (on top of the bf16 what-if): ratio → ~1.30–1.32.
- CSV: `~/Desktop/qwen7b_getitem_whatif.csv`
- This is an op-classification / cost-model issue.

**Caution:** do not zero it purely because it moves the ratio toward 1.0. `getitem` on a
view is free; `getitem` with advanced indexing (rotary cos/sin gathers, KV-cache slices
that force a copy) materialises and has real bandwidth cost. Profile the real 7B trace
(`torch.profiler`) first and classify accordingly.

### 3. 63.1 ms vs 11.66 ms Linear — NOT actually resolved
- `neusight_7b_fullgraph_perop_fw_ms.csv`: Linear = 63.11 ms (whole graph incl. lm_head).
- `neusight_7b_decoderlayer_segment_perop_fw_ms.csv`: Linear = 11.66 ms
  (segment `transformer_h_0_ln_1 → add_15`, one decoder layer).
- Previous claim: "both consistent with scope." Arithmetic says otherwise:
  - 28 layers × 11.66 ≈ 326 ms expected for full graph, observed 63.1.
  - 63.1 − 11.66 ≈ 51 ms attributed to lm_head; lm_head (~545M params, 152k×3584) is
    ~2.3 decoder-layers' worth of linear params, so ~27 ms expected.
- Item 2 assumes a 28× per-layer sum; item 3's files evidently do not. The two items use
  different aggregation conventions (per-unique-node? per-op average? total?).
- **Action:** define and record what one row in the per-op CSVs means (filename or header)
  before any further cross-file comparison.

### 4. File duplication
`neusight_7b_layer_perop_fw_ms.csv` and `neusight_7b_fullgraph_perop_fw_ms.csv` contain
identical data (Linear 63.11, EMBEDDING and `output` rows present) in different sort
order. One is mislabeled. Fix before plotting.

### 4.8 `getitem` cap coverage/scope regression (found + fixed)
- Regression signature before fix: `getitem` predicted nearly flat at ~56–57 ms on
  uncovered shapes and ~13.6 ms on `(512,1)`, dominating the total miss.
- Root cause: cap map had limited shape coverage and cap scope interacted with
  layer-replication semantics; uncovered shapes could bypass intended capping.
- Fix applied in `neusight/Prediction/predictor.py`:
  - switched to explicit full-model cap map by shape,
  - always apply a non-null fallback cap,
  - convert to per-template cap only when `single_layer=True`.
- Post-fix reconciliation: `getitem` residual drops to low single-digit share of total
  gap on all three eager shapes; dominant residual category shifts to
  `elementwise_copy`.
- Durable lesson: a fix validated on one shape can silently fail on others unless shape
  coverage is explicit and versioned in the reconciliation table.

### 5. Proxy mismatch for 14B / 32B / 72B (confirmed)
Proxy uses GPT-2 architecture (`GPT2LMHeadModel`, not `Qwen2ForCausalLM`):
- No `num_key_value_heads` → behaves as MHA, not GQA (k/v projections overcounted,
  up to 8× on 72B).
- No explicit `n_inner` → GPT-2 default `4 × n_embd`:
  - 14B: HF `intermediate_size` = 13824, proxy implied 20480
  - 32B: HF 27648, proxy implied 20480
  - 72B: HF 29568, proxy implied 32768
- Cross-model factor drift is therefore expected independent of calibration.

**Additional correction:** setting `n_inner = intermediate_size` still undercounts.
GPT-2 MLP is two matrices (h→n_inner→h); Qwen is gated SiLU with three (gate, up, down).
To match linear FLOPs use `n_inner ≈ 1.5 × intermediate_size`, or a proxy with three MLP
projections. Check whether the **7B** run is also on a GPT-2 proxy: if so, default
4×3584 = 14336 undercounts Qwen's 18944 MLP by ~2× in FLOPs, meaning the 9×
overprediction occurs *despite* undercounting the largest term — the FLOPs-path
error is larger than the what-if suggests.

### 6. Plot / measurement files (already on Desktop)
- `real_measurement_comparison.csv` — real-only, CI95, repeats/warm-up
- `plot_error_series_real_only.csv`
- `plot_predictions_only_nonreal.csv`
- `plot_ready_predictions_vs_real.csv` — contains `error_eligible` logic
Strict real-vs-nonreal separation is in place; keep it.

## Reference model shapes (Qwen2.5, HF configs)
| Model | hidden | intermediate | layers | heads / kv_heads |
|---|---|---|---|---|
| 7B  | 3584 | 18944 | 28 | 28 / 4 |
| 14B | 5120 | 13824 | 48 | 40 / 8 |
| 32B | 5120 | 27648 | 64 | 40 / 8 |
| 72B | 8192 | 29568 | 80 | 64 / 8 |

## Agreed patch order

0. **Measurement sanity check first.** Confirm the measured 7B run is bf16 with tensor
   cores engaged (cuBLAS `*_tensorop_*` kernels in the profiler). If any linears fell back
   to fp32 via autocast/default dtype, part of the gap is on the measurement side and
   the fixes below will overshoot.
1. **Retrain LINEAR on bf16 tensor-core data with small-M weighting.** Regenerate training
   sets with bf16 `torch.matmul` across NeuSight's M/N/K tile distribution on A100, but
   oversample Qwen-relevant small-M rows (`M in {128..1024}`) so they are not drowned out
   by large-M examples. Add a dtype-specific feature (`SingleFLOPsMM`, bf16/TC peak) and
   route LINEAR config to it. Keep fp32 rows as a separate dtype class if fp32 prediction
   is still needed.
   `SingleFLOPsVEC` path stays as-is (elementwise ops are bandwidth-bound; dtype only
   changes bytes moved). Swapping the feature value alone is NOT sufficient.
2. **BMM triage before retrain spend.** Under FA2, `aten::bmm` is absent in trace; treat
   BMM as eager-diagnostic only unless reconciliation shows eager BMM residual >5 ms.
   If residual <=5 ms, skip BMM retraining and prioritize flash-op modelling.
3. **Fix proxy generator.** Set `n_inner` from HF `intermediate_size` with the 1.5× gated-MLP
   correction (or add a three-projection MLP). Document that GPT-2 proxy cannot represent
   GQA or Qwen head_dim; make no model-level claims for 14B/32B/72B until a native Qwen
   trace path exists.
4. **`getitem`** — profile on real trace, then either (a) map view-only getitem to a
   zero-cost class, or (b) model materialising getitem as a bandwidth-bound VEC-like op.
   Do this AFTER step 1 so its contribution is attributed against a properly calibrated
   Linear, not an extrapolated one.
5. **Aggregation convention.** Fix the file mislabel (§4), define per-op CSV semantics
   (§3), and re-derive the 63.1 vs 11.66 comparison under one convention.
6. **Add flash attention op path, then gate on FA2 rows.** Add a flash-op latency term
   (roofline + fitted eta) before production gating. Production gate rows are FA2:
   `(512,1)=48.17 ms`, `(2048,1)=150.86 ms`, `(512,8)=283.53 ms`.
7. If needed, run eager 3-point only as an intermediate LINEAR diagnostic, not as a
   production-readiness gate.

## Reporting rules
- Only real measurements (with CI95) feed error metrics; predictions for non-measured
  configs are plotted separately and never enter error series.
- Do not report post-what-if residuals as calibrated error; only post-retrain numbers.
- State assumptions (proxy architecture, aggregation scope, dtype) alongside every
  pred/measured ratio.
- At low batch and short sequence (notably 7B `(512,1)` on HF eager), GPU idle gaps
  between small kernels can contribute several ms; treat this as an explicit
  `not_modelled` bucket unless/until host/runtime scheduling overhead is modelled.
- Do not use per-module hook sums as the measured reconciliation baseline: CUDA-event
  module instrumentation introduces sync overhead and can inflate component totals
  beyond end-to-end. Use kernel-level profiler device-time totals for measured columns.
- Report MM holdout error binned by M (`<=512`, `512-2048`, `>2048`) alongside aggregate
  MAPE. Use the small-M bin as the decision signal for 7B closure, not aggregate alone.

## Pre-registered expectation
- Baseline to beat for FA2 gate rows is currently over-prediction at approximately
  `(1.62, 1.33, 1.48)` for `(512,1)`, `(2048,1)`, `(512,8)`.
- Expected direction is intentionally **TBD from the reconciliation table** until
  predictor-version and getitem-mode reconciliation is complete.
- After small-M LINEAR retrain and flash-op integration, expected movement is reduction
  versus this baseline, but exact post-retrain ratios remain intentionally TBD.
- If observed movement is not a clear reduction from baseline on all three rows, treat it
  as a third systematic term (not noise) and isolate it explicitly in reconciliation.

## Reconciliation guardrail (must pass before closure artifact)
- Every predicted row must include `predictor_version` and
  `getitem_mode in {legacy_ratio, absolute_cap}`.
- Eager baseline used for closure scoring must be generated in one fresh rerun block
  (same measurement run-id for measured columns, no copied predicted values).
- If `legacy_ratio` and `absolute_cap` both appear, keep both, label both, and never
  mix them in one scoring series.

## Current status update (2026-09-18, post small-M LINEAR retrain)
- Small-M LINEAR augmentation + retrain was executed and produced a **null result** on
  eager baseline rows (no meaningful error reduction):

| Shape | Pre-retrain predicted ms | Pre ratio | Post-retrain predicted ms | Post ratio | Delta |
|---|---:|---:|---:|---:|---:|
| (512,1) | 65.45 | 1.35 | 66.52 | 1.37 | +1.6% |
| (2048,1) | 320.40 | 1.33 | 323.96 | 1.34 | +1.1% |
| (512,8) | 481.70 | 1.48 | 487.56 | 1.50 | +1.2% |

- Decision: do **not** run additional retraining loops (LINEAR/BMM/other) until the
  kernel-level reconciliation table is produced and identifies the dominant residual term.
- Working hypothesis update: residual over-prediction is likely dominated by components
  not corrected by this LINEAR retrain (candidate buckets: eager attention/BMM term,
  launch/idle-gap overhead, or aggregation mismatch).
