# Project Issues And Resolutions Log

This file is the persistent register for data problems, decisions, fixes, and verification results.
Update it every time a new issue appears or an issue is resolved.

## Reference Rules (from project guides)

- Minimum history for TFT training: keep stocks with at least 500 trading days.
- Key stocks target: key stocks should have strong history (guide target is 1000+ days for major names).
- Bad OHLC rows: remove impossible rows instead of unsafe imputation.
- Indicator warmup is expected: fewer indicator rows than price rows is normal due to MA/RSI windows.

## Decision Framework For Missing Years

Use this order:

1. Verify if missing years are expected (stock listed later, suspended, delisted).
2. If expected: keep stock if it still has >=500 trading days.
3. If unexpected and affects key stocks: recover missing source files and reload that stock.
4. If stock remains very sparse or highly illiquid after checks: exclude from model training.

Do not remove a company only because early years are missing if it still satisfies coverage and quality rules.

## Issue Register

### ISSUE-001: Coverage Heatmap Shows Many Missing Years
- Date opened: 2026-03-19
- Date decision confirmed: 2026-03-19
- Source: notebooks/data_quality_coverage_issues.csv
- Evidence summary:
  - Total issue cells: 250
  - missing_year: 219
  - partial_year: 31
  - Key-stock issue cells: 5 (all for CARTHAGE CEMENT in years 2016-2020)
- Interpretation:
  - Most missing-year cells are on non-key stocks and often match listing/suspension behavior.
  - For key stocks, only CARTHAGE CEMENT has missing early years; later years appear complete.
- Risk to TFT accuracy:
  - Moderate if key stocks are missing long recent windows.
  - Low if missing years are historical and each kept stock still has enough total history.
- Recommended action:
  - Keep current strategy: do not drop all companies with missing years.
  - Apply training filter >=500 days per stock.
  - Verify CARTHAGE CEMENT listing/history explanation in report notes.
  - Recover missing files only when a key stock fails history threshold or has unexpected recent gaps.
- User decision: ACCEPTED recommended default strategy (keep companies with >=500 days that pass quality checks, even with early missing years).
- Status: RESOLVED (decision finalized), continue monitoring in future refreshes.

### ISSUE-002: Price Logic Mismatch (close outside [low, high])
- Date opened: 2026-03-19
- Evidence: earlier check flagged out-of-range rows.
- Actions taken:
  - Notebook logic was corrected and rerun.
  - Structured audit cells were added.
  - Controlled DB fix cell with dry-run and change log was added.
- Verification after rerun:
  - close_outside_range = 0
  - low_gt_high = 0
  - low_high_zero_close_pos = 0
  - Final quality report returned GO.
- Status: RESOLVED.

### ISSUE-003: Pre-Training Checklist Integration Before TFT Launch
- Date opened: 2026-03-19
- Source: TFT_PreTraining_Checklist.pdf
- Objective:
  - Convert checklist requirements into executable gates before model training.
- Actions taken:
  - Updated training/train_tft.py with pre-training sanity checks and leakage guards.
  - Added a hard-fail gate for NULLs, duplicate (ticker_id, time_idx), bad RSI, bad MA20, and bad close_price.
  - Added coverage policy enforcement: keep only stocks with >=500 trading days.
  - Added walk-forward overlap checks (Train/Val and Val/Test date overlap must be 0).
  - Added smoke-test mode toggle (3 stocks, 3 epochs) for fast pipeline validation.
- Comments:
  - Class imbalance is currently surfaced as warning (not hard-fail) to keep flexibility.
  - Recommended next upgrade: pass class weights into the loss if imbalance drifts outside 35/65 to 65/35.
- Status: IN PROGRESS (run smoke test, then full training go/no-go).

### ISSUE-004: Scanned PDF Extraction Failure (pdfplumber returns empty text)
- Date opened: 2026-03-21
- Source: bank/financial PDF extraction pipeline (12 failed records, confidence 0.00)
- Problem summary:
  - Some PDFs are image-based scans, so text extractors fail even when tables are visible.
  - This causes near-empty extraction and repeated low-confidence outputs.
- Decision:
  - Switch these failed files to a vision-first path:
    - Convert pages to images with pymupdf (fitz)
    - Send page images (pages 1-6) to GPT-4o vision with high detail
  - Keep scope targeted to failed records only to control cost and time.
- Implementation notes:
  - New notebook created: notebooks/Fix-scanned-pdfs.ipynb
  - Notebook contains 8 code cells:
    1) imports + token/client setup
    2) PDF-to-base64 image conversion
    3) vision prompts (bank + non-bank)
    4) GPT-4o vision API call helper
    5) validation logic
    6) DB helper functions
    7) targeted run for the failed-record list
    8) verification query
- Expected impact:
  - Recover extraction for scanned layouts without changing the main text pipeline.
  - Preserve existing validation/upsert logic and confidence review flow.
- Status: IN PROGRESS (notebook prepared, awaiting manual run).

### ISSUE-005: Fix-scanned-pdfs Notebook Runtime/Completeness Problems
- Date opened: 2026-03-21
- Source: notebooks/Fix-scanned-pdfs.ipynb
- Problem summary:
  - Processing cell failed at runtime with KeyError in prompt formatting.
  - Verification cell was missing from the notebook tail.
- Root cause:
  - PROMPT_VISION_DT used `.format(...)` while also containing JSON braces, causing placeholder parsing errors.
  - Notebook ended at processing step without a dedicated verification step.
- Actions taken:
  - Reworked DT prompt templating to use `__COMPANY_TYPE__` token + `.replace(...)`.
  - Updated vision call helper to use the safe replacement strategy.
  - Added a new final verification cell that queries and prints extracted rows for failed tickers.
- Verification status:
  - Structure fixed (notebook now contains a complete verification cell).
  - Runtime verification pending manual re-run of cells 3 -> 8.
- Status: RESOLVED (code fix), awaiting execution validation.

### ISSUE-007: TFT Training Crash On Lightning AI (numpy.object_ target conversion)
- Date opened: 2026-03-21
- Source: training/train_tft.py run on Lightning AI
- Evidence summary:
  - TimeSeriesDataSet build failed with:
    - TypeError: can't convert np.ndarray of type numpy.object_
  - Stack trace points to pytorch_forecasting encoder transform during target handling.
- Root cause hypothesis:
  - `direction_7d` target is stored as string labels ('0'/'1').
  - `target_normalizer` was set to `None`, so labels were not encoded to numeric class IDs before tensor conversion.
- Decision:
  - Keep categorical string labels but explicitly encode them with `NaNLabelEncoder()` in TimeSeriesDataSet.
- Changes applied:
  - Updated `training/train_tft.py` imports to include `NaNLabelEncoder`.
  - Changed TimeSeriesDataSet `target_normalizer` from `None` to `NaNLabelEncoder()`.
  - Updated Section 6 comments to document why this is required.
- Verification result:
  - Static code update complete.
  - Runtime verification pending manual rerun on Lightning AI.
- Status: RESOLVED (code fix), awaiting execution validation.

### ISSUE-008: Lightning Sanity Check Crash (limit_val_batches too small)
- Date opened: 2026-03-21
- Source: training/train_tft.py run on Lightning AI
- Evidence summary:
  - Trainer crashed during sanity check with:
    - MisconfigurationException: You requested to check 0.5 of the val_dataloader but 0.5 * 1 < 1
  - Validation loader in smoke mode had only 1 batch.
- Root cause hypothesis:
  - `limit_val_batches=0.5` is invalid when val dataloader length is 1 because fractional limit resolves to <1 batch.
- Decision:
  - Keep fast validation in smoke mode when possible, but force at least one full batch for tiny validation loaders.
- Changes applied:
  - Added dynamic `limit_val_batches` logic in `training/train_tft.py`:
    - if smoke mode and `len(val_dataloader) < 2` -> use `1`
    - else keep existing behavior (`0.5` for smoke, `1.0` for full)
  - Updated Trainer argument to use computed `limit_val_batches` variable.
- Verification result:
  - Code fix applied.
  - Runtime verification pending manual rerun on Lightning AI.
- Status: RESOLVED (code fix), awaiting execution validation.

### ISSUE-009: Lightning AI Editor Red Lines (Pylance type-analysis false positives)
- Date opened: 2026-03-21
- Source: training/train_tft.py diagnostics in Lightning AI editor
- Evidence summary:
  - Multiple `reportAttributeAccessIssue` warnings for pandas operations (`astype`, `fillna`, `unique`, `isin`, `columns`, `reset_index`).
  - `reportArgumentType` warnings for `TimeSeriesDataSet` DataFrame args and `Trainer` callbacks/logger/model due mixed lightning/pytorch_lightning import unions.
- Root cause hypothesis:
  - Static analyzer inferred broad union types (`DataFrame | Series | ndarray | scalar`) in pandas flows.
  - Fallback import pattern creates union callback/logger types that are runtime-valid but noisy for static type checking.
- Decision:
  - Keep runtime behavior unchanged.
  - Add explicit DataFrame guards and narrow/cast points so type checker can prove correct types.
- Changes applied:
  - Added `ensure_dataframe()` helper and applied it to key DataFrame creation/slicing/reset points.
  - Split `pd.to_numeric()` conversions into intermediate variables and normalized to `pd.Series` before `.astype()` / `.fillna()`.
  - Added explicit series casts for `nunique/unique/isin` usage in smoke-test and logging.
  - Added explicit casts for `TimeSeriesDataSet` inputs and `Trainer` callbacks/logger/model call sites.
- Verification result:
  - Code edits complete.
  - Runtime behavior intentionally unchanged; training metrics path unaffected.
  - Static diagnostics expected to be significantly reduced after reload.
- Status: RESOLVED (code fix), awaiting editor refresh validation.

### ISSUE-010: TechnicalAgent Inference Crash (empty TimeSeriesDataSet index + predict output API mismatch)
- Date opened: 2026-03-23
- Source: agents/technical_agent.py runtime on local Windows GPU env
- Evidence summary:
  - Inference failed after model load with:
    - AssertionError: filters should not remove entries all entries - check encoder/decoder lengths and lags
  - Warning indicated dataset constraints removed the only series/group.
  - Checkpoint parameters showed `max_encoder_length=60`, `max_prediction_length=7`, `min_prediction_idx=73`.
  - After fixing index constraints, a second runtime error appeared:
    - AttributeError: 'Output' object has no attribute 'output'
- Root cause hypothesis:
  - Inference dataframe used relative `time_idx` ending at 59, below checkpoint `min_prediction_idx` constraints in predict path.
  - Feature window size did not include decoder horizon (`max_prediction_length`).
  - Installed `pytorch_forecasting` version returns predict output with `.prediction` instead of `.output.prediction`.
- Decision:
  - Make inference prep dynamic from checkpoint dataset parameters and support both predict output formats.
- Changes applied:
  - Updated `agents/technical_agent.py`:
    - Increased SQL fetch buffer to keep enough clean rows after indicator filters.
    - Derived required rows at runtime as `max(max_encoder_length, window_size) + max_prediction_length`.
    - Passed `min_prediction_idx` and `max_prediction_length` into dataframe prep.
    - Shifted `time_idx` so decoder start satisfies checkpoint constraints.
    - Added compatibility extraction for both `raw_output.output.prediction` and `raw_output.prediction`.
    - Added clearer `AssertionError` mapping when dataset filtering empties all entries.
- Verification result:
  - Re-ran: `python agents/technical_agent.py "AMEN BANK"`
  - Output now succeeds with valid signal:
    - `NEUTRAL BULLISH (54%)`, `P(DOWN)=0.4573`, `P(UP)=0.5427`
  - Data quality reported:
    - `rows_available=67`, `rows_required=67`, `min_prediction_idx=73`, `null_count=0`
- Status: RESOLVED (code fix validated at runtime).

### ISSUE-011: TFT Evaluation Script Crash (missing labels path + metric variable mismatch)
- Date opened: 2026-03-23
- Source: training/evaluate_tft.py runtime on local Windows GPU env
- Evidence summary:
  - Script failed during label extraction with:
    - KeyError: 'direction_7d'
  - After partial fixes, script progressed but then failed in metric reporting with:
    - NameError: name 'n_up_correct' is not defined
- Root cause hypothesis:
  - Prediction output index does not carry the target label column for this evaluation path.
  - Inference block briefly used an invalid `model.predict(x_batch_dict, ...)` call signature.
  - Metric section contained inconsistent variable names (`prec_up`, `rec_up`, `all_actuals`, etc.) not defined in scope.
- Decision:
  - Keep evaluation flow simple and stable: predict from dataloader once, extract labels from returned `x.decoder_target`, then compute/report metrics with consistent variable names.
- Changes applied:
  - Updated `training/evaluate_tft.py`:
    - Kept `predict=False` for test dataset reconstruction to preserve evaluation labels.
    - Replaced per-batch dict prediction call with `model.predict(test_dataloader, mode="raw", return_x=True)`.
    - Added robust output extraction compatible with different `pytorch-forecasting` output shapes (`.prediction`, dict, tuple/list fallback).
    - Extracted true labels from `raw_predictions.x["decoder_target"]`.
    - Fixed metric/report variable mismatches and computed missing per-class counters/metrics.
- Verification result:
  - Re-ran: `python training/evaluate_tft.py`
  - Script completed successfully and saved results:
    - `results/eval_tft.json`
  - Reported top-line metric:
    - Overall Accuracy: `52.11%`
- Status: RESOLVED (code fix validated at runtime).

### ISSUE-012: TFT 2025 Evaluation Quality Comment
- Date opened: 2026-03-23
- Source: `python training/evaluate_tft.py` final metrics
- Evidence summary:
  - Accuracy: `52.11%`
  - F1 macro: `0.3972`
  - UP recall: `0.0702`
  - DOWN recall: `0.9421`
  - Prediction pattern: model predicts DOWN most of the time.
- Comment (good or not):
  - Mixed result.
  - Good for raw speed/inference stability.
  - Not good for balanced directional forecasting because UP moves are mostly missed (very low UP recall, low macro F1).
  - Conclusion for thesis wording: performance is only slightly above random in overall accuracy, but class behavior is strongly biased and not yet reliable for UP-signal capture.
- Recommended next check:
  - Compare against CNN-LSTM on the same 2025 split before deciding final model.
- Status: DOCUMENTED.

### ISSUE-013: CNN-LSTM Evaluation Crash (checkpoint key/config compatibility)
- Date opened: 2026-03-24
- Source: `python training/evaluate_cnn_lstm.py` runtime on local Windows GPU env
- Evidence summary:
  - Initial failure:
    - `KeyError: 'acc_up'`
  - Follow-up failure after first fix:
    - `KeyError: 'num_filters3'`
  - Follow-up failure after second fix:
    - `RuntimeError` while loading state dict (`proj.*` missing and LSTM input size mismatch)
- Root cause hypothesis:
  - Evaluation script expected a newer checkpoint schema (extra checkpoint metrics and newer 3-conv architecture fields).
  - Existing checkpoint (`models/cnn_lstm_v2.pt`) uses an older 2-conv architecture config (`kernel_size` only, no `num_filters3`, no projection layer).
- Decision:
  - Make evaluation code backward-compatible with both old and new checkpoint formats.
- Changes applied:
  - Updated `training/evaluate_cnn_lstm.py`:
    - Replaced hard key access for `acc_up` / `acc_down` with optional prints (`N/A` when absent).
    - Added safe `.get(...)` handling for model config fields (supports `kernel_size` alias and missing new fields).
    - Made model architecture conditional:
      - Old format: 2-conv path without projection, LSTM input from conv2 output.
      - New format: existing 3-conv + projection path.
    - Kept result export stable while safely reading training validation metrics.
- Verification result:
  - Re-ran: `python training/evaluate_cnn_lstm.py`
  - Script completed and saved:
    - `results/eval_cnn_lstm.json`
  - Final metrics observed:
    - Accuracy: `45.40%`
    - F1 macro: `0.4492`
    - UP recall: `0.5658`
    - DOWN recall: `0.3493`
  - Runtime note:
    - `WARNING: No per-stock stats found — using test data stats (suboptimal)`
- Status: RESOLVED (code fix validated at runtime).

### ISSUE-014: TFT v3 model integration + pipeline sync updates
- Date opened: 2026-03-24
- Source: user update after training/downloading new checkpoint `tft_bvmt_full_v3_20260324_0246_epoch=16_val_loss=0.5161.ckpt`
- Objective:
  - Synchronize runtime and evaluation pipeline with TFT v3 feature schema and new checkpoint.
- Changes applied:
  - Updated `agents/technical_agent.py`:
    - Model selection now prefers latest `tft_bvmt_full_v3_*.ckpt` (fallback to `tft_bvmt_best.ckpt`).
    - Expanded inference feature schema from 12 to 15 features.
    - Updated SQL query to compute the 3 relative features inline:
      - `price_to_ma20_ratio`
      - `rsi_momentum` (5-day lag diff per isin)
      - `volume_spike`
  - Updated `training/evaluate_tft.py`:
    - Model selection now prefers latest `tft_bvmt_full_v3_*.ckpt`.
    - Added v3 feature support by deriving missing relative features from base columns when CSV lacks them.
    - Fixed sample-count handling (`n_samples`) after filtering.
    - Added zero-division-safe precision/recall for robustness.
    - Corrected next-step command to `training/compare_models_cnnlstm_tft.py`.
  - Updated `training/evaluate_cnn_lstm.py` next-step command to `training/compare_models_cnnlstm_tft.py`.
  - Updated `training/compare_models_cnnlstm_tft.py`:
    - Corrected run instructions to actual script filename.
    - Added explicit warning when TFT and CNN-LSTM sample counts differ.
- Verification result:
  - `python -m py_compile agents/technical_agent.py training/evaluate_tft.py training/evaluate_cnn_lstm.py training/compare_models_cnnlstm_tft.py` succeeded.
  - Re-ran `python training/evaluate_tft.py` with v3 model:
    - Accuracy: `51.08%`
    - F1 macro: `0.4788`
    - UP recall: `0.2722`
    - DOWN recall: `0.7337`
    - Output: `results/eval_tft.json`
  - Re-ran `python training/compare_models_cnnlstm_tft.py`:
    - Output: `results/compare_models.json`
    - Comparison currently shows CNN-LSTM slightly ahead on accuracy (`51.28%` vs `51.08%`), but with a sample-count mismatch warning.
- Important note:
  - Current TFT and CNN-LSTM evaluations are not on equal sample counts (`196945` vs `7734`), so direct ranking should be interpreted cautiously until both scripts use identical test construction.
- Status: RESOLVED (pipeline synced and validated), with comparison methodology caution documented.

### ISSUE-015: SentimentAgent Query Crash (timestamp argument type mismatch)
- Date opened: 2026-04-06
- Source: `notebooks/test_pipeline.ipynb` Cell 3 runtime logs
- Evidence summary:
  - Sentiment step failed with:
    - `invalid input for query argument $3: '2025-06-30' ('str' object has no attribute 'toordinal')`
  - This prevented sentiment lookup and forced neutral fallback output.
- Root cause hypothesis:
  - `agents/sentiment_agent.py` passed a plain string as the `$3` bound value for timestamp comparison in asyncpg.
  - asyncpg date/timestamp binding expects Python `date`/`datetime` objects, not raw strings.
- Decision:
  - Normalize the reference time before query execution and always pass a DB-safe `datetime` object.
- Changes applied:
  - Updated `agents/sentiment_agent.py`:
    - Added `_resolve_reference_timestamp(state)` helper.
    - Added safe parsing order: `state.seance` -> `state.created_at` -> `datetime.utcnow()` fallback.
    - Added string/date/datetime handling and timezone normalization for DB compatibility.
    - Replaced query argument `$3` from string literal to resolved datetime object.
- Verification result:
  - Re-ran `notebooks/test_pipeline.ipynb` (Cell 1 then Cell 3) after kernel restart.
  - Sentiment step no longer throws the asyncpg type error.
  - Execution log now shows:
    - `SentimentAgent started`
    - `Articles in 30-day window: 0`
    - `No articles — signal = 0.0`
    - `Sentiment signal: +0.0000  conf=0.000`
  - Pipeline completed successfully (`direction=UP`, confidence emitted).
- Status: RESOLVED (runtime type crash fixed and validated).

  ### ISSUE-006: Pass-2 Notebook Creation For NPL And Review Fixes
  - Date opened: 2026-03-21
  - Source: user-requested combined pass-2 workflow script (NPL + re-review updates)
  - Problem summary:
    - A separate notebook was needed to run a combined second pass without modifying prior notebooks.
    - The notebook had to be created from provided code with no automatic execution.
  - Decision:
    - Create a new notebook under notebooks named `Fix npl and review.ipynb`.
    - Keep the notebook code-only and unexecuted.
  - Changes applied:
    - Added `notebooks/Fix npl and review.ipynb` with structured cells for:
      - setup and API/token helpers
      - NPL page discovery and extraction logic
      - targeted re-extractions (BNA ASSURANCES and SERVICOM)
      - AMS FY2018 verification SQL update
      - final verification queries
  - Verification result:
    - File creation completed successfully.
    - Execution intentionally not performed per user constraint.
  - Status: RESOLVED (creation complete, awaiting manual run).

## Pre-Training Go/No-Go Comments

- GO only if all hard checks pass:
  - 0 NULLs in feature table
  - 0 duplicate (ticker_id, time_idx)
  - RSI in [0,100], ma_20 > 0, close_price > 0
  - 0 date overlap between train/val/test splits
  - Feature file exists and row count remains strong after >=500-day stock filter
- WARNINGS (do not auto-block but must be documented):
  - Class imbalance outside 35/65 to 65/35
  - Key-stock coverage anomalies that are explainable by listing date/suspension

## Applied Changes Register

- 2026-03-19: Added focused coverage outputs and issue CSV export in check_data_quality notebook.
- 2026-03-19: Added data cleaning workflow section with audit exports and controlled DB fix dry-run.
- 2026-03-19: Standardized audit output paths to project-level reports/data_cleaning.
- 2026-03-19: Integrated pre-training checklist gates directly into training/train_tft.py (sanity checks, coverage policy, leakage overlap checks, smoke test mode).
- 2026-03-19: Added a pre-training checklist validation section at the end of notebooks/feature_engineering.ipynb.
- 2026-03-21: Added ISSUE-004 and documented scanned-PDF vision fix strategy.
- 2026-03-21: Created notebooks/Fix-scanned-pdfs.ipynb with a targeted GPT-4o vision extraction workflow (code only, not executed).
- 2026-03-21: Fixed prompt-format KeyError in notebooks/Fix-scanned-pdfs.ipynb and added missing verification cell.
- 2026-03-21: Added notebooks/Fix npl and review.ipynb (pass-2 combined NPL/review workflow, code only, not executed).
- 2026-03-21: Fixed Lightning AI TFT target conversion crash by switching Section 6 target_normalizer to NaNLabelEncoder() in training/train_tft.py.
- 2026-03-21: Fixed Lightning AI sanity-check crash by making limit_val_batches dynamic for tiny smoke-test validation sets in training/train_tft.py.
- 2026-03-21: Reduced Lightning AI Pylance red lines by adding explicit pandas type narrowing and safe casts in training/train_tft.py.
- 2026-03-23: Fixed TechnicalAgent inference dataset-index failure by aligning runtime window/time_idx to checkpoint dataset parameters and adding cross-version predict output extraction compatibility in agents/technical_agent.py.
- 2026-04-06: Fixed SentimentAgent asyncpg timestamp binding crash by converting reference date inputs to Python datetime in agents/sentiment_agent.py and validated via test_pipeline notebook rerun.

## Next Update Template

Copy this block for every new issue:

### ISSUE-XXX: <title>
- Date opened:
- Source:
- Evidence summary:
- Root cause hypothesis:
- Action options considered:
- Decision:
- Changes applied:
- Verification result:
- Status: OPEN / IN PROGRESS / RESOLVED
## TFT Final Result — v3 — March 24, 2026

| Model         | Val Loss | vs Baseline | Best Epoch |
|---------------|----------|-------------|------------|
| Random        | 0.6931   | —           | —          |
| TFT v2        | 0.6036   | +12.9%      | 6          |
| TFT v3 FINAL  | 0.5161   | +25.5%      | 16         |
| CNN-LSTM      | 0.6971   | ~0%         | 1          |

Changes that drove the improvement:
- LR 0.001 → 0.0003 + 3-epoch warmup (most impactful)
- Dropout locked at 0.3
- 3 new relative features: price_to_ma20_ratio, rsi_momentum, volume_spike
- gradient_clip 0.1 → 0.5, patience 5 → 8, MAX_EPOCHS 30 → 50