@echo off
REM ============================================================================
REM patch1_run_gate_ablation.bat
REM ----------------------------------------------------------------------------
REM Gated-vs-ungated ablation on the external (Qiu et al.) cohort.
REM Addresses the single heaviest criticism: the manuscript never tests whether
REM the quality gate actually improves downstream HR estimation.
REM
REM Runs 5 conditions, all with IDENTICAL downstream parameters
REM (rho=0.12, k=1) -- only the gate changes:
REM     ungated   : --disable_gate  (every 1-s interval marked usable)
REM     beta0.30  : gate ON, block threshold 0.30
REM     beta0.50  : gate ON, block threshold 0.50   <-- manuscript's operating point
REM     beta0.70  : gate ON, block threshold 0.70
REM     beta0.90  : gate ON, block threshold 0.90
REM
REM Then run:  python patch1_compare_gate_ablation.py
REM
REM RUNTIME: ~5x a single inference pass over 46 recordings. Budget accordingly.
REM          Use --predict_batch_size 512 on the RTX 4060 if VRAM allows.
REM ============================================================================

setlocal

REM ---- EDIT THESE IF YOUR LAYOUT DIFFERS ----
set MODEL_DIR=cv_output_nested_v2\final_model
set INPUT_DIR=NewData_processed
set GLOB=Sub*_aligned_data_ecg.txt
set ROOT=gate_ablation
set PROM=0.12
set MINJJ=1
set BATCH=256
REM -------------------------------------------

if not exist "%MODEL_DIR%\model.keras" (
    echo ERROR: %MODEL_DIR%\model.keras not found.
    echo Edit MODEL_DIR at the top of this script.
    exit /b 1
)
if not exist "%INPUT_DIR%" (
    echo ERROR: %INPUT_DIR% not found.
    echo Edit INPUT_DIR at the top of this script.
    exit /b 1
)

if not exist "%ROOT%" mkdir "%ROOT%"

echo.
echo ============================================================
echo [1/5] UNGATED baseline (gate bypassed entirely)
echo ============================================================
python run_hr_pipeline.py ^
    --model_dir "%MODEL_DIR%" ^
    --input_dir "%INPUT_DIR%" ^
    --output_dir "%ROOT%\ungated" ^
    --glob "%GLOB%" ^
    --disable_gate ^
    --prominence_coef %PROM% ^
    --min_valid_jj_intervals %MINJJ% ^
    --predict_batch_size %BATCH%
if errorlevel 1 goto :failed

for %%B in (0.30 0.50 0.70 0.90) do (
    echo.
    echo ============================================================
    echo GATED, beta=%%B
    echo ============================================================
    python run_hr_pipeline.py ^
        --model_dir "%MODEL_DIR%" ^
        --input_dir "%INPUT_DIR%" ^
        --output_dir "%ROOT%\beta%%B" ^
        --glob "%GLOB%" ^
        --bcg_fraction_threshold %%B ^
        --prominence_coef %PROM% ^
        --min_valid_jj_intervals %MINJJ% ^
        --predict_batch_size %BATCH%
    if errorlevel 1 goto :failed
)

echo.
echo ============================================================
echo All 5 conditions complete. Aggregating...
echo ============================================================
python patch1_compare_gate_ablation.py --ablation_root "%ROOT%" --out_dir figures
if errorlevel 1 goto :failed

echo.
echo DONE. See figures\patch1_gate_ablation_summary.txt
goto :eof

:failed
echo.
echo *** A step FAILED (exit code %errorlevel%). Stopping. ***
exit /b %errorlevel%
