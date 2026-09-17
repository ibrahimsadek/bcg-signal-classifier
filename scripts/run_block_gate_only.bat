@echo off
REM ============================================================================
REM patch7_run_block_gate_only.bat
REM ----------------------------------------------------------------------------
REM Runs the BLOCK-GATING-ONLY condition and compares it against the existing
REM ungated and beta=0.50 runs from patch1.
REM
REM patch6 showed 99.1% of the gate's harm is INTERVAL-level restriction
REM (stripping 27.8% of seconds -> losing 41.6% of J-J intervals), not block
REM rejection. This condition keeps block-level gating and removes the
REM interval-level masking:
REM
REM     ungated          block gate OFF   interval mask OFF   MAE 13.907
REM     beta0.50         block gate ON    interval mask ON    MAE 14.632
REM     blockonly0.50    block gate ON    interval mask OFF   <-- THIS RUN
REM
REM If blockonly comes in at or below ungated, the gate has value once applied
REM at the right granularity, and you have a positive result plus a mechanism.
REM If it lands at ungated exactly, block gating is simply neutral at beta=0.50
REM (it only removes 0.7% of blocks, so that is a plausible outcome).
REM
REM RUNTIME: ~1 condition, roughly 1.5-2 h on your hardware.
REM ============================================================================

setlocal

set MODEL_DIR=cv_output_nested_v2\final_model
set INPUT_DIR=NewData_processed
set GLOB=Sub*_aligned_data_ecg.txt
set ROOT=gate_ablation
set PROM=0.12
set MINJJ=1
set BATCH=256

if not exist "run_hr_pipeline_blockgate.py" (
    echo ERROR: run_hr_pipeline_blockgate.py not found.
    echo Run this first:  python patch7_make_block_gate_only.py
    exit /b 1
)
if not exist "%MODEL_DIR%\model.keras" (
    echo ERROR: %MODEL_DIR%\model.keras not found. Edit MODEL_DIR above.
    exit /b 1
)

echo.
echo ============================================================
echo BLOCK-GATING-ONLY, beta=0.50
echo   block-level gate: ON   interval-level mask: OFF
echo ============================================================
python run_hr_pipeline_blockgate.py ^
    --model_dir "%MODEL_DIR%" ^
    --input_dir "%INPUT_DIR%" ^
    --output_dir "%ROOT%\blockonly0.50" ^
    --glob "%GLOB%" ^
    --block_gate_only ^
    --bcg_fraction_threshold 0.50 ^
    --prominence_coef %PROM% ^
    --min_valid_jj_intervals %MINJJ% ^
    --predict_batch_size %BATCH%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo Comparing block-only against ungated
echo ============================================================
python patch6_common_blocks_decomposition.py ^
    --ablation_root "%ROOT%" --ungated ungated --gated blockonly0.50 ^
    --out_dir figures_blockonly
if errorlevel 1 goto :failed

echo.
echo DONE.
echo   figures_blockonly\patch6_common_blocks_summary.txt   (vs ungated)
echo   gate_ablation\blockonly0.50\analysis\                (full results)
goto :eof

:failed
echo.
echo *** FAILED (exit code %errorlevel%) ***
exit /b %errorlevel%
