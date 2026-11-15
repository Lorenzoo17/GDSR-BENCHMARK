@echo off
REM Protocollo del paper su TOFDSR: prima FP32 (riferimento, TF32 spento), poi FP16.
REM Va lanciato dal prompt con l'ambiente Python/conda attivo, oppure imposta PYTHON.
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" run_benchmark.py --dataset tofdsr --precision fp32 %*
"%PYTHON%" run_benchmark.py --dataset tofdsr --precision fp16 %*
pause
