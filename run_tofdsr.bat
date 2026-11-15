@echo off
REM Test su TOFDSR (protocollo completo, FP32). Argomenti extra, es.: run_tofdsr.bat --limit 50
REM Va lanciato dal prompt con l'ambiente Python/conda attivo, oppure imposta PYTHON:
REM   set PYTHON=C:\percorso\python.exe
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" run_benchmark.py --dataset tofdsr %*
if errorlevel 1 (echo. & echo Terminato con errori: leggi i messaggi sopra.)
pause
