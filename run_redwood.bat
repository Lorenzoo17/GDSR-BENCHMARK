@echo off
REM Test su Redwood (formulazione B, x4).
REM Va lanciato dal prompt con l'ambiente Python/conda attivo, oppure imposta PYTHON:
REM   set PYTHON=C:\percorso\python.exe
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" run_benchmark.py --dataset redwood %*
if errorlevel 1 (echo. & echo Terminato con errori: leggi i messaggi sopra.)
pause
