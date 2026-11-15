@echo off
REM Scarica le reti del benchmark e i pesi. Aggiungi --pip per installare i pacchetti.
REM Va lanciato dal prompt con l'ambiente Python/conda attivo, oppure imposta PYTHON:
REM   set PYTHON=C:\percorso\python.exe
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" download_models.py %*
if errorlevel 1 (echo. & echo Terminato con errori: leggi i messaggi sopra.)
pause
