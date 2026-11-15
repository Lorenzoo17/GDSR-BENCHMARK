@echo off
REM Scarica il dataset TOFDSR, lo estrae in datasets\TOFDSR e aggiorna config.json.
REM Opzioni utili: --solo-test   --da-archivio C:\Download\TOFDSR.zip   --dest D:\dati\TOFDSR
REM Va lanciato dal prompt con l'ambiente Python/conda attivo, oppure imposta PYTHON.
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" download_tofdsr.py %*
if errorlevel 1 (echo. & echo Terminato con errori: leggi i messaggi sopra.)
pause
