@echo off
REM ========================================
REM 🚀 Lancement automatique du Energy ML System
REM ========================================

REM Chemin vers l'environnement virtuel
set VENV_PATH=Energy_ML_System_env

REM Activer l'environnement virtuel
call %VENV_PATH%\Scripts\activate.bat

REM Vérifier que l'environnement est activé
echo Environnement virtuel activé : %VENV_PATH%

REM Lancer le setup initial (si besoin)
python main.py setup

REM Lancer le dashboard Streamlit
python main.py dashboard --port 8501 --host localhost

REM Pause pour voir les messages
pause
