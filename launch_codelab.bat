@echo off
rem CODE Lab Imaging Pipeline launcher -- Windows (double-clickable).
rem Same resolution order as launch_codelab.command, so both platforms
rem behave identically:
rem   1. CODELAB_PYTHON env var (explicit interpreter -- the reliable knob)
rem   2. conda (on PATH, or common miniforge/miniconda install locations),
rem      running the cellclassifier env
rem   3. plain python and hope the active environment has the deps
cd /d "%~dp0"

if defined CODELAB_PYTHON (
    "%CODELAB_PYTHON%" main.py
    goto :eof
)

where conda >nul 2>nul
if %errorlevel%==0 (
    conda run -n cellclassifier --no-capture-output python main.py
    goto :eof
)

for %%C in ("%USERPROFILE%\miniforge3\Scripts\conda.exe"
            "%USERPROFILE%\miniconda3\Scripts\conda.exe"
            "%USERPROFILE%\anaconda3\Scripts\conda.exe") do (
    if exist %%C (
        %%C run -n cellclassifier --no-capture-output python main.py
        goto :eof
    )
)

python main.py
