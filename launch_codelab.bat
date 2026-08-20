@echo off
rem CODE Lab Imaging Pipeline launcher -- Windows (double-clickable).
rem Same resolution order as launch_codelab.command, so both platforms
rem behave identically:
rem   1. CODELAB_PYTHON env var (explicit interpreter -- the reliable knob)
rem   2. conda (on PATH, or common miniforge/miniconda/anaconda install
rem      locations), running the cellclassifier env -- but ONLY if that env
rem      actually exists. A machine with conda but no cellclassifier used to
rem      die here on "EnvironmentLocationNotFound" instead of falling
rem      through, even when conda's base env had every dependency.
rem   3. conda's base env, then plain python, and hope the active
rem      environment has the deps
setlocal
cd /d "%~dp0"

set "CODELAB_ENV=cellclassifier"

if defined CODELAB_PYTHON (
    "%CODELAB_PYTHON%" main.py
    goto :end
)

rem -- locate a conda: PATH first, then the usual install roots --
set "CONDA="
for /f "delims=" %%C in ('where conda 2^>nul') do if not defined CONDA set "CONDA=%%C"
if not defined CONDA call :try_conda "%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%ProgramData%\miniforge3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%ProgramData%\Anaconda3\Scripts\conda.exe"
if not defined CONDA goto :plain

rem conda.exe lives in <root>\Scripts\, so the install root is two levels up.
for %%I in ("%CONDA%\..\..") do set "CONDA_ROOT=%%~fI"

if exist "%CONDA_ROOT%\envs\%CODELAB_ENV%\python.exe" (
    "%CONDA%" run -n %CODELAB_ENV% --no-capture-output python main.py
    goto :end
)

if exist "%CONDA_ROOT%\python.exe" (
    echo [launcher] conda env "%CODELAB_ENV%" not found -- using conda base at "%CONDA_ROOT%".
    "%CONDA%" run -n base --no-capture-output python main.py
    goto :end
)

:plain
where python >nul 2>nul
if not errorlevel 1 (
    python main.py
    goto :end
)

echo [launcher] No usable Python found.
echo [launcher] Point CODELAB_PYTHON at the interpreter you want, e.g.
echo [launcher]   setx CODELAB_PYTHON "%USERPROFILE%\anaconda3\python.exe"
pause
goto :end

:try_conda
if exist %1 set "CONDA=%~1"
goto :eof

:end
endlocal
