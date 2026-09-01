@echo off
rem CODE Lab Imaging Pipeline launcher -- Windows (double-clickable).
rem Same resolution order as launch_codelab.command, so both platforms
rem behave identically:
rem   1. CODELAB_PYTHON env var (explicit interpreter -- the reliable knob)
rem   2. the code_lab_imaging_pipeline conda env, if conda says it exists
rem      (or the pre-rename "cellclassifier" env, if only that one exists)
rem   3. offer to CREATE that env from requirements.txt and then use it
rem   4. conda's base env, then plain python, and hope the active
rem      environment has the deps
rem
rem CODELAB_LAUNCH_QUIET=1 (set by launch_codelab.vbs, the no-console
rem launcher) means this window is HIDDEN: never prompt and never pause,
rem because there is nothing on screen to answer and the wait would be
rem invisible and forever. It reports through its exit code instead.
rem
rem Step 3 exists because base is not a safe fallback for this project any
rem more: requirements.txt pins cellpose below 4 (4.x deleted the
rem models.Cellpose class segment.py builds), and a base env that has
rem cellpose 4 imports fine and only fails once someone runs segmentation.
rem A purpose-built env is the fix; creating it should not be a manual
rem chore, so the launcher offers to do it.
setlocal
cd /d "%~dp0"

set "CODELAB_ENV=code_lab_imaging_pipeline"
rem The env was called "cellclassifier" before the project got its own
rem name. An existing one is still perfectly good -- same
rem requirements.txt -- so it is used rather than making anyone
rem re-download several GB just because the canonical name changed.
set "CODELAB_LEGACY_ENV=cellclassifier"
set "CODELAB_PY_VERSION=3.11"

if defined CODELAB_PYTHON (
    call :run_direct "%CODELAB_PYTHON%"
    goto :end
)

rem -- fast path: a remembered interpreter, run without conda -----------
rem
rem Resolving the interpreter through conda costs 5.8 s on this machine,
rem measured: `where conda` 133 ms, a dead Anaconda husk probed at 62 ms
rem before the working conda at 175 ms, `conda env list` 1977 ms to ask
rem whether the env exists, and finally `conda run` -- 3536 ms against
rem 71 ms for invoking the same interpreter directly, i.e. ~3.5 s of pure
rem wrapper before python's first line. That was more than the entire
rem application import (2.5 s) and the user stares at nothing for all of
rem it.
rem
rem None of it changes between launches, so it is resolved ONCE and
rem remembered next to this script. The remembered path is trusted only
rem while the file it names still exists, and it is only ever written
rem after that interpreter has been PROVEN to import the packages the app
rem cannot start without -- so a cache hit is a validated interpreter,
rem not a guess. Delete .codelab_python to force re-resolution (after
rem moving or rebuilding the env, say).
set "CODELAB_PY_CACHE=%~dp0.codelab_python"
set "CACHED_PY="
if not exist "%CODELAB_PY_CACHE%" goto :find_conda
set /p CACHED_PY=<"%CODELAB_PY_CACHE%"
if not defined CACHED_PY goto :stale_cache
if not exist "%CACHED_PY%" goto :stale_cache
call :run_direct "%CACHED_PY%"
goto :end

:stale_cache
rem The remembered interpreter is gone -- env moved, renamed, or rebuilt
rem somewhere else. FORGET it rather than merely stepping around it: a
rem cache file that survives while naming a dead path sends this launch
rem and every later one back down the 5.8 s conda road, which is the exact
rem cost this is here to avoid.
del "%CODELAB_PY_CACHE%" >nul 2>&1
set "CACHED_PY="

:find_conda
rem -- locate a conda: PATH first, then the usual install roots --
rem
rem The non-system-drive roots are not decoration. This machine ran C: out
rem of space, so the conda install itself was moved to D: -- and a conda
rem that is not on PATH (installing without "Add to PATH" is the default
rem and the recommended choice) is then invisible to every %USERPROFILE%
rem and %ProgramData% probe below, which sends the launcher all the way
rem down to :plain and the wrong interpreter. CODELAB_PYTHON above is
rem still the reliable knob; this just makes the common layout work
rem without one.
set "CONDA="
rem via :try_conda like every other candidate, so a conda that is ON PATH
rem but broken cannot win either -- a stale PATH entry outlives an
rem uninstall exactly the way a stale file does.
for /f "delims=" %%C in ('where conda 2^>nul') do if not defined CONDA call :try_conda "%%C"
if not defined CONDA call :try_conda "%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%ProgramData%\miniforge3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA call :try_conda "%ProgramData%\Anaconda3\Scripts\conda.exe"
rem Installs moved off the system drive. D: first because that is where
rem this machine's env and package dirs already live (see .condarc).
for %%D in (D E) do (
    if not defined CONDA call :try_conda "%%D:\miniforge3\Scripts\conda.exe"
    if not defined CONDA call :try_conda "%%D:\miniconda3\Scripts\conda.exe"
    if not defined CONDA call :try_conda "%%D:\anaconda3\Scripts\conda.exe"
    if not defined CONDA call :try_conda "%%D:\Anaconda3\Scripts\conda.exe"
    if not defined CONDA call :try_conda "%%D:\conda\Scripts\conda.exe"
)
if not defined CONDA goto :plain

rem -- Ask CONDA whether the env exists, never the filesystem: envs_dirs can
rem    put it anywhere, and this project's own env lives on a different
rem    drive than the conda install (C: was full). --
set "ENV_FOUND="
"%CONDA%" env list 2>nul | findstr /b /c:"%CODELAB_ENV% " >nul
if not errorlevel 1 set "ENV_FOUND=1"
if defined ENV_FOUND goto :run_env

"%CONDA%" env list 2>nul | findstr /b /c:"%CODELAB_LEGACY_ENV% " >nul
if errorlevel 1 goto :no_env
echo [launcher] "%CODELAB_ENV%" not found; using the pre-rename env "%CODELAB_LEGACY_ENV%".
set "CODELAB_ENV=%CODELAB_LEGACY_ENV%"
goto :run_env

:no_env
echo [launcher] conda env "%CODELAB_ENV%" does not exist yet.
if /i "%CODELAB_BOOTSTRAP%"=="never" goto :base_fallback
if "%CODELAB_LAUNCH_QUIET%"=="1" (
    echo [launcher] hidden launch -- not prompting. Run launch_codelab.bat
    echo [launcher] directly once to create the environment.
    goto :base_fallback
)
if /i "%CODELAB_BOOTSTRAP%"=="yes" goto :do_bootstrap
echo [launcher] It can be created now from requirements.txt. That downloads
echo [launcher] a few GB (CUDA torch) and takes several minutes, once.
echo [launcher] Set CODELAB_BOOTSTRAP=yes to skip this prompt, or =never to
echo [launcher] stop being asked at all.
set "ANSWER="
set /p ANSWER=Create it now? [y/N] 
if /i not "%ANSWER%"=="y" goto :base_fallback

:do_bootstrap
echo [launcher] creating %CODELAB_ENV% (python %CODELAB_PY_VERSION%)...
"%CONDA%" create -y -n %CODELAB_ENV% python=%CODELAB_PY_VERSION%
if errorlevel 1 goto :bootstrap_failed
echo [launcher] installing requirements.txt...
"%CONDA%" run -n %CODELAB_ENV% --no-capture-output python -m pip install -r requirements.txt
if errorlevel 1 goto :bootstrap_failed
echo [launcher] environment ready.
goto :run_env

:bootstrap_failed
echo [launcher] setting up "%CODELAB_ENV%" FAILED -- see the messages above.
echo [launcher] Falling back for this run; the app may not segment correctly.
goto :base_fallback

:run_env
rem The ONE slow resolution, paid once per machine: conda names the
rem interpreter and the interpreter itself writes its path to the cache,
rem so nothing has to parse `conda env list` output (whose columns shift
rem with the active-env marker). The imports in the probe are the ones
rem whose DLLs live in the env's Library\bin -- if a direct call could not
rem load them, this fails HERE, with conda still available, rather than
rem leaving a broken path cached for every future launch.
if exist "%CODELAB_PY_CACHE%" goto :run_env_cached
"%CONDA%" run -n %CODELAB_ENV% --no-capture-output python -c "import sys, h5py, cv2, PyQt5; open(r'%CODELAB_PY_CACHE%', 'w').write(sys.executable)" 2>nul
:run_env_cached
set "CACHED_PY="
if not exist "%CODELAB_PY_CACHE%" goto :run_env_via_conda
set /p CACHED_PY=<"%CODELAB_PY_CACHE%"
if not defined CACHED_PY goto :run_env_via_conda
if not exist "%CACHED_PY%" goto :run_env_via_conda
call :run_direct "%CACHED_PY%"
goto :end

:run_env_via_conda
rem the probe could not produce a usable interpreter -- keep the original
rem behavior rather than failing to launch at all
"%CONDA%" run -n %CODELAB_ENV% --no-capture-output python main.py
goto :end

:base_fallback
rem conda.exe lives in <root>\Scripts\, so the install root is two levels up.
for %%I in ("%CONDA%\..\..") do set "CONDA_ROOT=%%~fI"
if exist "%CONDA_ROOT%\python.exe" (
    echo [launcher] using conda base at "%CONDA_ROOT%".
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
rem exit 1 so the hidden launcher can surface this in a dialog; a pause
rem here would hang invisibly forever.
if "%CODELAB_LAUNCH_QUIET%"=="1" (
    endlocal
    exit /b 1
)
pause
goto :end

:run_direct
rem Run an interpreter WITHOUT `conda run`.
rem
rem What conda activation actually contributes on Windows is the env's DLL
rem directories on PATH: python.exe finds its own prefix, but the HDF5,
rem OpenCV and Qt DLLs live in <prefix>\Library\bin, which it would not
rem search. Prepending exactly the directories activation prepends gives
rem the direct call the same environment for the cost of a `set` instead
rem of a 3.5 s conda process. PATH is restored by the endlocal at :end.
set "CODELAB_PREFIX=%~dp1"
set "PATH=%CODELAB_PREFIX%;%CODELAB_PREFIX%Library\mingw-w64\bin;%CODELAB_PREFIX%Library\usr\bin;%CODELAB_PREFIX%Library\bin;%CODELAB_PREFIX%Scripts;%CODELAB_PREFIX%bin;%PATH%"
"%~1" main.py
goto :eof

:try_conda
rem Accept a candidate only if it actually RUNS. "if exist" alone was not
rem enough: uninstalling Anaconda left a husk behind -- Scripts\conda.exe
rem still on disk, but conda-script.py deleted, so it exits 105 on every
rem invocation. The probe stopped at that dead file and never looked at
rem the working conda further down the list. Worse than not finding one:
rem `%CONDA% env list` then returns nothing, the launcher concludes the
rem project env does not exist, and offers to re-create an env that is
rem sitting there in perfect health.
rem
rem Files left behind by an uninstaller are normal (anything loaded at the
rem time cannot be deleted), so this state is not exotic and the probe has
rem to survive it. --version is the cheapest call that proves conda works,
rem and it runs at most once per candidate before the first success.
if not exist %1 goto :eof
%1 --version >nul 2>&1
if errorlevel 1 goto :eof
set "CONDA=%~1"
goto :eof

:end
endlocal
