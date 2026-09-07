@echo off
rem Spot Check launcher -- what a lab member double-clicks.
rem
rem It does ONE thing: find the interpreter and start the app. It asks
rem nothing.
rem
rem An earlier version prompted for the bundle folder and the reviewer
rem name with set /p, and it was the wrong place for those questions
rem twice over. cmd parses parentheses inside a prompt as grouping, so
rem the file died with "The syntax of the command is incorrect" before it
rem ever reached python; and a console prompt behind a GUI is a worse
rem place to pick a folder than a folder picker. The app now asks for
rem both itself, with a QFileDialog and a QInputDialog, and remembers the
rem answers -- see spotcheck/app.py's ask_session.
rem
rem Arguments, if any, pass straight through, so a desktop shortcut can
rem pin a bundle and a reviewer and skip the dialogs entirely:
rem
rem   launch_spotcheck.bat "D:\bundles\MP58_RNA_Hyb109_ch635" --reviewer shj
rem   launch_spotcheck.bat "D:\bundles\..." --reviewer shj --max-per-crop 24

setlocal
cd /d "%~dp0"

set "PYEXE=D:\conda-envs\code_lab_imaging_pipeline\python.exe"
if not exist "%PYEXE%" set "PYEXE=pythonw.exe"

rem pythonw over python for the no-argument case: a double-click should
rem open a window, not a window plus a console box that stays until the
rem app is closed. The named env's python.exe is used when it exists
rem because that is the interpreter this repo is known to run on.

"%PYEXE%" "%~dp0spotcheck\app.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo   Spot Check exited with code %RC%.
    echo.
    echo   If it could not start at all, the interpreter was:
    echo     %PYEXE%
    echo   The app needs PyQt5, matplotlib, numpy and h5py.
    echo.
    pause
)
endlocal
