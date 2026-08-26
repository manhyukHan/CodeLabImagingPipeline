' CODE Lab Imaging Pipeline -- Windows launcher with NO console window.
'
' Double-click this instead of launch_codelab.bat. A .bat is always
' hosted by a cmd.exe console; wscript is not, so this runs the same
' launcher with its window hidden and you see only the app.
'
' It deliberately does NOT re-implement the .bat's interpreter
' resolution (CODELAB_PYTHON -> conda env -> base -> plain python).
' That logic has real history in it and must not exist twice; this only
' changes HOW the .bat is hosted.
'
' Two things follow from hiding the window, and both are handled:
'
'   * Nothing may block on input. A hidden "Create the env? [y/N]" or
'     "pause" would wait forever with nothing on screen to answer. The
'     .bat sees CODELAB_LAUNCH_QUIET=1 and never prompts.
'   * Nothing may write to a console that isn't there. Output is
'     redirected to a log file, so stdout/stderr stay real handles
'     (running the GUI under pythonw.exe instead would leave sys.stdout
'     as None, and any stray print() would then raise).
'
' On failure -- and only then -- a message box appears with the end of
' that log, so a launch that dies is never silent.

Option Explicit

Dim shell, fso, here, logPath, cmd, rc
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
logPath = shell.ExpandEnvironmentStrings("%TEMP%") & "\codelab_launch.log"

' Quiet mode: no prompts, no pause -- see the note above.
shell.Environment("PROCESS")("CODELAB_LAUNCH_QUIET") = "1"

' Wrapping quotes around the whole redirected command is cmd.exe's own
' rule for a command that itself contains quoted paths.
cmd = "cmd /c """"" & here & "\launch_codelab.bat"" > """ & logPath & """ 2>&1"""

' 0 = hidden window, True = wait, so a failure can still be reported.
' The app's own Qt windows are unaffected: hiding applies to the
' console, not to GUI windows the process creates.
rc = shell.Run(cmd, 0, True)

If rc <> 0 Then
    Dim tail
    tail = ""
    If fso.FileExists(logPath) Then
        Dim stream, lines, i, startAt
        Set stream = fso.OpenTextFile(logPath, 1)
        If Not stream.AtEndOfStream Then lines = Split(stream.ReadAll(), vbLf)
        stream.Close
        If IsArray(lines) Then
            startAt = UBound(lines) - 25
            If startAt < 0 Then startAt = 0
            For i = startAt To UBound(lines)
                tail = tail & lines(i) & vbLf
            Next
        End If
    End If
    MsgBox "The CODE Lab Imaging Pipeline could not start (exit code " & rc & ")." & vbLf & vbLf & _
           "Full log: " & logPath & vbLf & vbLf & _
           "If the conda environment has not been created yet, run" & vbLf & _
           "launch_codelab.bat once -- it prompts and does the setup." & vbLf & vbLf & _
           "--- end of log ---" & vbLf & tail, _
           vbExclamation, "CODE Lab Imaging Pipeline"
End If
