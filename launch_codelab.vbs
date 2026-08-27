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

Dim shell, fso, here, tempDir, logPath, cmd, rc
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)

' Logs live in log\ BESIDE the app, not in %TEMP% -- a log you have to go
' hunting for is one nobody reads, but one log per launch dropped in the
' project root buries the actual project. The folder is committed (git
' tracks log\.gitkeep and ignores everything else in it) so it is normally
' already there; creating it is only for a fresh checkout that lost it.
'
' Falls back to %TEMP% when that folder cannot be made or written (an
' install under a protected path, or a read-only share), because failing
' to launch over a log file would be absurd.
tempDir = here & "\log"
If Not EnsureFolder(tempDir) Then
    tempDir = shell.ExpandEnvironmentStrings("%TEMP%")
ElseIf Not FolderIsWritable(tempDir) Then
    tempDir = shell.ExpandEnvironmentStrings("%TEMP%")
End If

' ONE LOG PER LAUNCH. A fixed name was wrong as soon as two copies of the
' app run at once -- nothing stops that, since each double-click starts an
' independent wscript. Both redirected into the same file with ">", so the
' second launch truncated the first's log while it was still being written,
' the two then interleaved, and a failure dialog could show the OTHER
' instance's output. Stamping the name keeps each launch's log its own.
logPath = ClaimLogPath(tempDir)

' Nothing else cleans these up, so sweep launch logs older than a week --
' enough to still have yesterday's failure to look at, without letting
' them accumulate in the project folder for ever.
PruneOldLogs tempDir, 7

Function EnsureFolder(folder)
    ' True if the folder exists afterwards. Two launches racing here both
    ' try to create it and one loses -- that loser's CreateFolder raises,
    ' so the existence re-check after the error is what decides, not the
    ' error itself.
    On Error Resume Next
    If fso.FolderExists(folder) Then
        EnsureFolder = True
        Exit Function
    End If
    fso.CreateFolder folder
    Err.Clear
    EnsureFolder = fso.FolderExists(folder)
End Function

Function FolderIsWritable(folder)
    ' The probe name must be unique per launch too. A fixed one was itself
    ' a collision: two launches starting together would each create and
    ' then DELETE the same probe, so one could delete the other's file
    ' mid-check and wrongly conclude the folder was read-only, sending its
    ' log somewhere the failure dialog did not expect. GetTempName is
    ' exactly what it is for.
    On Error Resume Next
    Dim probe, f
    probe = folder & "\" & fso.GetTempName()
    Set f = fso.CreateTextFile(probe, False)
    If Err.Number <> 0 Then
        FolderIsWritable = False
        Err.Clear
        Exit Function
    End If
    f.Close
    fso.DeleteFile probe, True
    FolderIsWritable = (Err.Number = 0)
    Err.Clear
End Function

Function ClaimLogPath(folder)
    ' CLAIMS the name by creating the file, rather than picking a name and
    ' hoping. A timestamp alone collides when two launches land in the same
    ' second, and adding Rnd() does not close it: Randomize seeds from the
    ' system timer, whose resolution is coarser than the gap between two
    ' processes started together, so both can draw the SAME number.
    '
    ' CreateTextFile(path, False) fails if the file already exists, so
    ' whichever launch creates it first owns that name and the other moves
    ' on. The redirect below then truncates the empty file we just made,
    ' which is what it would have done anyway.
    On Error Resume Next
    Dim n, base, candidate, i, f
    Randomize
    n = Now()
    base = folder & "\codelab_launch_" & _
           Year(n) & Pad2(Month(n)) & Pad2(Day(n)) & "_" & _
           Pad2(Hour(n)) & Pad2(Minute(n)) & Pad2(Second(n))
    For i = 0 To 999
        candidate = base
        If i > 0 Then candidate = candidate & "_" & CStr(i)
        candidate = candidate & ".log"
        Err.Clear
        Set f = fso.CreateTextFile(candidate, False)
        If Err.Number = 0 Then
            f.Close
            ClaimLogPath = candidate
            Err.Clear
            Exit Function
        End If
    Next
    ' 1000 collisions in one second is not a real scenario; if it somehow
    ' happens, a random name is still better than failing to launch.
    ClaimLogPath = base & "_" & CStr(Int(Rnd() * 1000000)) & ".log"
    Err.Clear
End Function

Function Pad2(v)
    Pad2 = Right("0" & CStr(v), 2)
End Function

Sub PruneOldLogs(folder, maxAgeDays)
    On Error Resume Next          ' housekeeping must never block a launch
    Dim f, file
    Set f = fso.GetFolder(folder)
    For Each file In f.Files
        If LCase(Left(file.Name, 16)) = "codelab_launch_2" Then
            If DateDiff("d", file.DateLastModified, Now()) > maxAgeDays Then
                file.Delete True
            End If
        End If
    Next
End Sub

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
