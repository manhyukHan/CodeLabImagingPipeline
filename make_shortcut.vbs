' Puts a CODE Lab Imaging Pipeline shortcut -- with the app's own icon --
' next to the app and on the Desktop. Double-click this once.
'
' WHY A SHORTCUT AND NOT THE .vbs ITSELF. A .vbs cannot carry an icon.
' Explorer draws every .vbs with the Windows Script Host icon, and there
' is no field in the file, the filesystem or the registry that changes
' that for one script without re-associating the whole .vbs type for the
' machine. A .lnk is the supported way to attach an icon to something
' that is not an .exe, and assets\codelab_o.ico already exists for
' exactly this (see the note in main.py).
'
' The shortcut launches wscript.exe explicitly rather than the .vbs by
' association: on a machine where .vbs has been re-associated -- to
' Notepad, or to a policy handler that blocks scripts -- double-clicking
' the file opens an editor instead of running it. Naming the host removes
' that dependency.
'
' The .lnk is NOT committed: it stores absolute paths, so it is only ever
' valid on the machine that made it. Re-run this after moving the folder.

Option Explicit

Dim shell, fso, here, target, icon, madeHere, madeDesktop, msg
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Under CSCRIPT (a console) say everything on stdout; under WSCRIPT (a
' double-click) use dialogs. A double-clicked script has nowhere to print,
' and a scripted one must never block on a dialog nobody will click --
' which is exactly what happened the first time this was run from a
' terminal: the shortcut was written and the process then waited forever
' on an OK button.
Dim isConsole
isConsole = (LCase(fso.GetFileName(WScript.FullName)) = "cscript.exe")

Sub Say(text, level, title)
    If isConsole Then
        WScript.Echo text
    Else
        MsgBox text, level, title
    End If
End Sub

here = fso.GetParentFolderName(WScript.ScriptFullName)
target = here & "\launch_codelab.vbs"
icon = here & "\assets\codelab_o.ico"

If Not fso.FileExists(target) Then
    Say "launch_codelab.vbs is not beside this script." & vbLf & vbLf & _
        "Expected: " & target, vbCritical, "CODE Lab Imaging Pipeline"
    WScript.Quit 1
End If
If Not fso.FileExists(icon) Then
    ' the icon is the whole point of this script, so its absence is worth
    ' saying out loud rather than silently producing a default-icon .lnk
    Say "The app icon is missing." & vbLf & vbLf & _
        "Expected: " & icon & vbLf & vbLf & _
        "tools\make_app_icon.py regenerates it.", _
        vbCritical, "CODE Lab Imaging Pipeline"
    WScript.Quit 1
End If

madeHere = MakeShortcut(here & "\CODE Lab Imaging Pipeline.lnk")
madeDesktop = MakeShortcut(shell.SpecialFolders("Desktop") & _
                           "\CODE Lab Imaging Pipeline.lnk")

msg = ""
If madeHere <> "" Then msg = msg & madeHere & vbLf
If madeDesktop <> "" Then msg = msg & madeDesktop & vbLf
If msg = "" Then
    Say "No shortcut could be written -- both the app folder and the " & _
        "Desktop refused.", vbExclamation, "CODE Lab Imaging Pipeline"
    WScript.Quit 1
End If
Say "Shortcut created:" & vbLf & vbLf & msg & vbLf & _
    "Double-click it to start the app.", _
    vbInformation, "CODE Lab Imaging Pipeline"

Function MakeShortcut(path)
    ' Returns the path written, or "" when it could not be. A Desktop that
    ' is redirected to a disconnected network share is a real case, and it
    ' must not stop the local shortcut from being made.
    On Error Resume Next
    Dim lnk
    Set lnk = shell.CreateShortcut(path)
    lnk.TargetPath = shell.ExpandEnvironmentStrings("%WINDIR%") & _
                     "\System32\wscript.exe"
    lnk.Arguments = """" & target & """"
    ' WorkingDirectory matters: launch_codelab.bat resolves everything
    ' relative to its own folder, but the .lnk is what sets the process's
    ' starting directory and a wrong one would surface as a confusing
    ' "file not found" long after the click.
    lnk.WorkingDirectory = here
    lnk.IconLocation = icon & ",0"
    lnk.Description = "CODE Lab Imaging Pipeline"
    lnk.WindowStyle = 1
    lnk.Save
    If Err.Number = 0 And fso.FileExists(path) Then
        MakeShortcut = path
    Else
        MakeShortcut = ""
    End If
    Err.Clear
End Function
