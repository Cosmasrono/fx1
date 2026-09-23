# Registers run_always.ps1 as a Windows scheduled task for the current user.
#
# Runs at logon (no admin rights needed) and every 10 minutes as a safety net:
# the task engine skips a start when the task is already running, so a live
# engine is never restarted, while a machine that was asleep resumes within
# ten minutes instead of waiting for the next logon.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root 'run_always.ps1'
$name = 'fx1-paper-engine'

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $root
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 10)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $name -Action $action -Trigger @($logon, $repeat) `
    -Settings $settings -Description 'EUR/USD paper trading engine (paper only)' -Force | Out-Null
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Format-List TaskName, State
