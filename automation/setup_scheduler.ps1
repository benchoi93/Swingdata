# setup_scheduler.ps1 — Register Windows Task Scheduler job
# Run this ONCE (elevated/admin) to set up the daily 10 PM agent
#
# Usage: Run as Administrator
#   powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1

$ErrorActionPreference = "Stop"

$TaskName    = "SwingData-DailyAgent"
$Description = "Runs Claude Code agent on e-scooter safety project daily at 10 PM for ~2 hours"
$ScriptPath  = "C:\Users\chois\Gitsrcs\Swingdata\automation\run_daily.ps1"

# --- Remove existing task if present ---
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# --- Create the scheduled task ---
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`"" `
    -WorkingDirectory "C:\Users\chois\Gitsrcs\Swingdata"

$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "10:00PM"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Run as current user (no password prompt needed for "Run only when user is logged on")
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description $Description

Write-Host ""
Write-Host "Task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host "Schedule: Daily at 10:00 PM"
Write-Host "Script:   $ScriptPath"
Write-Host "Timeout:  3 hours (hard limit)"
Write-Host ""
Write-Host "To verify:  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "To test:    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To remove:  Unregister-ScheduledTask -TaskName '$TaskName'"
Write-Host "To disable: Disable-ScheduledTask -TaskName '$TaskName'"
