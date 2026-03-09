# check_status.ps1 — Quick status check for the daily agent
# Shows: last report, task progress, next scheduled run

$ProjectDir = "C:\Users\chois\Gitsrcs\Swingdata"

Write-Host "=== SwingData Agent Status ===" -ForegroundColor Cyan
Write-Host ""

# --- Task progress ---
$tasks = Get-Content "$ProjectDir\TASKS.md" -Raw
$total    = ([regex]::Matches($tasks, '- \[[ x~!]\]')).Count
$done     = ([regex]::Matches($tasks, '- \[x\]')).Count
$progress = ([regex]::Matches($tasks, '- \[~\]')).Count
$blocked  = ([regex]::Matches($tasks, '- \[!\]')).Count
$pending  = $total - $done - $progress - $blocked
$pct      = if ($total -gt 0) { [math]::Round($done / $total * 100, 1) } else { 0 }

Write-Host "Task Progress: $done/$total completed ($pct%)" -ForegroundColor Green
Write-Host "  Pending: $pending | In Progress: $progress | Blocked: $blocked"
Write-Host ""

# --- Latest report ---
$reports = Get-ChildItem "$ProjectDir\reports\*.md" -ErrorAction SilentlyContinue | Sort-Object Name -Descending
if ($reports) {
    $latest = $reports[0]
    Write-Host "Latest Report: $($latest.Name)" -ForegroundColor Yellow
    Write-Host "---"
    # Show first 20 lines
    Get-Content $latest.FullName -TotalCount 20
    Write-Host "---"
} else {
    Write-Host "No reports yet." -ForegroundColor DarkGray
}
Write-Host ""

# --- Latest log ---
$logs = Get-ChildItem "$ProjectDir\automation\logs\*.log" -ErrorAction SilentlyContinue | Sort-Object Name -Descending
if ($logs) {
    $latestLog = $logs[0]
    $lastLine = Get-Content $latestLog.FullName -Tail 3
    Write-Host "Latest Log ($($latestLog.Name)):" -ForegroundColor DarkYellow
    $lastLine | ForEach-Object { Write-Host "  $_" }
}
Write-Host ""

# --- Scheduler status ---
$task = Get-ScheduledTask -TaskName "SwingData-DailyAgent" -ErrorAction SilentlyContinue
if ($task) {
    $info = $task | Get-ScheduledTaskInfo
    Write-Host "Scheduler: $($task.State)" -ForegroundColor $(if ($task.State -eq 'Ready') {'Green'} else {'Yellow'})
    Write-Host "  Last run: $($info.LastRunTime)"
    Write-Host "  Next run: $($info.NextRunTime)"
    Write-Host "  Last result: $($info.LastTaskResult)"
} else {
    Write-Host "Scheduler: NOT SET UP" -ForegroundColor Red
    Write-Host "  Run setup_scheduler.ps1 as administrator to enable daily runs."
}
