# run_daily.ps1 — Daily autonomous agent launcher
# Runs Claude Code for ~2 hours on the e-scooter safety project
#
# Usage: powershell -ExecutionPolicy Bypass -File run_daily.ps1
# Scheduled via Windows Task Scheduler at 10:00 PM daily

$ErrorActionPreference = "Continue"

# --- Configuration ---
$ProjectDir = "C:\Users\chois\Gitsrcs\Swingdata"
$PromptFile = "$ProjectDir\automation\agent_prompt.md"
$ReportsDir = "$ProjectDir\reports"
$LogDir     = "$ProjectDir\automation\logs"
$Date       = Get-Date -Format "yyyy-MM-dd"
$LogFile    = "$LogDir\agent_$Date.log"
$BudgetUSD  = 15        # Max spend per session (~2 hours of heavy tool use)

# --- Setup ---
if (-not (Test-Path $LogDir))     { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null }

# --- Build the prompt ---
$AgentInstructions = Get-Content $PromptFile -Raw

$Prompt = @"
$AgentInstructions

---

TODAY'S DATE: $Date

Begin your work session now. Read TASKS.md to find where you left off, then work on the next task. You have approximately 2 hours. Write your daily report to reports/$Date.md before finishing.
"@

# --- Log start ---
$StartTime = Get-Date
"[$StartTime] Starting daily agent session (budget: $BudgetUSD USD)" | Out-File -FilePath $LogFile -Encoding utf8

Set-Location $ProjectDir

# --- Run Claude Code ---
# Pipe prompt via stdin to handle long prompts; merge stderr into stdout
$allOutput = $Prompt | claude -p --dangerously-skip-permissions --max-budget-usd $BudgetUSD --verbose 2>&1

# --- Write output to log ---
$allOutput | Out-File -FilePath $LogFile -Append -Encoding utf8

# --- Log completion ---
$EndTime = Get-Date
$Duration = $EndTime - $StartTime
"[$EndTime] Session completed. Duration: $($Duration.TotalMinutes.ToString('F1')) minutes" | Out-File -FilePath $LogFile -Append -Encoding utf8

# --- Verify report was written ---
$ReportFile = "$ReportsDir\$Date.md"
if (Test-Path $ReportFile) {
    "Report generated: $ReportFile" | Out-File -FilePath $LogFile -Append -Encoding utf8
} else {
    @"
# Daily Progress Report — $Date

## Session Summary
Automated session ran but did not produce a structured report. Check logs at:
``automation/logs/agent_$Date.log``

## Duration
$($Duration.TotalMinutes.ToString('F1')) minutes
"@ | Out-File -FilePath $ReportFile -Encoding utf8
    "WARNING: Agent did not write report. Fallback report created." | Out-File -FilePath $LogFile -Append -Encoding utf8
}
