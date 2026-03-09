# test_run.ps1 — Quick test of the agent pipeline with a small budget cap

$ProjectDir = "C:\Users\chois\Gitsrcs\Swingdata"
$PromptFile = "$ProjectDir\automation\agent_prompt.md"
$LogDir     = "$ProjectDir\automation\logs"
$ReportsDir = "$ProjectDir\reports"
$Date       = Get-Date -Format "yyyy-MM-dd"
$LogFile    = "$LogDir\agent_${Date}_test.log"
$BudgetUSD  = 3

if (-not (Test-Path $LogDir))     { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null }

$AgentInstructions = Get-Content $PromptFile -Raw

$Prompt = @"
$AgentInstructions

---

TODAY'S DATE: $Date

THIS IS A TEST RUN with a small budget. Read TASKS.md, work on task 1.1 (profile the dataset schema and completeness), and write your daily report to reports/$Date.md. Focus on completing just this one task to verify the pipeline works.
"@

# Write prompt to temp file to avoid argument length issues
$TempPrompt = "$LogDir\_prompt_temp.txt"
[System.IO.File]::WriteAllText($TempPrompt, $Prompt, [System.Text.Encoding]::UTF8)

$StartTime = Get-Date
Write-Host "[$StartTime] Starting TEST agent session (budget: $BudgetUSD USD)"
Write-Host "Prompt length: $($Prompt.Length) chars"

Set-Location $ProjectDir

# Pipe prompt via stdin to claude
# -p = print mode (non-interactive), prompt comes from stdin when piped
$allOutput = Get-Content $TempPrompt -Raw | claude -p --dangerously-skip-permissions --max-budget-usd $BudgetUSD --verbose 2>&1

# Write to log and console
$allOutput | Out-File -FilePath $LogFile -Encoding utf8
$allOutput | ForEach-Object { Write-Host $_ }

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "[$EndTime] Test completed. Duration: $($Duration.TotalMinutes.ToString('F1')) min"

if (Test-Path "$ReportsDir\$Date.md") {
    Write-Host "SUCCESS: Report generated at reports\$Date.md" -ForegroundColor Green
} else {
    Write-Host "WARNING: No report file found" -ForegroundColor Yellow
}

# Cleanup
Remove-Item $TempPrompt -ErrorAction SilentlyContinue
