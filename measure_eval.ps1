# Measure peak RAM of an evaluate.py run (for the 4 GiB evaluation budget).
# Usage: .\measure_eval.ps1 -Checkpoint runs\full\checkpoint-avg.pt -Split test
# Runs the scorer in-process and samples WorkingSet64 every second; prints
# the scorer's own timing JSON at the end (already written by evaluate.py).
param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Split = 'test'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$proc = Start-Process -FilePath '.\.venv\Scripts\python.exe' `
    -ArgumentList @('evaluate.py', '--checkpoint', $Checkpoint, '--device', 'cpu', '--precision', 'fp32', '--split', $Split) `
    -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\measure_eval_out.txt" -RedirectStandardError "$env:TEMP\measure_eval_err.txt"
$peak = 0.0
while (-not $proc.HasExited) {
    $proc.Refresh()
    if ($proc.WorkingSet64 -gt $peak) { $peak = $proc.WorkingSet64 }
    Start-Sleep -Milliseconds 500
}
$proc.WaitForExit()
$proc.Refresh()
if ($proc.WorkingSet64 -gt $peak) { $peak = $proc.WorkingSet64 }
Write-Host ("Peak working set: {0:N0} MiB" -f ($peak / 1MB))
Get-Content "$env:TEMP\measure_eval_out.txt" -ErrorAction SilentlyContinue
Get-Content "$env:TEMP\measure_eval_err.txt" -ErrorAction SilentlyContinue
