$ErrorActionPreference = "Stop"
$statePath = Join-Path $PSScriptRoot "..\data\runtime\benchmark.json"

if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Output "No active T/TEA benchmark PID state exists."
    exit 0
}

try {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $workerPid = [int]$state.worker_pid
    $expectedStart = [double]$state.worker_started_epoch
    if ($workerPid -le 0 -or $expectedStart -le 0) { throw "missing worker identity" }
    $process = Get-Process -Id $workerPid -ErrorAction Stop
    $actualStart = [DateTimeOffset]$process.StartTime.ToUniversalTime()
    $actualEpoch = $actualStart.ToUnixTimeMilliseconds() / 1000.0
    if ([Math]::Abs($actualEpoch - $expectedStart) -gt 2.0) {
        Write-Output "Recorded benchmark PID identity does not match; no process was stopped."
        exit 0
    }
    Stop-Process -Id $workerPid -Force -ErrorAction Stop
    Remove-Item -LiteralPath $statePath -Force
    Write-Output "Stopped recorded T/TEA benchmark worker PID $workerPid."
} catch {
    Write-Output "Benchmark PID state is stale or invalid; no process was stopped."
    exit 0
}
