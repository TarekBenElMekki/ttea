$ErrorActionPreference = "Stop"
$statePath = Join-Path $PSScriptRoot "..\data\runtime\benchmark.json"
$identityToleranceSeconds = 2.0
$shutdownTimeoutSeconds = 10.0

function Get-VerifiedProcess([object]$processId, [object]$startedEpoch) {
    if ($null -eq $processId -or $null -eq $startedEpoch) { return $null }
    try {
        $numericPid = [int]$processId
        $numericStart = [double]$startedEpoch
        if ($numericPid -le 0 -or $numericStart -le 0) { return $null }
        $process = Get-Process -Id $numericPid -ErrorAction Stop
        $actualEpoch = ([DateTimeOffset]$process.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
        if ([Math]::Abs($actualEpoch - $numericStart) -gt $identityToleranceSeconds) { return $null }
        return $process
    } catch { return $null }
}

function Wait-VerifiedProcessGone([int]$processId, [double]$startedEpoch) {
    $deadline = [DateTime]::UtcNow.AddSeconds($shutdownTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($null -eq (Get-VerifiedProcess $processId $startedEpoch)) { return $true }
        Start-Sleep -Milliseconds 100
    }
    return $null -eq (Get-VerifiedProcess $processId $startedEpoch)
}

if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Output "No active T/TEA benchmark PID state exists."
    exit 0
}

try {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $worker = Get-VerifiedProcess $state.worker_pid $state.worker_started_epoch
    $server = Get-VerifiedProcess $state.server_pid $state.server_started_epoch
    $supervisor = Get-VerifiedProcess $state.supervisor_pid $state.supervisor_started_epoch
} catch {
    Write-Output "Benchmark PID state is malformed; no process was stopped."
    exit 0
}

$failed = $false
if ($null -ne $worker) {
    $workerPid = [int]$state.worker_pid
    $workerStart = [double]$state.worker_started_epoch
    try { Stop-Process -Id $workerPid -Force -ErrorAction Stop } catch { $failed = $true }
    if (-not (Wait-VerifiedProcessGone $workerPid $workerStart)) { $failed = $true }
}

if ($null -ne $server) {
    $serverPid = [int]$state.server_pid
    $serverStart = [double]$state.server_started_epoch
    $descendants = @()
    try {
        $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop)
        $frontier = @($serverPid)
        while ($frontier.Count -gt 0) {
            $parentPid = $frontier[0]
            if ($frontier.Count -eq 1) { $frontier = @() } else { $frontier = @($frontier[1..($frontier.Count - 1)]) }
            foreach ($candidate in @($allProcesses | Where-Object { [int]$_.ParentProcessId -eq $parentPid })) {
                $candidateStart = ([DateTimeOffset]$candidate.CreationDate.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
                $candidateProcess = Get-VerifiedProcess ([int]$candidate.ProcessId) $candidateStart
                if ($null -ne $candidateProcess) {
                    $descendants += [pscustomobject]@{ Pid = $candidateProcess.Id; Started = $candidateStart }
                    $frontier += $candidateProcess.Id
                }
            }
        }
    } catch { $failed = $true }
    [array]::Reverse($descendants)
    foreach ($owned in $descendants) {
        if ($null -ne (Get-VerifiedProcess $owned.Pid $owned.Started)) {
            try { Stop-Process -Id $owned.Pid -Force -ErrorAction Stop } catch { $failed = $true }
            if (-not (Wait-VerifiedProcessGone $owned.Pid $owned.Started)) { $failed = $true }
        }
    }
    try { Stop-Process -Id $serverPid -Force -ErrorAction Stop } catch { $failed = $true }
    if (-not (Wait-VerifiedProcessGone $serverPid $serverStart)) { $failed = $true }
}

$workerStillActive = $null -ne (Get-VerifiedProcess $state.worker_pid $state.worker_started_epoch)
$serverStillActive = $null -ne (Get-VerifiedProcess $state.server_pid $state.server_started_epoch)
if ($failed -or $workerStillActive -or $serverStillActive) {
    Write-Output "Recorded benchmark shutdown could not be confirmed; ownership state was preserved."
    exit 1
}
if ($null -ne $supervisor) {
    Write-Output "Recorded worker and server are stopped; active supervisor will release ownership state."
    exit 0
}
Remove-Item -LiteralPath $statePath -Force
Write-Output "Recorded T/TEA benchmark worker and Ollama server are confirmed stopped."
