param(
    [string]$Run = "pilot_LR1e4",
    [int]$RefreshSeconds = 15,
    [int]$TotalEpochs = 3
)
$ErrorActionPreference = "Stop"
$RefreshSeconds = [Math]::Max(3, $RefreshSeconds)
$runDir = Join-Path $PSScriptRoot $Run
$logPath = Join-Path $runDir "train.log"
$errPath = Join-Path $runDir "train.err.log"
$pidPath = Join-Path $runDir "train.pid"
if (-not (Test-Path -LiteralPath $runDir)) { throw "Run directory not found: $runDir" }
$shownLines = 0
$lastWarningEpoch = 0
Write-Host "Monitoring: $runDir" -ForegroundColor Cyan
Write-Host "Ctrl+C only stops this monitor. Training remains running." -ForegroundColor DarkYellow
Write-Host "Trainer writes evaluation metrics AFTER each epoch; in-epoch progress is estimated." -ForegroundColor DarkYellow

while ($true) {
    $epochResults = @()
    if (Test-Path -LiteralPath $logPath) {
        $lines = @(Get-Content -LiteralPath $logPath -Encoding UTF8)
        if ($lines.Count -gt $shownLines) {
            for ($i=$shownLines; $i -lt $lines.Count; $i++) {
                $line = [string]$lines[$i]
                if ($line -match 'epoch [0-9]+:') {
                    Write-Host $line -ForegroundColor Green
                } elseif ($line -match 'saved |hybrid thresholds|Traceback|Error') {
                    Write-Host $line -ForegroundColor Yellow
                } else {
                    Write-Host $line
                }
            }
            $shownLines = $lines.Count
        }
        foreach ($line in $lines) {
            if ($line -match 'epoch\s+(\d+):.*?validation all\s+([0-9.]+)\s+holdout-phrasing all\s+([0-9.]+).*?\s+(\d+)s\s*$') {
                $epochResults += [pscustomobject]@{
                    Epoch = [int]$matches[1]
                    Validation = [double]::Parse($matches[2],[Globalization.CultureInfo]::InvariantCulture)
                    Holdout = [double]::Parse($matches[3],[Globalization.CultureInfo]::InvariantCulture)
                    Seconds = [int]$matches[4]
                }
            }
        }
    }
    $trainPid = $null
    if (Test-Path -LiteralPath $pidPath) {
        $pidText = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($pidText -match '^[0-9]+$') { $trainPid = [int]$pidText }
    }
    $proc = $null
    if ($trainPid) { $proc = Get-Process -Id $trainPid -ErrorAction SilentlyContinue }
    $done = if ($epochResults.Count) { ($epochResults | Measure-Object Epoch -Maximum).Maximum } else { 0 }
    $currentEpoch = [Math]::Min($TotalEpochs, $done + 1)
    $elapsedSeconds = 0
    $etaText = "ETA unknown until first epoch completes"
    $epochPctText = "in-epoch % unknown"
    $barPercent = -1
    if ($proc) {
        $epochStart = $proc.StartTime
        if ($done -gt 0 -and (Test-Path -LiteralPath $logPath)) {
            $epochStart = (Get-Item -LiteralPath $logPath).LastWriteTime
        }
        $elapsedSeconds = [Math]::Max(0,[int]((Get-Date)-$epochStart).TotalSeconds)
        if ($done -gt 0 -and $done -lt $TotalEpochs) {
            $avgEpoch = [int](($epochResults | Measure-Object Seconds -Average).Average)
            if ($avgEpoch -gt 0) {
                $barPercent = [int](100 * $done / $TotalEpochs)
                $epochPctText = "completed epochs: $done/$TotalEpochs; batch progress unavailable"
                if ($elapsedSeconds -gt 2*$avgEpoch) {
                    $etaText = "WARNING: >2x previous epoch duration; ETA unreliable"
                } else {
                    $etaText = "previous epoch $avgEpoch sec; current epoch $elapsedSeconds sec"
                }
            }
        } elseif ($done -eq $TotalEpochs) {
            $epochPctText = "epoch training complete"
            $etaText = "final reporting / checkpoint save"
            $barPercent = 99
        }
    }
    if ($epochResults.Count -ge 2) {
        $last = $epochResults[$epochResults.Count-1]
        $prev = $epochResults[$epochResults.Count-2]
        if (($last.Validation + $last.Holdout) -lt ($prev.Validation + $prev.Holdout) -and $last.Epoch -gt $lastWarningEpoch) {
            Write-Host ("OVERFIT WATCH: epoch {0} combined validation+holdout score DROPPED vs epoch {1}. The trainer saves the best checkpoint, not necessarily the last." -f $last.Epoch,$prev.Epoch) -ForegroundColor Yellow
            $lastWarningEpoch = $last.Epoch
        }
    }
    $gpuText = "GPU unavailable"
    try {
        $gpuLine = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuLine) { $gpuText = "GPU %/usedMiB/totalMiB: " + (($gpuLine | Select-Object -First 1).Trim()) }
    } catch {}
    $state = if ($proc) { "RUNNING" } elseif ($trainPid) { "STOPPED" } else { "WAITING" }
    $statusText = if ($proc) {
        ("epoch {0}/{1} | elapsed {2:mm\:ss} | {3} | {4}" -f $currentEpoch,$TotalEpochs,[TimeSpan]::FromSeconds($elapsedSeconds),$epochPctText,$etaText)
    } else { "not running; check files" }
    Write-Host ("[{0:HH:mm:ss}] {1} PID={2} | {3} | {4}" -f (Get-Date),$state,$trainPid,$statusText,$gpuText) -ForegroundColor DarkCyan
    if ($proc) {
        Write-Progress -Id 1 -Activity ("NLP V4 curated 200k: epoch {0}/{1}" -f $currentEpoch,$TotalEpochs) -Status $statusText -PercentComplete $barPercent
    } else { Write-Progress -Id 1 -Activity "NLP V4 curated 200k" -Completed }
    if ($trainPid -and -not $proc) {
        if (Test-Path -LiteralPath $errPath) {
            $errors = @(Get-Content -LiteralPath $errPath -Tail 15)
            if ($errors.Count) { Write-Host "STDERR:" -ForegroundColor Yellow; $errors | ForEach-Object { Write-Host $_ } }
        }
        Write-Host "Process has ended. Check train.log, report.json and checkpoints." -ForegroundColor Cyan
        break
    }
    Start-Sleep -Seconds $RefreshSeconds
}
