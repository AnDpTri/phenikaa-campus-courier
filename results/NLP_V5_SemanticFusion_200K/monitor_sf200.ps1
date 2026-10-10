# View V5-SF200 in a terminal. Default: live follow; -Once: status snapshot.
param([int]$Tail = 25, [switch]$Once)
$ErrorActionPreference = 'Stop'
$run = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $run 'sf200_train.log'
$err = Join-Path $run 'sf200_train.err.log'
$pidFile = Join-Path $run 'sf200_train.pid'
$result = Join-Path $run 'sf200_evaluation.json'
Write-Host '=== Phenikaa NLP V5-SF200 | TRAIN MONITOR ===' -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host 'Training launch PID file is not present.' -ForegroundColor Yellow
    return
}
[int]$procId = (Get-Content -LiteralPath $pidFile -Raw).Trim()
Write-Host ('Training PID: ' + $procId)
$live = [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
Write-Host ('Process running: ' + $live)
try {
    $gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null
    if ($gpu) { Write-Host ('GPU: ' + ($gpu -join ' ')) }
} catch {}
if (Test-Path -LiteralPath $log) {
    Write-Host '--- Recent training log ---'
    Get-Content -LiteralPath $log -Tail $Tail -Encoding UTF8
} else {
    Write-Host 'Output log has not been created yet.'
}
if (Test-Path -LiteralPath $err) {
    $errors = @(Get-Content -LiteralPath $err -Tail 8 -Encoding UTF8)
    if ($errors.Count -gt 0) {
        Write-Host '--- STDERR (last 8 lines) ---' -ForegroundColor Yellow
        $errors
    }
}
if (Test-Path -LiteralPath $result) {
    Write-Host ('Completed evaluation: ' + $result) -ForegroundColor Green
}
if ($Once) { return }
Write-Host '--- Live updates (Ctrl+C to stop watching; training continues) ---' -ForegroundColor Cyan
$shown = if (Test-Path -LiteralPath $log) {
    @(Get-Content -LiteralPath $log -Encoding UTF8).Count
} else { 0 }
while ($true) {
    Start-Sleep -Seconds 4
    if (Test-Path -LiteralPath $log) {
        $lines = @(Get-Content -LiteralPath $log -Encoding UTF8)
        if ($lines.Count -gt $shown) {
            $lines[$shown..($lines.Count - 1)] | ForEach-Object { Write-Host $_ }
            $shown = $lines.Count
        }
    }
    if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        if (Test-Path -LiteralPath $result) {
            Write-Host 'TRAINING COMPLETED. Evaluation and checkpoints available.' -ForegroundColor Green
        } else {
            Write-Host 'TRAINING STOPPED before completing the final report. Check STDERR.' -ForegroundColor Red
            if (Test-Path -LiteralPath $err) {
                Get-Content -LiteralPath $err -Tail 30 -Encoding UTF8
            }
        }
        break
    }
}
