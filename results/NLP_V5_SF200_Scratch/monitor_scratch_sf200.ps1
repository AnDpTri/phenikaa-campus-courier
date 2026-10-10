# Monitor 3-seed Scratch V5 training with loss/head metrics and overfit diagnostics.
param([switch]$Once,[int]$Tail=12)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$metrics=Join-Path $root 'training_metrics.jsonl'
$log=Join-Path $root 'scratch_stdout.log'
$err=Join-Path $root 'scratch_stderr.log'
$pidfile=Join-Path $root 'scratch_training.pid'
Write-Host '=== V5-SF200-Scratch | 3 SEEDS x 15 EPOCHS ===' -ForegroundColor Cyan
if (Test-Path -LiteralPath $pidfile) {
    [int]$trainPid=(Get-Content -LiteralPath $pidfile -Raw).Trim()
    $proc=Get-Process -Id $trainPid -ErrorAction SilentlyContinue
    if ($proc) {Write-Host ('Training RUNNING | PID '+$trainPid) -ForegroundColor Green}
    else {Write-Host ('Training not running | Last PID '+$trainPid) -ForegroundColor Yellow}
} else {
    Write-Host 'Training has not been started.' -ForegroundColor Yellow
}
try {
    $gpu=& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null
    if ($gpu) {Write-Host ('GPU: '+($gpu -join ' '))}
} catch {}
$done=Join-Path $root 'scratch_final_report.json'
if (Test-Path -LiteralPath $done) {Write-Host 'TRAIN COMPLETED: final report and 3-net ensemble ready' -ForegroundColor Green}
if (Test-Path -LiteralPath $metrics) {
    $last=@(Get-Content -LiteralPath $metrics -Tail ([math]::Max($Tail,16)) -Encoding UTF8)
    $events=@()
    foreach($line in $last) {
        try {$events+=($line|ConvertFrom-Json)} catch {}
    }
    $lastEvent=$events|Select-Object -Last 1
    if ($lastEvent) {
        Write-Host ('Last event: '+$lastEvent.event+' | '+[DateTimeOffset]::FromUnixTimeSeconds([long]$lastEvent.time_unix).ToLocalTime())
        if ($null -ne $lastEvent.seed_index) {
            Write-Host ('Model '+([int]$lastEvent.seed_index+1)+'/3 | Epoch '+$lastEvent.epoch+'/15')
        }
    }
    $epochEvent=$events|Where-Object event -eq 'EPOCH_COMPLETE'|Select-Object -Last 1
    if ($epochEvent) {
        Write-Host '--- Last epoch diagnostics ---' -ForegroundColor Cyan
        Write-Host ('Train loss: '+[math]::Round($epochEvent.train_loss,4))
        Write-Host ('Seen train probe loss: '+[math]::Round($epochEvent.overfit.seen_train_probe_loss,4))
        Write-Host ('Heldout synthetic loss: '+[math]::Round($epochEvent.overfit.synthetic_holdout_mean_loss,4))
        Write-Host ('Generalization gap: '+[math]::Round($epochEvent.overfit.holdout_minus_seen_gap,4))
        Write-Host ('Overfit warning: '+$epochEvent.overfit.overfit_warning)
        foreach($type in @('v2','weighted','hard')) {
            $m=$epochEvent.synthetic_heldout.$type
            if ($m) {
                Write-Host ($type+': all='+[math]::Round($m.exact.all*100,2)+'% goal='+[math]::Round($m.exact.goal*100,2)+'% via='+[math]::Round($m.exact.via*100,2)+'% via_FPR='+[math]::Round($m.via_fpr*100,2)+'%')
            }
        }
    }
    $p=$events|Where-Object event -eq 'TRAIN_PROGRESS'|Select-Object -Last 1
    if ($p) {
        Write-Host ('Last batch update: '+$p.updates+'/'+$p.updates_per_epoch+' optimizer steps | loss '+[math]::Round($p.mean_train_loss,4)+' | LR '+$p.lr)
    }
    $warnings=@($events|Where-Object event -eq 'OVERFIT_WARNING')
    if ($warnings.Count) {
        Write-Host ('OVERFIT WARNINGS IN RECENT EVENTS: '+$warnings.Count) -ForegroundColor Yellow
    }
}
if (Test-Path -LiteralPath $log) {
    Write-Host '--- Last stdout lines ---'
    Get-Content -LiteralPath $log -Tail $Tail -Encoding UTF8
}
if (Test-Path -LiteralPath $err) {
    $lastError=@(Get-Content -LiteralPath $err -Tail 14 -Encoding UTF8)
    if ($lastError.Count -gt 0) {
        Write-Host '--- STDERR ---' -ForegroundColor Red
        $lastError
    }
}
if($Once){return}
Write-Host '--- LIVE LOG; Ctrl+C stops monitor only ---' -ForegroundColor Cyan
if(Test-Path -LiteralPath $log) {
    Get-Content -LiteralPath $log -Wait -Tail 0 -Encoding UTF8
} else {
    Write-Host 'Log not ready: rerun monitor after training launches.'
}
