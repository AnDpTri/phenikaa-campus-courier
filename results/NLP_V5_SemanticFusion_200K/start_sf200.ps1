# Launch V5-SF200 independently from this terminal. New files only.
$ErrorActionPreference = 'Stop'
$run = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainer = Join-Path $run 'train_sf200.py'
$qa = Join-Path $run 'sf200_epoch2_quality_audit.json'
$log = Join-Path $run 'sf200_train.log'
$err = Join-Path $run 'sf200_train.err.log'
$pidFile = Join-Path $run 'sf200_train.pid'
$protected = @($log, $err, $pidFile,
    (Join-Path $run 'sf200_e1.pt'),
    (Join-Path $run 'sf200_e2.pt'),
    (Join-Path $run 'sf200_best.pt'),
    (Join-Path $run 'sf200_evaluation.json'))
foreach ($f in $protected) {
    if (Test-Path -LiteralPath $f) {
        throw "Refusing to overwrite existing output: $f"
    }
}
if (-not (Test-Path -LiteralPath $qa)) {
    throw "Epoch 2 independent QA must complete before starting."
}
$audit = Get-Content -LiteralPath $qa -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $audit.passed -or $audit.total -ne 200000) {
    throw "Epoch 2 full-population QA not accepted."
}
$env:PYTHONPATH = 'D:\phenikaa\src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pyexe = (Get-Command py.exe -ErrorAction Stop).Source
$arguments = '-3.12 -B -u "' + $trainer + '"'
$process = Start-Process -FilePath $pyexe -ArgumentList $arguments -WorkingDirectory 'D:\phenikaa' -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
[System.IO.File]::WriteAllText($pidFile, [string]$process.Id + [Environment]::NewLine)
Write-Host ('V5-SF200 LAUNCHED; Python PID: ' + $process.Id)
Write-Host ('Log: ' + $log)
Write-Host ('Follow with: powershell -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $run 'monitor_sf200.ps1') + '"')
