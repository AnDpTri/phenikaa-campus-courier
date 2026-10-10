# Start V5-SF200-Scratch once; fail if any output already exists.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$py=Join-Path $root 'train_scratch_sf200_v2.py'
$metrics=Join-Path $root 'training_metrics.jsonl'
$pidfile=Join-Path $root 'scratch_training.pid'
$out=Join-Path $root 'scratch_stdout.log'
$err=Join-Path $root 'scratch_stderr.log'
$final=Join-Path $root 'scratch_final_report.json'
$ensemble=Join-Path $root 'scratch_3seed_epoch15_ensemble.pt'
foreach ($path in @($pidfile,$out,$err,$metrics,$final,$ensemble)) {
    if(Test-Path -LiteralPath $path){throw "Refusing to overwrite existing path: $path"}
}
if (-not (Test-Path -LiteralPath $py)) {throw 'Trainer script missing'}
$env:PYTHONPATH='D:\phenikaa\src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$exe=(Get-Command py.exe -ErrorAction Stop).Source
$args='-3.12 -B -u "'+$py+'"'
$proc=Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory 'D:\phenikaa' -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
$pidBytes=[System.Text.Encoding]::UTF8.GetBytes([string]$proc.Id+[Environment]::NewLine)
$handle=[System.IO.File]::Open($pidfile,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
try{$handle.Write($pidBytes,0,$pidBytes.Length)}finally{$handle.Close()}
Write-Host ('SCRATCH TRAIN LAUNCHED | PID '+$proc.Id) -ForegroundColor Green
Write-Host ('Loss + overfit metrics: '+$metrics)
Write-Host ('Log: '+$out)
Write-Host ('Monitor: powershell -NoProfile -ExecutionPolicy Bypass -File "'+(Join-Path $root 'monitor_scratch_sf200.ps1')+'"')
