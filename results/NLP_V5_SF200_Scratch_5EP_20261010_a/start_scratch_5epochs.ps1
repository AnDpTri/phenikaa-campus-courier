# Launch NEW five-epoch scratch training. Never overwrite any existing files.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$py=Join-Path $root 'train_scratch_sf200_5epochs.py'
$metrics=Join-Path $root 'training_metrics.jsonl'
$pidfile=Join-Path $root 'scratch_training.pid'
$out=Join-Path $root 'scratch_stdout.log'
$err=Join-Path $root 'scratch_stderr.log'
$final=Join-Path $root 'scratch_final_report.json'
$ensemble=Join-Path $root 'scratch_3seed_epoch05_ensemble.pt'
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
Write-Host ('SCRATCH 5 EPOCHS LAUNCHED PID '+$proc.Id) -ForegroundColor Green
Write-Host ('Output: '+$root)
