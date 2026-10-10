# Run-once, fail-closed post-training evaluation launcher.
# Invoked by the hourly automation ONLY AFTER full Scratch completion.
# Never edits old checkpoints/sources and never loads official validation/test.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$trainingPidFile=Join-Path $root 'scratch_training.pid'
$trainingReport=Join-Path $root 'scratch_final_report.json'
$ensemble=Join-Path $root 'scratch_3seed_epoch15_ensemble.pt'
$trainMetrics=Join-Path $root 'training_metrics.jsonl'
$evalScript=Join-Path $root 'posttrain_evaluator.py'
$evalPid=Join-Path $root 'posttrain_evaluation.pid'
$evalLog=Join-Path $root 'posttrain_stdout.log'
$evalError=Join-Path $root 'posttrain_stderr.log'
$report=Join-Path $root 'posttrain_analysis_report.md'
$results=Join-Path $root 'posttrain_analysis_results.json'
$examples=Join-Path $root 'posttrain_discordant_examples.jsonl'
$curves=Join-Path $root 'posttrain_learning_curves.csv'
foreach($path in @($evalPid,$evalLog,$evalError,$report,$results,$examples,$curves)){
    if(Test-Path -LiteralPath $path) {
        throw ('Posttraining already started or output exists; refusing duplicate: '+$path)
    }
}
if(-not (Test-Path -LiteralPath $trainingPidFile)){throw 'Training PID file missing'}
[int]$trainingPid=(Get-Content -LiteralPath $trainingPidFile -Raw).Trim()
$active=Get-Process -Id $trainingPid -ErrorAction SilentlyContinue
if($active){throw ('Training still active PID '+$trainingPid+'; do not evaluate yet')}
foreach($needed in @($trainingReport,$ensemble,$trainMetrics,$evalScript)){
    if(-not (Test-Path -LiteralPath $needed)){throw ('Missing completed training artifact: '+$needed)}
}
$meta=Get-Content -LiteralPath $trainingReport -Raw -Encoding UTF8 | ConvertFrom-Json
if($meta.status -ne 'DONE'){throw 'Final report status not DONE'}
$last=(Get-Content -LiteralPath $trainMetrics -Tail 1 -Encoding UTF8) | ConvertFrom-Json
if($last.event -ne 'RUN_COMPLETE'){throw 'Training did not write RUN_COMPLETE; refusing incomplete evaluation'}
$env:PYTHONPATH='D:\phenikaa\src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$exe=(Get-Command py.exe -ErrorAction Stop).Source
$args='-3.12 -B -u "'+$evalScript+'"'
$proc=Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory 'D:\phenikaa' -RedirectStandardOutput $evalLog -RedirectStandardError $evalError -WindowStyle Hidden -PassThru
$pidBytes=[System.Text.Encoding]::UTF8.GetBytes([string]$proc.Id+[Environment]::NewLine)
$handle=[System.IO.File]::Open($evalPid,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
try{$handle.Write($pidBytes,0,$pidBytes.Length)}finally{$handle.Close()}
Write-Host ('POSTTRAIN_EVALUATION_STARTED PID '+$proc.Id) -ForegroundColor Green
Write-Host ('Results: '+$results)
Write-Host ('Report: '+$report)
