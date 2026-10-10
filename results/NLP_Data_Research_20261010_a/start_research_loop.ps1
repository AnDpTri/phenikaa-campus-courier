$ErrorActionPreference='Stop'
$root='D:\phenikaa\results\NLP_Data_Research_20261010_a'
$script=Join-Path $root 'research_loop_4h_v1.py'
$pidFile=Join-Path $root 'research_loop.pid'
$stdout=Join-Path $root 'research_loop_stdout.log'
$stderr=Join-Path $root 'research_loop_stderr.log'
foreach($path in @($pidFile,$stdout,$stderr,(Join-Path $root 'checkpoints\round_000.json'))){
 if(Test-Path -LiteralPath $path){throw ('Refusing duplicate: '+$path)}
}
if(!(Test-Path -LiteralPath $script)){throw 'Missing research script'}
$env:PYTHONPATH='D:\phenikaa\src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$exe=(Get-Command py.exe -ErrorAction Stop).Source
$args='-3.12 -B -u "'+$script+'"'
$proc=Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory 'D:\phenikaa' -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$txt=[System.Text.Encoding]::UTF8.GetBytes([string]$proc.Id+[Environment]::NewLine)
$f=[System.IO.File]::Open($pidFile,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
try{$f.Write($txt,0,$txt.Length)}finally{$f.Dispose()}
Write-Output ('FOUR_HOUR_RESEARCH_LAUNCHED PID='+$proc.Id)
Write-Output ('CHECKPOINT_DIR='+ (Join-Path $root 'checkpoints'))