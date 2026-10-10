"""Create a five-epoch restart as NEW files only; never edit legacy run."""
from pathlib import Path
import hashlib,json,ast,shutil
root=Path(r"D:\phenikaa\results")
src=root/"NLP_V5_SF200_Scratch"
dst=root/"NLP_V5_SF200_Scratch_5EP_20261010_a"
assert dst.is_dir() and src.is_dir()
orig=(src/"train_scratch_sf200_v2.py").read_text(encoding="utf-8")
assert 'RUN=ROOT/"results/NLP_V5_SF200_Scratch"' in orig
assert 'CFG["epochs_per_seed"]==15' in orig
assert 'assert len(seed_diagnostics)==15' in orig
assert '"selected_epoch=15"' not in orig
assert 'selected_epoch=15' in orig
assert 'checkpoint_path(seed_i,15)' in orig
out=orig.replace("V5-SF200-Scratch: from-zero, 3-seed, 15-epoch","V5-SF200-Scratch: from-zero, 3-seed, 5-epoch")
out=out.replace('RUN=ROOT/"results/NLP_V5_SF200_Scratch"','RUN=ROOT/"results/NLP_V5_SF200_Scratch_5EP_20261010_a"')
out=out.replace('CFG["epochs_per_seed"]==15','CFG["epochs_per_seed"]==5')
out=out.replace('assert len(seed_diagnostics)==15','assert len(seed_diagnostics)==5')
out=out.replace('selected_epoch=15','selected_epoch=5')
out=out.replace('checkpoint_path(seed_i,15)','checkpoint_path(seed_i,5)')
out=out.replace('epoch15','epoch05').replace('epoch 15','epoch 5').replace('15th epoch','5th epoch')
out=out.replace('selected_epoch":15','selected_epoch":5').replace('epoch 15 checkpoints','epoch 5 checkpoints')
out=out.replace('FIXED_LAST_EPOCH_NO_VALIDATION_SELECTION','FIXED_LAST_EPOCH_NO_VALIDATION_SELECTION')
out=out.replace('15 epochs','5 epochs')
# Important: all programmatic epoch loops and OneCycle total steps derive from CFG.
assert 'CFG["epochs_per_seed"]==15' not in out
assert 'assert len(seed_diagnostics)==15' not in out
assert 'checkpoint_path(seed_i,15)' not in out
ast.parse(out)
(dst/"train_scratch_sf200_5epochs.py").open("x",encoding="utf-8",newline="\n").write(out)
cfg=json.loads((src/"scratch_config.json").read_text(encoding="utf-8"))
assert cfg["epochs_per_seed"]==15
cfg["name"]="V5-SF200-Scratch-5Epochs"
cfg["epochs_per_seed"]=5
cfg["scheduler"]="OneCycleLR across 5 epochs per seed"
cfg["diagnostic_policy"]="DIAGNOSTICS ONLY. Fixed epoch 5 checkpoint (no official validation). No automatic test/assessment after training."
cfg["file_safety"]="NEW dir results/NLP_V5_SF200_Scratch_5EP_20261010_a, create-only, original interrupted run preserved"
(dst/"scratch_config.json").open("x",encoding="utf-8").write(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n")
for name,key in [("synth_weighted_v4.py","generator_sha256"),("synth_v2_clone.py","v2_clone_sha256")]:
    with (src/name).open("rb") as fin,(dst/name).open("xb") as fout:shutil.copyfileobj(fin,fout)
    assert hashlib.sha256((dst/name).read_bytes()).hexdigest()==cfg[key]
launcher=r'''# Launch NEW five-epoch scratch training. Never overwrite any existing files.
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
'''
(dst/"start_scratch_5epochs.ps1").open("x",encoding="utf-8",newline="\n").write(launcher)
assert sum("RUN=ROOT/\"results/NLP_V5_SF200_Scratch_5EP_20261010_a\"" in x for x in [out])==1
assert "total_steps=CFG[\"epochs_per_seed\"]*updates" in out
assert "range(1,CFG[\"epochs_per_seed\"]+1)" in out
print("READY_FOR_LAUNCH NEW_RUN",dst)
print("EPOCHS",cfg["epochs_per_seed"],"SEEDS",cfg["model_seeds"],"CYCLES",cfg["epochs_per_seed"]*len(cfg["model_seeds"]))
print("TRAINER_SHA256",hashlib.sha256(out.encode()).hexdigest())
