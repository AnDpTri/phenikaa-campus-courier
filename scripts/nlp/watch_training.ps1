# NLP v3 training progress viewer
#
# Mục đích:
#   Stream log train trong terminal, đồng thời hiển thị epoch mới, PID,
#   CPU/RAM, GPU/VRAM/nhiệt độ, stderr và trạng thái checkpoint.
#
# Cấu hình mặc định đang theo dõi:
#   Log       = D:\phenikaa\results\nlp_neural_v3_resume\training.log
#   Error log = D:\phenikaa\results\nlp_neural_v3_resume\training.stderr.log
#   Process   = PID 32316 (tự tìm lại tiến trình train nếu PID này đổi)
#   Checkpoint= D:\phenikaa\results\nlp_v3_resume_ckpt
#
# Cách chạy PowerShell 5/Windows PowerShell:
#   cd D:\phenikaa
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\nlp\watch_training.ps1
#
# Chạy kiểm tra một lần rồi thoát:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\nlp\watch_training.ps1 -Once
#
# Tùy chọn:
#   -Log <path> -ErrorLog <path> -ProcessId <pid>
#   -PollSeconds 2 -StatusSeconds 30
#
# Ctrl+C chỉ dừng cửa sổ theo dõi; không dừng tiến trình train.

param(
    [string]$Log = "D:\phenikaa\results\nlp_neural_v3\full_run\training.log",
    [string]$ErrorLog = "D:\phenikaa\results\nlp_neural_v3\full_run\training.stderr.log",
    [string]$CheckpointDir = "D:\phenikaa\results\nlp_v3_ckpt",
    [int]$ProcessId = 21968,
    [int]$PollSeconds = 2,
    [int]$StatusSeconds = 30,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$stdoutLines = 0
$stderrLines = 0
$lastStatus = [datetime]::MinValue

function Find-TrainingProcess {
    param([int]$PreferredId)

    if ($PreferredId -gt 0) {
        $preferred = Get-Process -Id $PreferredId -ErrorAction SilentlyContinue
        if ($preferred) { return $preferred }
    }

    $candidate = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python(\.exe)?$' -and
            $_.CommandLine -match 'train_neural_parser\.py'
        } |
        Select-Object -First 1
    if ($candidate) {
        return Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
    }
    return $null
}

function Write-NewLines {
    param(
        [string]$Path,
        [ref]$Seen,
        [ConsoleColor]$Color
    )

    if (-not (Test-Path -LiteralPath $Path)) { return }
    $content = @(Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue)
    if ($content.Count -lt $Seen.Value) { $Seen.Value = 0 }
    if ($content.Count -gt $Seen.Value) {
        $content[$Seen.Value..($content.Count - 1)] |
            ForEach-Object { Write-Host $_ -ForegroundColor $Color }
        $Seen.Value = $content.Count
    }
}

function Write-Status {
    param([System.Diagnostics.Process]$TrainingProcess)

    $stamp = Get-Date -Format "HH:mm:ss"
    $processText = if ($TrainingProcess) {
        "PID=$($TrainingProcess.Id) CPU=$([math]::Round($TrainingProcess.CPU, 1))s RAM=$([math]::Round($TrainingProcess.WorkingSet64 / 1GB, 2))GB"
    } else {
        "process=stopped"
    }

    $gpuText = "GPU unavailable"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $reading = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($reading) {
            $parts = $reading -split ',' | ForEach-Object { $_.Trim() }
            $gpuText = "$($parts[0]) GPU=$($parts[1])% VRAM=$($parts[2])/$($parts[3])MiB temp=$($parts[4])C"
        }
    }

    $progress = "epoch not logged yet"
    $totalEpochs = "?"
    $totalSeeds = "?"
    $seedValue = $null
    if ($TrainingProcess) {
        $command = (Get-CimInstance Win32_Process -Filter "ProcessId=$($TrainingProcess.Id)" -ErrorAction SilentlyContinue).CommandLine
        if ($command -match '--epochs\s+(\d+)') { $totalEpochs = $Matches[1] }
        if ($command -match '--seeds\s+(\d+)') { $totalSeeds = $Matches[1] }
    }
    if (Test-Path -LiteralPath $Log) {
        $lastEpoch = Get-Content -LiteralPath $Log |
            Select-String -Pattern '\[seed (\d+)\] epoch (\d+):' |
            Select-Object -Last 1
        if ($lastEpoch -and $lastEpoch.Matches.Count) {
            $seedValue = $lastEpoch.Matches[0].Groups[1].Value
            $epochValue = $lastEpoch.Matches[0].Groups[2].Value
            $seenSeeds = @(Get-Content -LiteralPath $Log |
                Select-String -Pattern '\[seed (\d+)\]' |
                ForEach-Object { $_.Matches[0].Groups[1].Value } |
                Select-Object -Unique)
            $seedIndex = [array]::IndexOf($seenSeeds, $seedValue) + 1
            $progress = "seed $seedIndex/$totalSeeds ($seedValue) | epoch $epochValue/$totalEpochs"
        }
    }

    $checkpointText = "checkpoint=none"
    $checkpoint = if ($seedValue) {
        Get-Item -LiteralPath (Join-Path $CheckpointDir "seed_${seedValue}_best.pt") -ErrorAction SilentlyContinue
    } else {
        Get-ChildItem -LiteralPath $CheckpointDir -Filter "seed_*_best.pt" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    if ($checkpoint) {
        $checkpointText = "checkpoint=$($checkpoint.Name) size=$([math]::Round($checkpoint.Length / 1MB, 1))MB"
    }

    Write-Host "[$stamp] $progress | $processText | $gpuText | $checkpointText" -ForegroundColor Cyan
}

Write-Host "Watching NLP v3. Ctrl+C stops this viewer; training keeps running." -ForegroundColor Green
Write-Host "Log: $Log"
Write-Host "Error log: $ErrorLog"

while ($true) {
    Write-NewLines -Path $Log -Seen ([ref]$stdoutLines) -Color Gray
    Write-NewLines -Path $ErrorLog -Seen ([ref]$stderrLines) -Color Red

    $trainingProcess = Find-TrainingProcess -PreferredId $ProcessId
    $now = Get-Date
    if ($Once -or ($now - $lastStatus).TotalSeconds -ge $StatusSeconds) {
        Write-Status -TrainingProcess $trainingProcess
        $lastStatus = $now
    }

    if ($Once) { break }
    if (-not $trainingProcess) {
        Write-NewLines -Path $Log -Seen ([ref]$stdoutLines) -Color Gray
        Write-NewLines -Path $ErrorLog -Seen ([ref]$stderrLines) -Color Red
        Write-Host "Training process has finished." -ForegroundColor Yellow
        if (Test-Path -LiteralPath "D:\phenikaa\artifacts\nlp\neural_parser_v3_resume.pt") {
            Write-Host "Artifact: D:\phenikaa\artifacts\nlp\neural_parser_v3_resume.pt" -ForegroundColor Green
        }
        if (Test-Path -LiteralPath "D:\phenikaa\results\nlp_neural_v2\report.json") {
            Write-Host "Report: D:\phenikaa\results\nlp_neural_v2\report.json" -ForegroundColor Green
        }
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
