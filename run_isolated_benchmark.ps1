Write-Host "========================================================================="
Write-Host "  ISOLATED 5-PASS C++ BENCHMARK WITH PROCESS GUARD & TELEMETRY LOGGING"
Write-Host "========================================================================="

# 1. Process Guard: Check for concurrent heavy tasks
$competingProcs = Get-Process -Name "benchmark_realtime_loop", "python", "trtexec", "eval_bounce", "eval_real_trt_system", "tt_trajectory_engine" -ErrorAction SilentlyContinue
if ($competingProcs) {
    Write-Error "CRITICAL ERROR: Competing processes detected! Benchmark refused to prevent contamination."
    $competingProcs | Format-Table Id, ProcessName, CPU, WorkingSet
    exit 1
}

# Stop any old PowerShell background jobs
Get-Job | Stop-Job -ErrorAction SilentlyContinue
Get-Job | Remove-Job -ErrorAction SilentlyContinue

Write-Host "No competing processes found. System isolated."

$scriptDir = Get-Location
$gpuLogPath = Join-Path $scriptDir "gpu_log.csv"
$cpuLogPath = Join-Path $scriptDir "cpu_log.txt"

# 2. Start GPU telemetry logging (querying clocks, temp, power, throttle reasons)
Write-Host "Starting GPU telemetry logging to $gpuLogPath..."
$gpuProc = Start-Process -FilePath "nvidia-smi" -ArgumentList "--query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv -l 1" -RedirectStandardOutput $gpuLogPath -PassThru -NoNewWindow

# 3. Start CPU telemetry logging
Write-Host "Starting CPU telemetry logging to $cpuLogPath..."
$cpuProc = Start-Process -FilePath "powershell" -ArgumentList "-Command Get-Counter '\Processor Information(_Total)\% Processor Performance' -SampleInterval 1 | Out-File -FilePath '$cpuLogPath'" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# 4. Execute 5-pass benchmark on clean idle machine
Write-Host "Executing 5-pass benchmark on clean idle machine..."
.\backend_cpp\build\Release\benchmark_realtime_loop.exe models\yolov11_ttgames.engine

# 5. Stop telemetry logging processes
Write-Host "Benchmark complete. Stopping telemetry logging processes..."
if ($gpuProc -and -not $gpuProc.HasExited) { Stop-Process -Id $gpuProc.Id -Force -ErrorAction SilentlyContinue }
if ($cpuProc -and -not $cpuProc.HasExited) { Stop-Process -Id $cpuProc.Id -Force -ErrorAction SilentlyContinue }

Write-Host "Telemetry jobs stopped. Benchmark pipeline complete."
