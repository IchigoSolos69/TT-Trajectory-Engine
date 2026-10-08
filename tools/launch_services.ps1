param([ValidateSet('simulation','live')][string]$Mode='simulation', [switch]$Restart, [switch]$StatusOnly,
    [ValidateRange(1024,65535)][int]$BackendPort=9002, [ValidateRange(1024,65535)][int]$FrontendPort=3000,
    [switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$backendPath=Join-Path $projectRoot 'backend_cpp\build\Release\tt_trajectory_engine.exe'
$nextPath=Join-Path $projectRoot 'frontend\node_modules\next\dist\bin\next'
$probePath=Join-Path $PSScriptRoot 'service_probe.cjs'

function Get-ServiceState([int]$Port,[string]$Kind) {
    # Inspection failure must never be interpreted as a free port.
    $listener=Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object LocalPort -eq $Port | Select-Object -First 1
    if(!$listener){return [pscustomobject]@{Port=$Port;Kind=$Kind;State='FREE';ProcessId=0;Mode='';Executable=''}}
    $owner=Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $health=$null
    try { $response=& node $probePath $Kind $Port 2>$null; if($LASTEXITCODE -eq 0){$health=$response|ConvertFrom-Json} } catch { $health=$null }
    $expected=if($Kind -eq 'backend'){'TT_TRAJECTORY_ENGINE'}else{'TT_TRAJECTORY_DASHBOARD'}
    $healthy=$health -and $health.application -eq $expected -and $health.pid -eq $owner.ProcessId -and $health.root.TrimEnd('\') -eq $projectRoot
    # A stale binary can be owned without satisfying the health protocol. Never
    # infer ownership of arbitrary node.exe processes from the executable name.
    $owned=$healthy -or ($Kind -eq 'backend' -and $owner.ExecutablePath -eq $backendPath) -or
        ($Kind -eq 'frontend' -and $owner.CommandLine -and $owner.CommandLine.Contains($nextPath))
    $state=if($healthy){'HEALTHY'}elseif($owned){'PROJECT_UNHEALTHY'}else{'UNRELATED'}
    return [pscustomobject]@{Port=$Port;Kind=$Kind;State=$state;ProcessId=$owner.ProcessId;Mode=if($healthy -and $Kind -eq 'backend'){$health.mode}else{''};Executable=$owner.ExecutablePath;CreationTicks=$owner.CreationDate.ToUniversalTime().Ticks.ToString()}
}
function Stop-OwnedService($Service){
    $current=Get-ServiceState $Service.Port $Service.Kind
    if($current.State -eq 'FREE'){return}
    if($current.State -eq 'UNRELATED' -or $current.ProcessId -ne $Service.ProcessId -or $current.CreationTicks -ne $Service.CreationTicks){throw 'Service ownership changed; no process stopped.'}
    if($current.Kind -eq 'backend' -and $current.State -eq 'HEALTHY'){
        & node $probePath shutdown $current.Port | Out-Null
        for($attempt=0;$attempt -lt 40;$attempt++){if(!(Get-Process -Id $current.ProcessId -ErrorAction SilentlyContinue)){return};Start-Sleep -Milliseconds 100}
    }
    $same=Get-CimInstance Win32_Process -Filter "ProcessId=$($current.ProcessId)"
    if($same -and $same.CreationDate.ToUniversalTime().Ticks.ToString() -eq $current.CreationTicks){Stop-Process -Id $current.ProcessId -Force}
}

$backend=Get-ServiceState $BackendPort backend
$frontend=Get-ServiceState $FrontendPort frontend
if($StatusOnly){@($backend,$frontend)|ConvertTo-Json;exit 0}
foreach($service in @($backend,$frontend)){
    if($service.State -eq 'UNRELATED'){throw "Port $($service.Port) belongs to an unverified service, PID $($service.ProcessId), $($service.Executable). It was not stopped."}
}
$needsRestart=$backend.State -eq 'PROJECT_UNHEALTHY' -or $frontend.State -eq 'PROJECT_UNHEALTHY' -or ($backend.State -eq 'HEALTHY' -and $backend.Mode -ne $Mode)
if($needsRestart -and !$Restart){
    $answer=Read-Host 'Existing project services need a restart for this mode. Enter R to restart them, or C to cancel'
    if($answer -ne 'R'){exit 1};$Restart=$true
}
if($Restart){Stop-OwnedService $backend;Stop-OwnedService $frontend;$backend=Get-ServiceState $BackendPort backend;$frontend=Get-ServiceState $FrontendPort frontend}
$logs=Join-Path $projectRoot 'artifacts';[IO.Directory]::CreateDirectory($logs)|Out-Null
if($backend.State -eq 'FREE'){
    if(!(Test-Path -LiteralPath $backendPath)){throw 'Build the C++ backend before launching.'}
    $runMode=if($Mode -eq 'live'){'--live'}else{'--sim'}
    Start-Process -FilePath $backendPath -ArgumentList @($runMode,'--port',$BackendPort) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs "launcher_backend_$BackendPort.log") -RedirectStandardError (Join-Path $logs "launcher_backend_${BackendPort}_errors.log")|Out-Null
}
if($frontend.State -eq 'FREE'){
    $nodePath=(Get-Command node).Source
    $command=if(Test-Path -LiteralPath (Join-Path $projectRoot 'frontend\.next\BUILD_ID')){'start'}else{'dev'}
    Start-Process -FilePath $nodePath -ArgumentList @(('"'+$nextPath+'"'),$command,'-H','127.0.0.1','-p',$FrontendPort) -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs "launcher_frontend_$FrontendPort.log") -RedirectStandardError (Join-Path $logs "launcher_frontend_${FrontendPort}_errors.log")|Out-Null
}
for($attempt=0;$attempt -lt 30;$attempt++){
    $backend=Get-ServiceState $BackendPort backend;$frontend=Get-ServiceState $FrontendPort frontend
    if($backend.State -eq 'HEALTHY' -and $backend.Mode -eq $Mode -and $frontend.State -eq 'HEALTHY'){Write-Host 'Project services are ready. Reusing healthy services where available.';if(!$NoBrowser){Start-Process "http://localhost:$FrontendPort"};exit 0}
    Start-Sleep -Milliseconds 500
}
throw 'Services did not become healthy. See artifacts/launcher_* logs.'
