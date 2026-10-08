$ErrorActionPreference='Stop'
$launcher=Join-Path $PSScriptRoot 'launch_services.ps1'
$argsBase=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$launcher,'-BackendPort','9122','-FrontendPort','3122','-NoBrowser')
function Status { $result=& powershell @argsBase -StatusOnly; if($LASTEXITCODE){throw 'Status inspection failed'}; return @($result|ConvertFrom-Json) }
function Run-Launcher([string]$Extra){
    $run=Start-Process powershell -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "'+$launcher+'" -BackendPort 9122 -FrontendPort 3122 -NoBrowser -Mode live '+$Extra) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot '../artifacts/launcher_test_command.log') -RedirectStandardError (Join-Path $PSScriptRoot '../artifacts/launcher_test_command_errors.log')
    # Wait for the launcher itself, not the intentionally long-lived services.
    if(!$run.WaitForExit(60000)){throw 'Launcher process timed out'}
    if($run.ExitCode){throw 'Launcher failed; see launcher_test_command_errors.log'}
}
$before=Status
if(@($before|Where-Object State -ne HEALTHY).Count){throw 'Start the isolated live launcher fixture before this test'}
try {
    Run-Launcher ''
    $reused=Status
    if(($before.ProcessId -join ',') -ne ($reused.ProcessId -join ',')){throw 'Reuse replaced a healthy process'}
    Run-Launcher '-Restart'
    $restarted=Status
    Write-Host 'Restart health verified'
    if($restarted[0].ProcessId -eq $before[0].ProcessId -or $restarted[1].ProcessId -eq $before[1].ProcessId){throw 'Restart did not replace both owned services'}
    $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,9123)
    $listener.Start()
    Write-Host 'Unresponsive port fixture started'
    try {
        $conflictArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$launcher,'-BackendPort','9123','-FrontendPort','3122','-NoBrowser')
        $conflict=& powershell @conflictArgs -StatusOnly | ConvertFrom-Json
        Write-Host 'Unrelated owner inspected'
        if($LASTEXITCODE -or $conflict[0].State -ne 'UNRELATED' -or $conflict[0].ProcessId -ne $PID){throw 'Unrelated process was misidentified'}
        # A real launch must refuse the conflict, including when restart is requested.
        $attempt=Start-Process powershell -ArgumentList ('-NoProfile -File "'+$launcher+'" -BackendPort 9123 -FrontendPort 3122 -NoBrowser -Restart') -WindowStyle Hidden -PassThru -Wait -RedirectStandardError (Join-Path $PSScriptRoot '../artifacts/launcher_conflict.log')
        Write-Host 'Conflict launch returned'
        if($attempt.ExitCode -eq 0 -or !$listener.Server.IsBound){throw 'Unrelated owner was not preserved'}
    } finally {$listener.Stop()}
    @{passed=$true;before=$before;reused=$reused;restarted=$restarted;checks=@('healthy live standby reused','explicit owned restart','unrelated port owner preserved under restart','camera never authorized')}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $PSScriptRoot '../artifacts/launcher_runtime.json')
    Write-Host 'PASS: launcher reuse, owned restart, unrelated port protection'
} finally {
    $current=Status
    foreach($service in $current){
        if($service.State -eq 'HEALTHY'){
            $process=Get-CimInstance Win32_Process -Filter "ProcessId=$($service.ProcessId)"
            if($process.CreationDate.ToUniversalTime().Ticks.ToString() -eq $service.CreationTicks){Stop-Process -Id $service.ProcessId -Force}
        }
    }
    if(@((Status)|Where-Object State -ne FREE).Count){throw 'Test-service cleanup failed'}
}
