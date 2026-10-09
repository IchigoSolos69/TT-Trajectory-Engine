param([switch]$SkipNativeBuild)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location $projectRoot
try {
    if(!$SkipNativeBuild){
        & cmake --build backend_cpp/build --config Release --target tt_intelligence_candidate -j 4
        if($LASTEXITCODE -ne 0){throw 'Native desktop build failed'}
    }
    Push-Location (Join-Path $projectRoot 'frontend')
    $previousBuild=$env:TT_DESKTOP_BUILD;$previousTelemetry=$env:NEXT_TELEMETRY_DISABLED
    try {
        $env:TT_DESKTOP_BUILD='1';$env:NEXT_TELEMETRY_DISABLED='1'
        & npm.cmd run build
        if($LASTEXITCODE -ne 0){throw 'Desktop frontend build failed'}
        Copy-Item -LiteralPath '.next-desktop/static' -Destination '.next-desktop/standalone/.next-desktop/static' -Recurse -Force
        & node -e "require('node:fs').cpSync('public','.next-desktop/standalone/public',{recursive:true,filter:file=>!/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file)})"
        if($LASTEXITCODE -ne 0){throw 'Desktop public asset copy failed'}
    } finally {$env:TT_DESKTOP_BUILD=$previousBuild;$env:NEXT_TELEMETRY_DISABLED=$previousTelemetry;Pop-Location}
    & npm.cmd --prefix desktop ci --no-audit --no-fund
    if($LASTEXITCODE -ne 0){throw 'Desktop dependency install failed'}
    & node desktop/scripts/package.cjs
    if($LASTEXITCODE -ne 0){throw 'Desktop packaging failed'}
} finally {Pop-Location}
