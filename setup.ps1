param([string]$AssetsDirectory = '', [string]$ModelsFrom = '', [switch]$Offline)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($AssetsDirectory -and $ModelsFrom) { throw 'Choose either -AssetsDirectory or -ModelsFrom.' }
function Run-Python {
    param([string[]]$Arguments)
    & '.\.venv\Scripts\python.exe' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed: $Arguments" }
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64 with the Windows py launcher first.' }
}
if ($Offline) {
    Run-Python -Arguments @('-m','pip','install','--no-index','--find-links','wheelhouse','-r','requirements.e5.lock.txt')
} else {
    Run-Python -Arguments @('-m','pip','install','--extra-index-url','https://download.pytorch.org/whl/cu126','-r','requirements.e5.lock.txt')
}
if ($Offline -and -not ($AssetsDirectory -or $ModelsFrom)) { throw '-Offline requires -AssetsDirectory or -ModelsFrom.' }
if ($ModelsFrom) {
    Run-Python -Arguments @('deployment/install_assets.py','--models-from',$ModelsFrom)
} elseif ($AssetsDirectory) {
    Run-Python -Arguments @('deployment/install_assets.py','--from-dir',$AssetsDirectory)
} else {
    Run-Python -Arguments @('deployment/install_assets.py')
}
Run-Python -Arguments @('deployment/initialize_runtime.py')
Run-Python -Arguments @('deployment/check_runtime.py')
Write-Host 'Setup complete. Run: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-demo.ps1'
