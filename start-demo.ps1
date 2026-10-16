$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' 'deployment/check_runtime.py'
if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed. Run setup.ps1 first.' }
& '.\.venv\Scripts\python.exe' 'scripts/demo.py'
if ($LASTEXITCODE -ne 0) { throw 'Demo exited with an error.' }
